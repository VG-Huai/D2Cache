from utils import generate_func, read_prompt_list
from videosys import OpenSoraConfig, VideoSysEngine
import torch
from einops import rearrange
from videosys.models.transformers.open_sora_transformer_3d import t2i_modulate, auto_grad_checkpoint
from videosys.core.comm import all_to_all_with_pad, gather_sequence, get_pad, set_pad, split_sequence
import numpy as np
from videosys.utils.utils import batch_func
from functools import partial

def teacache_forward(
        self, x, timestep, all_timesteps, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs
    ):
        # === Split batch ===
        if self.parallel_manager.cp_size > 1:
            x, timestep, y, x_mask, mask = batch_func(
                partial(split_sequence, process_group=self.parallel_manager.cp_group, dim=0),
                x,
                timestep,
                y,
                x_mask,
                mask,
            )

        dtype = self.x_embedder.proj.weight.dtype
        B = x.size(0)
        x = x.to(dtype)
        timestep = timestep.to(dtype)
        y = y.to(dtype)

        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)

        # === get y embed ===
        if self.config.skip_y_embedder:
            y_lens = mask
            if isinstance(y_lens, torch.Tensor):
                y_lens = y_lens.long().tolist()
        else:
            y, y_lens = self.encode_text(y, mask)

        # === get x embed ===
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb

        if self.enable_teacache:
            inp = x.clone()
            inp = rearrange(inp, "B T S C -> B (T S) C", T=T, S=S)
            B, N, C = inp.shape
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.spatial_blocks[0].scale_shift_table[None] + t_mlp.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            modulated_inp = t2i_modulate(self.spatial_blocks[0].norm1(inp), shift_msa, scale_msa)
            if timestep[0]  == all_timesteps[0] or timestep[0]  == all_timesteps[-1]:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:       
                coefficients = [2.17546007e+02, -1.18329252e+02,  2.68662585e+01, -4.59364272e-02, 4.84426240e-02]
                rescale_func = np.poly1d(coefficients) 
                self.accumulated_rel_l1_distance +=  rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp

        # === blocks ===
        if self.enable_teacache:
            if not should_calc:
                x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
                x += self.previous_residual
            else:
                # shard over the sequence dim if sp is enabled
                if self.parallel_manager.sp_size > 1:
                    set_pad("temporal", T, self.parallel_manager.sp_group)
                    set_pad("spatial", S, self.parallel_manager.sp_group)
                    x = split_sequence(x, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal"))
                    T = x.shape[1]
                    x_mask_org = x_mask
                    x_mask = split_sequence(
                        x_mask, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal")
                    )
                x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
                origin_x = x.clone().detach()
                for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
                    x = auto_grad_checkpoint(
                        spatial_block,
                        x,
                        y,
                        t_mlp,
                        y_lens,
                        x_mask,
                        t0_mlp,
                        T,
                        S,
                        timestep,
                        all_timesteps=all_timesteps,
                    )

                    x = auto_grad_checkpoint(
                        temporal_block,
                        x,
                        y,
                        t_mlp,
                        y_lens,
                        x_mask,
                        t0_mlp,
                        T,
                        S,
                        timestep,
                        all_timesteps=all_timesteps,
                    )
                self.previous_residual = x - origin_x
        else:
            # shard over the sequence dim if sp is enabled
            if self.parallel_manager.sp_size > 1:
                set_pad("temporal", T, self.parallel_manager.sp_group)
                set_pad("spatial", S, self.parallel_manager.sp_group)
                x = split_sequence(x, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal"))
                T = x.shape[1]
                x_mask_org = x_mask
                x_mask = split_sequence(
                    x_mask, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal")
                )
            x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)

            for spatial_block, temporal_block in zip(self.spatial_blocks, self.temporal_blocks):
                x = auto_grad_checkpoint(
                    spatial_block,
                    x,
                    y,
                    t_mlp,
                    y_lens,
                    x_mask,
                    t0_mlp,
                    T,
                    S,
                    timestep,
                    all_timesteps=all_timesteps,
                )

                x = auto_grad_checkpoint(
                    temporal_block,
                    x,
                    y,
                    t_mlp,
                    y_lens,
                    x_mask,
                    t0_mlp,
                    T,
                    S,
                    timestep,
                    all_timesteps=all_timesteps,
                )

        if self.parallel_manager.sp_size > 1:
            if self.enable_teacache:
                if should_calc:
                    x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
                    self.previous_residual = rearrange(self.previous_residual, "B (T S) C -> B T S C", T=T, S=S)
                    x = gather_sequence(x, self.parallel_manager.sp_group, dim=1, grad_scale="up", pad=get_pad("temporal"))
                    self.previous_residual = gather_sequence(self.previous_residual, self.parallel_manager.sp_group, dim=1, grad_scale="up", pad=get_pad("temporal"))
                    T, S = x.shape[1], x.shape[2]
                    x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
                    self.previous_residual = rearrange(self.previous_residual, "B T S C -> B (T S) C", T=T, S=S)
                    x_mask = x_mask_org                                 
            else:
                x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
                x = gather_sequence(x, self.parallel_manager.sp_group, dim=1, grad_scale="up", pad=get_pad("temporal"))
                T, S = x.shape[1], x.shape[2]
                x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
                x_mask = x_mask_org
        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)

        # === Gather Output ===
        if self.parallel_manager.cp_size > 1:
            x = gather_sequence(x, self.parallel_manager.cp_group, dim=0)

        return x

def eval_base(prompt_list):
    config = OpenSoraConfig()
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensora_base", loop=5)

def eval_teacache_slow(prompt_list):
    config = OpenSoraConfig()
    engine = VideoSysEngine(config)
    engine.driver_worker.transformer.__class__.enable_teacache = True
    engine.driver_worker.transformer.__class__.rel_l1_thresh = 0.1
    engine.driver_worker.transformer.__class__.accumulated_rel_l1_distance = 0
    engine.driver_worker.transformer.__class__.previous_modulated_input = None
    engine.driver_worker.transformer.__class__.previous_residual = None
    engine.driver_worker.transformer.__class__.forward = teacache_forward
    generate_func(engine, prompt_list, "./samples/opensora_teacache_slow", loop=5)

def eval_teacache_fast(prompt_list):
    config = OpenSoraConfig()
    engine = VideoSysEngine(config)
    engine.driver_worker.transformer.__class__.enable_teacache = True
    engine.driver_worker.transformer.__class__.rel_l1_thresh = 0.2
    engine.driver_worker.transformer.__class__.accumulated_rel_l1_distance = 0
    engine.driver_worker.transformer.__class__.previous_modulated_input = None
    engine.driver_worker.transformer.__class__.previous_residual = None
    engine.driver_worker.transformer.__class__.forward = teacache_forward
    generate_func(engine, prompt_list, "./samples/opensora_teacache_fast", loop=5)

def eval_teacache_superfast(prompt_list):
    config = OpenSoraConfig()
    engine = VideoSysEngine(config)
    engine.driver_worker.transformer.__class__.enable_teacache = True
    engine.driver_worker.transformer.__class__.rel_l1_thresh = 0.3
    engine.driver_worker.transformer.__class__.accumulated_rel_l1_distance = 0
    engine.driver_worker.transformer.__class__.previous_modulated_input = None
    engine.driver_worker.transformer.__class__.previous_residual = None
    engine.driver_worker.transformer.__class__.forward = teacache_forward
    generate_func(engine, prompt_list, "./samples/opensora_teacache_superfast", loop=5)

if __name__ == "__main__":
    prompt_list = read_prompt_list("vbench/VBench_full_info.json")
    prompt_list = ['AAA fantasy landscape with castles and dragons']
    # eval_base(prompt_list)
    # eval_teacache_slow(prompt_list)
    eval_teacache_superfast(prompt_list)
    # eval_teacache_fast(prompt_list)
    