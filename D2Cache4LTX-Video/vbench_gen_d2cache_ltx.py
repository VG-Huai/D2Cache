import torch
from diffusers import LTXPipeline
from diffusers.models.transformers import LTXVideoTransformer3DModel
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils import export_to_video
from typing import Any, Dict, Optional, Tuple
import numpy as np
import json
import os
import time

def read_prompt_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list


def d2cache_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        num_frames: int,
        height: int,
        width: int,
        rope_interpolation_scale: Optional[Tuple[float, float, float]] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> torch.Tensor:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        image_rotary_emb = self.rope(hidden_states, num_frames, height, width, rope_interpolation_scale)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        batch_size = hidden_states.size(0)
        hidden_states = self.proj_in(hidden_states)

        temb, embedded_timestep = self.time_embed(
            timestep.flatten(),
            batch_size=batch_size,
            hidden_dtype=hidden_states.dtype,
        )

        temb = temb.view(batch_size, -1, temb.size(-1))
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.size(-1))

        encoder_hidden_states = self.caption_projection(encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.size(-1))

        delta_delta_scale = 1.0

        if self.enable_d2cache:
            inp = hidden_states.clone()
            temb_ = temb.clone()
            inp = self.transformer_blocks[0].norm1(inp)
            num_ada_params = self.transformer_blocks[0].scale_shift_table.shape[0]
            ada_values = self.transformer_blocks[0].scale_shift_table[None, None] + temb_.reshape(batch_size, temb_.size(1), num_ada_params, -1)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_values.unbind(dim=2)
            modulated_inp = inp * (1 + scale_msa) + shift_msa
            if self.cnt == 0 or self.cnt == self.num_steps-1:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
                self.previous_calc_step = self.cnt
                self.previous_error_score = self.accumulated_rel_l1_distance
                if self.cnt == 0:
                    self.previous_error_score = 0
                    self.previous_delta2_residual = None
                    self.previous_residual = None
                    self.previous_calc_step = 0
            else: 
                coefficients = [2.14700694e+01, -1.28016453e+01,  2.31279151e+00,  7.92487521e-01, 9.69274326e-03]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                    delta_delta_scale = self.accumulated_rel_l1_distance / (self.previous_error_score + 1e-8) * delta_delta_scale
                else:
                    should_calc = True
                    self.previous_error_score = self.accumulated_rel_l1_distance
                    self.accumulated_rel_l1_distance = 0
                    self.previous_calc_step = self.cnt
            self.previous_modulated_input = modulated_inp  
            self.cnt += 1
            if self.cnt == self.num_steps:
                self.cnt = 0         
        
        if self.enable_d2cache:
            if not should_calc:
                hidden_states += self.previous_residual
                if self.previous_delta2_residual is not None:
                    hidden_states += self.previous_delta2_residual * delta_delta_scale
            else:
                ori_hidden_states = hidden_states.clone()
                for block in self.transformer_blocks:
                    if torch.is_grad_enabled() and self.gradient_checkpointing:

                        def create_custom_forward(module, return_dict=None):
                            def custom_forward(*inputs):
                                if return_dict is not None:
                                    return module(*inputs, return_dict=return_dict)
                                else:
                                    return module(*inputs)

                            return custom_forward

                        ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            hidden_states,
                            encoder_hidden_states,
                            temb,
                            image_rotary_emb,
                            encoder_attention_mask,
                            **ckpt_kwargs,
                        )
                    else:
                        hidden_states = block(
                            hidden_states=hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=temb,
                            image_rotary_emb=image_rotary_emb,
                            encoder_attention_mask=encoder_attention_mask,
                        )

                scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
                shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

                hidden_states = self.norm_out(hidden_states)
                hidden_states = hidden_states * (1 + scale) + shift
                new_resuidual = hidden_states - ori_hidden_states
                if self.previous_residual is not None:
                    self.previous_delta2_residual = new_resuidual - self.previous_residual
                self.previous_residual = new_resuidual
        else:
            for block in self.transformer_blocks:
                if torch.is_grad_enabled() and self.gradient_checkpointing:

                    def create_custom_forward(module, return_dict=None):
                        def custom_forward(*inputs):
                            if return_dict is not None:
                                return module(*inputs, return_dict=return_dict)
                            else:
                                return module(*inputs)

                        return custom_forward

                    ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        hidden_states,
                        encoder_hidden_states,
                        temb,
                        image_rotary_emb,
                        encoder_attention_mask,
                        **ckpt_kwargs,
                    )
                else:
                    hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=temb,
                        image_rotary_emb=image_rotary_emb,
                        encoder_attention_mask=encoder_attention_mask,
                    )

            scale_shift_values = self.scale_shift_table[None, None] + embedded_timestep[:, :, None]
            shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

            hidden_states = self.norm_out(hidden_states)
            hidden_states = hidden_states * (1 + scale) + shift


        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)


def eval_base(prompt_list, loop=5, base_seed=1):
    LTXVideoTransformer3DModel.forward = d2cache_forward
    
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    num_inference_steps = 50
    pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)

    # d2cache
    pipe.transformer.__class__.enable_d2cache = True
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = num_inference_steps
    pipe.transformer.__class__.rel_l1_thresh = 0.00 # base; no speedup
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.previous_delta2_residual = None

    output_dir = "./ltx_base"
    os.makedirs(output_dir, exist_ok=True)

    gen_cnt = 0

    pipe.to("cuda")
    for i in range(loop):
        seed = base_seed + i
        for prompt in prompt_list:
            save_path = f"{output_dir}/{prompt}-{i}.mp4"
            if os.path.exists(save_path):
                print(f"Skip existing: {save_path}")
                gen_cnt += 1
                continue
            start = time.time()
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=768,
                height=512,
                num_frames=161,
                decode_timestep=0.03,
                decode_noise_scale=0.025,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cuda").manual_seed(seed)
            ).frames[0]

            end = time.time()
            print(f"Time taken for generation: {end - start} seconds")

            export_to_video(video, save_path, fps=24)

            gen_cnt += 1
            print(f"Generated {gen_cnt}: {save_path}")

            # refresh 
            pipe.transformer.__class__.cnt = 0
            pipe.transformer.__class__.accumulated_rel_l1_distance = 0
            pipe.transformer.__class__.previous_modulated_input = None
            pipe.transformer.__class__.previous_residual = None
            pipe.transformer.__class__.previous_delta2_residual = None

            # clear cache
            torch.cuda.empty_cache()

def eval_d2cache_slow(prompt_list, loop=5, base_seed=1):
    LTXVideoTransformer3DModel.forward = d2cache_forward
    
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    num_inference_steps = 50
    pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)

    # d2cache
    pipe.transformer.__class__.enable_d2cache = True
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = num_inference_steps
    pipe.transformer.__class__.rel_l1_thresh = 0.05 # 2 speedup
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.previous_delta2_residual = None

    output_dir = "./ltx_d2cache_slow"
    os.makedirs(output_dir, exist_ok=True)

    gen_cnt = 0

    pipe.to("cuda")
    for i in range(loop):
        seed = base_seed + i
        for prompt in prompt_list:
            save_path = f"{output_dir}/{prompt}-{i}.mp4"
            if os.path.exists(save_path):
                print(f"Skip existing: {save_path}")
                gen_cnt += 1
                continue
            start = time.time()
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=768,
                height=512,
                num_frames=161,
                decode_timestep=0.03,
                decode_noise_scale=0.025,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cuda").manual_seed(seed)
            ).frames[0]
            end = time.time()
            print(f"Time taken for generation: {end - start} seconds")
            export_to_video(video, save_path, fps=24)

            gen_cnt += 1
            print(f"Generated {gen_cnt}: {save_path}")

            # refresh 
            pipe.transformer.__class__.cnt = 0
            pipe.transformer.__class__.accumulated_rel_l1_distance = 0
            pipe.transformer.__class__.previous_modulated_input = None
            pipe.transformer.__class__.previous_residual = None
            pipe.transformer.__class__.previous_delta2_residual = None

            # clear cache
            torch.cuda.empty_cache()

def eval_d2cache_fast(prompt_list, loop=5, base_seed=1):
    LTXVideoTransformer3DModel.forward = d2cache_forward
    
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    num_inference_steps = 50
    pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)

    # d2cache
    pipe.transformer.__class__.enable_d2cache = True
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = num_inference_steps
    pipe.transformer.__class__.rel_l1_thresh = 0.10
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.previous_delta2_residual = None
    output_dir = "./ltx_d2cache_fast"
    os.makedirs(output_dir, exist_ok=True)
    gen_cnt = 0
    pipe.to("cuda")
    for i in range(loop):
        seed = base_seed + i
        for prompt in prompt_list:
            save_path = f"{output_dir}/{prompt}-{i}.mp4"
            if os.path.exists(save_path):
                print(f"Skip existing: {save_path}")
                gen_cnt += 1
                continue
            start = time.time()
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=768,
                height=512,
                num_frames=161,
                decode_timestep=0.03,
                decode_noise_scale=0.025,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cuda").manual_seed(seed)
            ).frames[0]
            end = time.time()
            print(f"Time taken for generation: {end - start} seconds")

            export_to_video(video, save_path, fps=24)

            gen_cnt += 1
            print(f"Generated {gen_cnt}: {save_path}")

            # refresh 
            pipe.transformer.__class__.cnt = 0
            pipe.transformer.__class__.accumulated_rel_l1_distance = 0
            pipe.transformer.__class__.previous_modulated_input = None
            pipe.transformer.__class__.previous_residual = None
            pipe.transformer.__class__.previous_delta2_residual = None
            # clear cache
            torch.cuda.empty_cache()

def eval_d2cache_superfast(prompt_list, loop=5, base_seed=1):
    LTXVideoTransformer3DModel.forward = d2cache_forward
    
    negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"

    num_inference_steps = 50
    pipe = LTXPipeline.from_pretrained("a-r-r-o-w/LTX-Video-0.9.1-diffusers", torch_dtype=torch.bfloat16)

    # d2cache
    pipe.transformer.__class__.enable_d2cache = True
    pipe.transformer.__class__.cnt = 0
    pipe.transformer.__class__.num_steps = num_inference_steps
    pipe.transformer.__class__.rel_l1_thresh = 0.15
    pipe.transformer.__class__.accumulated_rel_l1_distance = 0
    pipe.transformer.__class__.previous_modulated_input = None
    pipe.transformer.__class__.previous_residual = None
    pipe.transformer.__class__.previous_delta2_residual = None
    output_dir = "./ltx_d2cache_superfast"
    os.makedirs(output_dir, exist_ok=True)
    gen_cnt = 0
    pipe.to("cuda")
    for i in range(loop):
        seed = base_seed + i
        for prompt in prompt_list:
            save_path = f"{output_dir}/{prompt}-{i}.mp4"
            if os.path.exists(save_path):
                print(f"Skip existing: {save_path}")
                gen_cnt += 1
                continue
            start = time.time()
            video = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=768,
                height=512,
                num_frames=161,
                decode_timestep=0.03,
                decode_noise_scale=0.025,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cuda").manual_seed(seed)
            ).frames[0]
            end = time.time()
            print(f"Time taken for generation: {end - start} seconds")

            export_to_video(video, save_path, fps=24)

            gen_cnt += 1
            print(f"Generated {gen_cnt}: {save_path}")

            # refresh 
            pipe.transformer.__class__.cnt = 0
            pipe.transformer.__class__.accumulated_rel_l1_distance = 0
            pipe.transformer.__class__.previous_modulated_input = None
            pipe.transformer.__class__.previous_residual = None
            pipe.transformer.__class__.previous_delta2_residual = None
            # clear cache
            torch.cuda.empty_cache()


if __name__ == "__main__":
    vbench_json = '../eval/teacache/vbench/VBench_full_info.json'
    prompt_list = read_prompt_list(vbench_json)

    eval_base(prompt_list, loop=5, base_seed=1)
    eval_d2cache_slow(prompt_list, loop=5, base_seed=1)
    eval_d2cache_fast(prompt_list, loop=5, base_seed=1)
    eval_d2cache_superfast(prompt_list, loop=5, base_seed=1)

