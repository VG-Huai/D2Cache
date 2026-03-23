from utils import generate_func, read_prompt_list
from videosys import LatteConfig, VideoSysEngine
import torch
from einops import rearrange, repeat
from torch import nn
import numpy as np
from typing import Any, Dict, Optional, Tuple
from videosys.core.comm import all_to_all_with_pad, gather_sequence, get_pad, set_pad, split_sequence
from videosys.models.transformers.latte_transformer_3d import Transformer3DModelOutput
from videosys.utils.utils import batch_func
from functools import partial

def teacache_forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        all_timesteps=None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_image_num: int = 0,
        enable_temporal_attentions: bool = True,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, frame, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        # 0. Split batch for data parallelism
        if self.parallel_manager.cp_size > 1:
            (
                hidden_states,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs,
                class_labels,
                attention_mask,
                encoder_attention_mask,
            ) = batch_func(
                partial(split_sequence, process_group=self.parallel_manager.cp_group, dim=0),
                hidden_states,
                timestep,
                encoder_hidden_states,
                added_cond_kwargs,
                class_labels,
                attention_mask,
                encoder_attention_mask,
            )

        input_batch_size, c, frame, h, w = hidden_states.shape
        frame = frame - use_image_num
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w").contiguous()
        org_timestep = timestep

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:  # ndim == 2 means no image joint
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)
            encoder_attention_mask = repeat(encoder_attention_mask, "b 1 l -> (b f) 1 l", f=frame).contiguous()
        elif encoder_attention_mask is not None and encoder_attention_mask.ndim == 3:  # ndim == 3 means image joint
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask_video = encoder_attention_mask[:, :1, ...]
            encoder_attention_mask_video = repeat(
                encoder_attention_mask_video, "b 1 l -> b (1 f) l", f=frame
            ).contiguous()
            encoder_attention_mask_image = encoder_attention_mask[:, 1:, ...]
            encoder_attention_mask = torch.cat([encoder_attention_mask_video, encoder_attention_mask_image], dim=1)
            encoder_attention_mask = rearrange(encoder_attention_mask, "b n l -> (b n) l").contiguous().unsqueeze(1)

        # Retrieve lora scale.
        cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_patches:  # here
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            num_patches = height * width

            hidden_states = self.pos_embed(hidden_states)  # alrady add positional embeddings

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                # batch_size = hidden_states.shape[0]
                batch_size = input_batch_size
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

        # 2. Blocks
        if self.caption_projection is not None:
            batch_size = hidden_states.shape[0]
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)  # 3 120 1152

            if use_image_num != 0 and self.training:
                encoder_hidden_states_video = encoder_hidden_states[:, :1, ...]
                encoder_hidden_states_video = repeat(
                    encoder_hidden_states_video, "b 1 t d -> b (1 f) t d", f=frame
                ).contiguous()
                encoder_hidden_states_image = encoder_hidden_states[:, 1:, ...]
                encoder_hidden_states = torch.cat([encoder_hidden_states_video, encoder_hidden_states_image], dim=1)
                encoder_hidden_states_spatial = rearrange(encoder_hidden_states, "b f t d -> (b f) t d").contiguous()
            else:
                encoder_hidden_states_spatial = repeat(
                    encoder_hidden_states, "b t d -> (b f) t d", f=frame
                ).contiguous()

        # prepare timesteps for spatial and temporal block
        timestep_spatial = repeat(timestep, "b d -> (b f) d", f=frame + use_image_num).contiguous()
        timestep_temp = repeat(timestep, "b d -> (b p) d", p=num_patches).contiguous()

        if self.enable_teacache:
            inp = hidden_states.clone()
            batch_size = inp.shape[0]
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.transformer_blocks[0].scale_shift_table[None] + timestep_spatial.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            modulated_inp = self.transformer_blocks[0].norm1(inp) * (1 + scale_msa) + shift_msa
            if org_timestep[0]  == all_timesteps[0] or org_timestep[0]  == all_timesteps[-1]:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
            else:  
                coefficients = [-2.46434137e+03,  3.08044764e+02,  8.07447667e+01, -4.11385132e+00, 1.11001402e-01]
                rescale_func = np.poly1d(coefficients)
                self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
                if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                    should_calc = False
                else:
                    should_calc = True
                    self.accumulated_rel_l1_distance = 0
            self.previous_modulated_input = modulated_inp        

        if self.enable_teacache:
            if not should_calc:
                hidden_states += self.previous_residual
            else:
                if self.parallel_manager.sp_size > 1:
                    set_pad("temporal", frame + use_image_num, self.parallel_manager.sp_group)
                    set_pad("spatial", num_patches, self.parallel_manager.sp_group)
                    hidden_states = self.split_from_second_dim(hidden_states, input_batch_size)
                    encoder_hidden_states_spatial = self.split_from_second_dim(encoder_hidden_states_spatial, input_batch_size)
                    timestep_spatial = self.split_from_second_dim(timestep_spatial, input_batch_size)
                    temp_pos_embed = split_sequence(
                        self.temp_pos_embed, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal")
                    )
                else:
                    temp_pos_embed = self.temp_pos_embed

                hidden_states_origin = hidden_states.clone().detach()
                for i, (spatial_block, temp_block) in enumerate(zip(self.transformer_blocks, self.temporal_transformer_blocks)):
                    if self.training and self.gradient_checkpointing:
                        hidden_states = torch.utils.checkpoint.checkpoint(
                            spatial_block,
                            hidden_states,
                            attention_mask,
                            encoder_hidden_states_spatial,
                            encoder_attention_mask,
                            timestep_spatial,
                            cross_attention_kwargs,
                            class_labels,
                            use_reentrant=False,
                        )

                        if enable_temporal_attentions:
                            hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size).contiguous()

                            if use_image_num != 0:  # image-video joitn training
                                hidden_states_video = hidden_states[:, :frame, ...]
                                hidden_states_image = hidden_states[:, frame:, ...]

                                if i == 0:
                                    hidden_states_video = hidden_states_video + temp_pos_embed

                                hidden_states_video = torch.utils.checkpoint.checkpoint(
                                    temp_block,
                                    hidden_states_video,
                                    None,  # attention_mask
                                    None,  # encoder_hidden_states
                                    None,  # encoder_attention_mask
                                    timestep_temp,
                                    cross_attention_kwargs,
                                    class_labels,
                                    use_reentrant=False,
                                )

                                hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                                hidden_states = rearrange(
                                    hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                                ).contiguous()

                            else:
                                if i == 0:
                                    hidden_states = hidden_states + temp_pos_embed

                                hidden_states = torch.utils.checkpoint.checkpoint(
                                    temp_block,
                                    hidden_states,
                                    None,  # attention_mask
                                    None,  # encoder_hidden_states
                                    None,  # encoder_attention_mask
                                    timestep_temp,
                                    cross_attention_kwargs,
                                    class_labels,
                                    use_reentrant=False,
                                )

                                hidden_states = rearrange(
                                    hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                                ).contiguous()
                    else:
                        hidden_states = spatial_block(
                            hidden_states,
                            attention_mask,
                            encoder_hidden_states_spatial,
                            encoder_attention_mask,
                            timestep_spatial,
                            cross_attention_kwargs,
                            class_labels,
                            None,
                            org_timestep,
                            all_timesteps=all_timesteps,
                        )

                        if enable_temporal_attentions:
                            hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size).contiguous()

                            if use_image_num != 0 and self.training:
                                hidden_states_video = hidden_states[:, :frame, ...]
                                hidden_states_image = hidden_states[:, frame:, ...]

                                hidden_states_video = temp_block(
                                    hidden_states_video,
                                    None,  # attention_mask
                                    None,  # encoder_hidden_states
                                    None,  # encoder_attention_mask
                                    timestep_temp,
                                    cross_attention_kwargs,
                                    class_labels,
                                    org_timestep,
                                )

                                hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                                hidden_states = rearrange(
                                    hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                                ).contiguous()

                            else:
                                if i == 0 and frame > 1:
                                    hidden_states = hidden_states + temp_pos_embed
                                hidden_states = temp_block(
                                    hidden_states,
                                    None,  # attention_mask
                                    None,  # encoder_hidden_states
                                    None,  # encoder_attention_mask
                                    timestep_temp,
                                    cross_attention_kwargs,
                                    class_labels,
                                    org_timestep,
                                    all_timesteps=all_timesteps,
                                )

                                hidden_states = rearrange(
                                    hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                                ).contiguous()
                self.previous_residual = hidden_states - hidden_states_origin
        else:
            if self.parallel_manager.sp_size > 1:
                set_pad("temporal", frame + use_image_num, self.parallel_manager.sp_group)
                set_pad("spatial", num_patches, self.parallel_manager.sp_group)
                hidden_states = self.split_from_second_dim(hidden_states, input_batch_size)
                encoder_hidden_states_spatial = self.split_from_second_dim(encoder_hidden_states_spatial, input_batch_size)
                timestep_spatial = self.split_from_second_dim(timestep_spatial, input_batch_size)
                temp_pos_embed = split_sequence(
                    self.temp_pos_embed, self.parallel_manager.sp_group, dim=1, grad_scale="down", pad=get_pad("temporal")
                )
            else:
                temp_pos_embed = self.temp_pos_embed

            for i, (spatial_block, temp_block) in enumerate(zip(self.transformer_blocks, self.temporal_transformer_blocks)):
                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        spatial_block,
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states_spatial,
                        encoder_attention_mask,
                        timestep_spatial,
                        cross_attention_kwargs,
                        class_labels,
                        use_reentrant=False,
                    )

                    if enable_temporal_attentions:
                        hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size).contiguous()

                        if use_image_num != 0:  # image-video joitn training
                            hidden_states_video = hidden_states[:, :frame, ...]
                            hidden_states_image = hidden_states[:, frame:, ...]

                            if i == 0:
                                hidden_states_video = hidden_states_video + temp_pos_embed

                            hidden_states_video = torch.utils.checkpoint.checkpoint(
                                temp_block,
                                hidden_states_video,
                                None,  # attention_mask
                                None,  # encoder_hidden_states
                                None,  # encoder_attention_mask
                                timestep_temp,
                                cross_attention_kwargs,
                                class_labels,
                                use_reentrant=False,
                            )

                            hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                            hidden_states = rearrange(
                                hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                            ).contiguous()

                        else:
                            if i == 0:
                                hidden_states = hidden_states + temp_pos_embed

                            hidden_states = torch.utils.checkpoint.checkpoint(
                                temp_block,
                                hidden_states,
                                None,  # attention_mask
                                None,  # encoder_hidden_states
                                None,  # encoder_attention_mask
                                timestep_temp,
                                cross_attention_kwargs,
                                class_labels,
                                use_reentrant=False,
                            )

                            hidden_states = rearrange(
                                hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                            ).contiguous()
                else:
                    hidden_states = spatial_block(
                        hidden_states,
                        attention_mask,
                        encoder_hidden_states_spatial,
                        encoder_attention_mask,
                        timestep_spatial,
                        cross_attention_kwargs,
                        class_labels,
                        None,
                        org_timestep,
                        all_timesteps=all_timesteps,
                    )

                    if enable_temporal_attentions:
                        hidden_states = rearrange(hidden_states, "(b f) t d -> (b t) f d", b=input_batch_size).contiguous()

                        if use_image_num != 0 and self.training:
                            hidden_states_video = hidden_states[:, :frame, ...]
                            hidden_states_image = hidden_states[:, frame:, ...]

                            hidden_states_video = temp_block(
                                hidden_states_video,
                                None,  # attention_mask
                                None,  # encoder_hidden_states
                                None,  # encoder_attention_mask
                                timestep_temp,
                                cross_attention_kwargs,
                                class_labels,
                                org_timestep,
                            )

                            hidden_states = torch.cat([hidden_states_video, hidden_states_image], dim=1)
                            hidden_states = rearrange(
                                hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                            ).contiguous()

                        else:
                            if i == 0 and frame > 1:
                                hidden_states = hidden_states + temp_pos_embed
                            hidden_states = temp_block(
                                hidden_states,
                                None,  # attention_mask
                                None,  # encoder_hidden_states
                                None,  # encoder_attention_mask
                                timestep_temp,
                                cross_attention_kwargs,
                                class_labels,
                                org_timestep,
                                all_timesteps=all_timesteps,
                            )

                            hidden_states = rearrange(
                                hidden_states, "(b t) f d -> (b f) t d", b=input_batch_size
                            ).contiguous()


        if self.parallel_manager.sp_size > 1:
            if self.enable_teacache:
                if should_calc:
                    hidden_states = self.gather_from_second_dim(hidden_states, input_batch_size)
                    self.previous_residual = self.gather_from_second_dim(self.previous_residual, input_batch_size)
            else:
                hidden_states = self.gather_from_second_dim(hidden_states, input_batch_size)

        if self.is_input_patches:
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                embedded_timestep = repeat(embedded_timestep, "b d -> (b f) d", f=frame + use_image_num).contiguous()
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )
            output = rearrange(output, "(b f) c h w -> b c f h w", b=input_batch_size).contiguous()

        # 3. Gather batch for data parallelism
        if self.parallel_manager.cp_size > 1:
            output = gather_sequence(output, self.parallel_manager.cp_group, dim=0)

        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)



def eval_teacache_slow(prompt_list):
    config = LatteConfig()
    engine = VideoSysEngine(config)
    engine.driver_worker.transformer.__class__.enable_teacache = True
    engine.driver_worker.transformer.__class__.rel_l1_thresh = 0.1
    engine.driver_worker.transformer.__class__.accumulated_rel_l1_distance = 0
    engine.driver_worker.transformer.__class__.previous_modulated_input = None
    engine.driver_worker.transformer.__class__.previous_residual = None
    engine.driver_worker.transformer.__class__.forward = teacache_forward
    generate_func(engine, prompt_list, "./samples/latte_teacache_slow", loop=5)
    
def eval_teacache_fast(prompt_list):
    config = LatteConfig()
    engine = VideoSysEngine(config)
    engine.driver_worker.transformer.__class__.enable_teacache = True
    engine.driver_worker.transformer.__class__.rel_l1_thresh = 0.2
    engine.driver_worker.transformer.__class__.accumulated_rel_l1_distance = 0
    engine.driver_worker.transformer.__class__.previous_modulated_input = None
    engine.driver_worker.transformer.__class__.previous_residual = None
    engine.driver_worker.transformer.__class__.forward = teacache_forward
    generate_func(engine, prompt_list, "./samples/latte_teacache_fast", loop=5)
    
def eval_teacache_superfast(prompt_list):
    config = LatteConfig()
    engine = VideoSysEngine(config)
    engine.driver_worker.transformer.__class__.enable_teacache = True
    engine.driver_worker.transformer.__class__.rel_l1_thresh = 0.3
    engine.driver_worker.transformer.__class__.accumulated_rel_l1_distance = 0
    engine.driver_worker.transformer.__class__.previous_modulated_input = None
    engine.driver_worker.transformer.__class__.previous_residual = None
    engine.driver_worker.transformer.__class__.forward = teacache_forward
    generate_func(engine, prompt_list, "./samples/latte_teacache_superfast", loop=5)
    
def eval_pab(prompt_list):
    config = LatteConfig()
    config.enable_pab = True
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/latte_pab", loop=5)


def eval_base(prompt_list):
    config = LatteConfig()
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/latte_base", loop=5)



if __name__ == "__main__":
    prompt_list = read_prompt_list("vbench/VBench_full_info.json")
    prompt_list.reverse()
    
    # eval_pab(prompt_list)
    
    # eval_teacache_slow(prompt_list)
    # eval_base(prompt_list) 
    # eval_teacache_fast(prompt_list)
    eval_teacache_superfast(prompt_list)
    # eval_teacache_fast(prompt_list)