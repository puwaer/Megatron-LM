# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""FujiModel: GPT-style language model with mHC and Engram.

FujiModel extends GPTModel with two architectural improvements from DeepSeek-AI:

  1. **mHC (Manifold-Constrained Hyper-Connections)** — replaces standard
     single-stream residual connections with n parallel streams constrained via
     Sinkhorn-Knopp projection (arXiv:2512.24880).

  2. **Engram (Conditional Memory via Scalable Lookup)** — augments specific
     transformer layers with deterministic N-gram hash-based memory retrieval
     (arXiv:2601.07372).

Usage example (standalone)::

    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.fuji.fuji_model import FujiModel
    from megatron.core.models.fuji.fuji_layer_specs import get_fuji_layer_spec
    from megatron.core.models.engram.engram_module import EngramConfig
    from megatron.core.models.backends import LocalSpecProvider

    config = TransformerConfig(
        num_layers=12, hidden_size=768, num_attention_heads=12,
        use_mhc=True, mhc_num_streams=4,
        use_engram=True, engram_layer_ids=[0, 5],
    )
    engram_config = EngramConfig(
        max_ngram_size=3, n_embed_per_ngram=512, n_head_per_ngram=8,
        engram_layer_ids=[0, 5],
    )
    layer_spec = get_fuji_layer_spec(LocalSpecProvider())
    model = FujiModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=50257,
        max_sequence_length=2048,
        engram_config=engram_config,
    )
"""

from typing import Literal, Optional

from torch import Tensor

from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.engram.engram_module import EngramConfig
from megatron.core.models.fuji.fuji_block import FujiBlock
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


class FujiModel(GPTModel):
    """GPT language model augmented with mHC residual streams and Engram memory.

    FujiModel is a drop-in replacement for GPTModel in training and inference
    pipelines.  It overrides the decoder construction to use FujiBlock (which
    handles stream expansion/collapse internally) and passes ``input_ids`` to the
    decoder for Engram lookup.

    Args:
        config:                      TransformerConfig (set use_mhc=True and/or
                                     use_engram=True to enable the new features).
        transformer_layer_spec:      Layer spec — should use MHCTransformerLayer.
                                     See :func:`get_fuji_layer_spec`.
        vocab_size:                  Vocabulary size.
        max_sequence_length:         Maximum sequence length.
        pre_process:                 Whether this pipeline stage contains the embedding.
        post_process:                Whether this pipeline stage contains the output layer.
        fp16_lm_cross_entropy:       Use FP16 for cross-entropy loss.
        parallel_output:             Keep the output logits split across TP ranks.
        share_embeddings_and_output_weights: Tie embedding and output weights.
        position_embedding_type:     Position embedding type.
        rotary_percent:              Fraction of hidden dim to apply RoPE to.
        rotary_base:                 RoPE base frequency.
        rope_scaling:                Enable RoPE scaling.
        rope_scaling_factor:         RoPE scaling factor.
        scatter_embedding_sequence_parallel: Scatter embedding for SP.
        seq_len_interpolation_factor: Sequence length interpolation factor.
        mtp_block_spec:              Multi-Token Prediction block spec.
        pg_collection:               Process-group collection.
        vp_stage:                    Virtual-pipeline stage index.
        engram_config:               Engram configuration. If None, Engram is
                                     disabled even when config.use_engram=True.
    """

    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            'learned_absolute', 'rope', 'mrope', 'yarn', 'none'
        ] = 'learned_absolute',
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        scatter_embedding_sequence_parallel: bool = True,
        seq_len_interpolation_factor: Optional[float] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        engram_config: Optional[EngramConfig] = None,
    ) -> None:
        # Store engram_config BEFORE calling super().__init__ so that _build_decoder
        # (called from super) can access it.
        self._fuji_engram_config = engram_config

        super().__init__(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=fp16_lm_cross_entropy,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=share_embeddings_and_output_weights,
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            rope_scaling=rope_scaling,
            rope_scaling_factor=rope_scaling_factor,
            scatter_embedding_sequence_parallel=scatter_embedding_sequence_parallel,
            seq_len_interpolation_factor=seq_len_interpolation_factor,
            mtp_block_spec=mtp_block_spec,
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )
        # Replace the TransformerBlock created by GPTModel.__init__ with FujiBlock.
        # Must be called AFTER super().__init__ since GPTModel builds self.decoder there.
        self._replace_decoder()

    def _replace_decoder(self) -> None:
        """Replace the TransformerBlock decoder with a FujiBlock after super init."""
        # Delete the TransformerBlock built by GPTModel.__init__ to free its
        # parameters before allocating FujiBlock (avoids doubling decoder memory).
        del self.decoder
        # Re-create the decoder as a FujiBlock using the same parameters
        self.decoder = FujiBlock(
            config=self.config,
            spec=self.transformer_layer_spec,
            engram_config=self._fuji_engram_config,
            post_layer_norm=True,
            pre_process=self.pre_process,
            post_process=self.post_process,
            pg_collection=self.pg_collection,
            vp_stage=self.vp_stage,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_context=None,
        packed_seq_params=None,
        extra_block_kwargs: dict = None,
        runtime_gather_output: Optional[bool] = None,
        *,
        inference_params=None,
        loss_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass.

        Identical to GPTModel.forward except that input_ids is also forwarded
        to the decoder (FujiBlock) so that Engram modules can perform N-gram
        memory lookup.

        Args:
            input_ids:    Token IDs [B, S].
            position_ids: Position IDs [B, S].
            attention_mask: Attention mask.
            See GPTModel.forward for the remaining args.

        Returns:
            Loss tensor when labels is given, otherwise logits [S, B, vocab].
        """
        # Inject input_ids into extra_block_kwargs for FujiBlock
        fuji_kwargs = dict(extra_block_kwargs or {})
        if getattr(self.config, 'use_engram', False):
            fuji_kwargs['input_ids'] = input_ids

        return super().forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=fuji_kwargs,
            runtime_gather_output=runtime_gather_output,
            inference_params=inference_params,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
        )


def build_fuji_model(
    config: TransformerConfig,
    transformer_layer_spec: ModuleSpec,
    vocab_size: int,
    max_sequence_length: int,
    engram_config: Optional[EngramConfig] = None,
    **kwargs,
) -> FujiModel:
    """Convenience factory to construct a FujiModel and immediately replace
    the TransformerBlock decoder with a FujiBlock.

    This is the recommended entry-point for creating FujiModel instances.

    Args:
        config:                 TransformerConfig.
        transformer_layer_spec: Layer spec (use get_fuji_layer_spec()).
        vocab_size:             Vocabulary size.
        max_sequence_length:    Maximum sequence length.
        engram_config:          Optional Engram configuration.
        **kwargs:               Forwarded to FujiModel.__init__.

    Returns:
        A fully-initialised FujiModel with FujiBlock as its decoder.
    """
    # FujiModel.__init__ automatically calls _replace_decoder().
    model = FujiModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_sequence_length,
        engram_config=engram_config,
        **kwargs,
    )
    return model
