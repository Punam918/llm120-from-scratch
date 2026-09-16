from __future__ import annotations

from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerBase

from llm120.config import ModelConfig


def make_hf_config(config: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> LlamaConfig:
    if len(tokenizer) > 65_535:
        raise ValueError("This pipeline's uint16 token format requires a vocabulary below 65,536")
    return LlamaConfig(
        vocab_size=len(tokenizer),
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        hidden_act="silu",
        max_position_embeddings=config.max_position_embeddings,
        initializer_range=config.initializer_range,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=config.tie_word_embeddings,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=False,
    )


def make_model(config: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> LlamaForCausalLM:
    return LlamaForCausalLM(make_hf_config(config, tokenizer))


def parameter_counts(model: LlamaForCausalLM) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def optimizer_groups(model: LlamaForCausalLM, weight_decay: float) -> list[dict]:
    """Decay matrix weights, excluding embeddings and all norm/scalar parameters."""
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "embed_tokens" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]

