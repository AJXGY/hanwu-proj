from __future__ import annotations

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def build_tokenizer() -> PreTrainedTokenizerFast:
    vocab_tokens = [
        "<pad>",
        "<s>",
        "</s>",
        "<unk>",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "train",
        "time",
        "profile",
        "pipeline",
        "microbatch",
        "loss",
        "optimizer",
        "stage",
        ".",
        ",",
        "!",
        "?",
    ]
    vocab = {token: index for index, token in enumerate(vocab_tokens)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )


def build_model() -> LlamaForCausalLM:
    tokenizer = build_tokenizer()
    config = LlamaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=False,
    )
    return LlamaForCausalLM(config)
