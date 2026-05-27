from __future__ import annotations

import argparse
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a tiny local Llama smoke model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


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
        "explain",
        "what",
        "a",
        "runtime",
        "estimator",
        "needs",
        "to",
        "measure",
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
    model = LlamaForCausalLM(config)
    return model


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(f"Smoke model already exists at {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = build_tokenizer()
    model = build_model()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Wrote smoke model to {output_dir}")


if __name__ == "__main__":
    main()
