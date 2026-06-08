"""
TLDR collaborative writing with QLoRA + LoRA — adapted from tldr-len-ratio.py.

Key changes from original:
- QLoRA (4-bit nf4) for 8GB GPUs
- LoRA (r=8) via PEFT
- Qwen3-0.6B as default (original used Qwen2.5-0.5B)
- No other changes — uses CoMLRL's IACTrainer as-is

Usage:
    python examples/tldr_qlora.py --model-name Qwen/Qwen3-0.6B --dataset-size 320
"""

import argparse
from functools import partial
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from comlrl.trainers.actor_critic import IACConfig, IACTrainer
from comlrl.models.actor_critic import CausalLMWithValueHead


# ─── Reward (unchanged from original) ─────────────────────────

def dual_length_reward(
    short_responses: List[str],
    long_responses: List[str],
    ratio_min: float = 2.0,
    ratio_max: float = 3.0,
    short_target: int = 220,
    short_scale: float | None = None,
) -> list[float]:
    """Reward two agents for matching a target length ratio."""
    if ratio_min <= 0:
        raise ValueError("ratio_min must be > 0.")
    if ratio_max <= ratio_min:
        raise ValueError("ratio_max must exceed ratio_min.")

    scale = short_scale if short_scale is not None else max(short_target / 2, 1.0)
    rewards = []

    for short_resp, long_resp in zip(short_responses, long_responses):
        short_text = short_resp.rstrip()
        long_text = long_resp.rstrip()
        short_len = len(short_text)
        long_len = len(long_text)

        if short_len == 0 or long_len == 0:
            rewards.append(-1.0)
            continue

        ratio = long_len / max(short_len, 1)
        if ratio_min <= ratio <= ratio_max:
            ratio_score = 1.0
        elif ratio < ratio_min:
            ratio_score = 1.0 - (ratio_min - ratio) / ratio_min
        else:
            ratio_score = 1.0 - (ratio - ratio_max) / ratio_max
        ratio_score = max(-1.0, ratio_score)

        short_score = 1.0 - abs(short_len - short_target) / scale
        short_score = max(-1.0, min(short_score, 1.0))

        combined = 0.5 * (ratio_score + short_score)
        rewards.append(float(max(-1.0, min(combined, 1.0))))

    return rewards


# ─── Prompt formatters (unchanged from original) ──────────────

def build_prompt_formatters(tokenizer):
    def make_formatter(system_prompt: str):
        def _formatter(example):
            prompt = example.get("prompt")
            if prompt is None:
                raise KeyError("Expected 'prompt' field in dataset example.")
            apply_template = getattr(tokenizer, "apply_chat_template", None)
            if callable(apply_template):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ]
                return apply_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            return f"{system_prompt}\n\n{prompt}"
        return _formatter

    concise = "You summarize Reddit posts into concise TL;DRs (~220 characters)."
    detailed = (
        "You summarize Reddit posts into detailed TL;DRs about 2-3x longer than a"
        " standard version."
    )
    return [make_formatter(concise), make_formatter(detailed)]


def rollout_metrics(rollouts):
    if not rollouts:
        return {}
    char_lengths = [sample.metadata.get("char_length", 0.0) for sample in rollouts]
    return {"response_char_length_mean": float(sum(char_lengths) / len(char_lengths))}


# ─── QLoRA model loading (NEW) ───────────────────────────────

def load_qlora_agent(model_name: str, lora_r: int = 8, lora_alpha: int = 16):
    """Load a model with 4-bit QLoRA + LoRA + ValueHead."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Required for LoRA on 4-bit
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_config)

    # Wrap with ValueHead (CoMLRL's actor-critic)
    model = CausalLMWithValueHead(model, attach_value_head=True)

    print(f"Loaded QLoRA agent: {model_name} (r={lora_r})")
    return model


# ─── Main ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two IAC agents on TL;DR with QLoRA."
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=str, default="./iac_tldr_qlora")
    parser.add_argument("--dataset-size", type=int, default=320)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--agent-learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-loss-coef", type=float, default=0.6)
    parser.add_argument("--rollout-buffer-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("trl-lib/tldr", split="train")
    usable = min(args.dataset_size, len(dataset))
    dataset = dataset.select(range(usable))
    print(f"Dataset: {usable} samples")

    # Load TWO agents with QLoRA + LoRA + ValueHead
    agents = [
        load_qlora_agent(args.model_name, lora_r=args.lora_r, lora_alpha=args.lora_alpha)
        for _ in range(2)
    ]

    config = IACConfig(
        num_train_epochs=args.num_train_epochs,
        agent_learning_rate=args.agent_learning_rate,
        value_loss_coef=args.value_loss_coef,
        rollout_buffer_size=args.rollout_buffer_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        use_separate_critic=False,
        num_agents=2,
        num_turns=1,
    )

    reward_fn = partial(
        dual_length_reward,
        ratio_min=2.0,
        ratio_max=3.0,
        short_target=220,
    )

    trainer = IACTrainer(
        agents=agents,
        tokenizer=tokenizer,
        reward_func=reward_fn,
        formatters=build_prompt_formatters(tokenizer),
        args=config,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
