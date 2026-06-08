"""
TLDR collaborative writing — no QLoRA, no LoRA, full-parameter fine-tuning.

Loads Qwen3-0.6B in BF16 and trains ALL parameters (base model + ValueHead).
Much faster than QLoRA. Use --agent-devices to split agents across GPUs
(e.g. Kaggle T4×2: --agent-devices cuda:0 cuda:1).

Usage:
    python examples/tldr_simple.py --model-name Qwen/Qwen3-0.6B --dataset-size 320

    # On Kaggle T4×2, split across GPUs:
    python examples/tldr_simple.py --agent-devices cuda:0 cuda:1
"""

import argparse
from functools import partial
from typing import List

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from comlrl.trainers.actor_critic import IACConfig, IACTrainer
from comlrl.models.actor_critic import CausalLMWithValueHead


# ─── Reward ─────────────────────────────────────────────────────

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


# ─── Prompt formatters ──────────────────────────────────────────

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


# ─── Model loading (full-param, no QLoRA/LoRA) ─────────────────

def load_agent_full(model_name: str):
    """Load model in BF16 with trainable base + ValueHead. No quantization."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    # Wrap with ValueHead — both base model and ValueHead are trainable
    model = CausalLMWithValueHead(model, attach_value_head=True)

    print(f"Loaded agent (full-param): {model_name}")
    return model


# ─── Main ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train two IAC agents on TL;DR (full-param, no QLoRA/LoRA)."
    )
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--output-dir", type=str, default="./iac_tldr_simple")
    parser.add_argument("--dataset-size", type=int, default=320)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--agent-learning-rate", type=float, default=1e-4)
    parser.add_argument("--value-loss-coef", type=float, default=0.6)
    parser.add_argument("--rollout-buffer-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--agent-devices", type=str, nargs="*", default=None,
                        help="GPU devices for each agent, e.g. 'cuda:0 cuda:1'")
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

    # Load TWO agents with full-param training
    agents = [
        load_agent_full(args.model_name)
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
        agent_devices=args.agent_devices,
        parallel_training="mp" if args.agent_devices else "none",
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
