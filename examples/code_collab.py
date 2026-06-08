"""
Collaborative code generation with IAC + HumanEval.

Adapted from leetcode-func-print.py:
- Uses IACTrainer (not MAGRPO — lower VRAM, fits T4×2)
- Qwen3-0.6B full bf16 fine-tune
- Two agents: Agent A (helper function) + Agent B (main + tests)
- Execution reward: +0.5 syntax, +0.5 runs, +0.1 per passing test
"""

import ast
import contextlib
import io
import re
import signal
import argparse
from functools import partial
from typing import Any, Dict, List

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from comlrl.trainers.reinforce import MAGRPOConfig, MAGRPOTrainer


# ─── Code cleanup & execution ────────────────────────────────

def cleanup_code(code: str) -> str:
    """Remove markdown and explanatory text."""
    code = re.sub(r"```python\s*", "", code)
    code = re.sub(r"```\s*", "", code)
    lines = []
    for line in code.split("\n"):
        s = line.strip()
        if s and not re.match(
            r"^(Here|This|The|Now|Let|We|In|Note|Make|You|I|Please|Remember|Below|Above)",
            s,
        ):
            lines.append(line)
    return "\n".join(lines).strip()


def extract_test_cases(code: str, func_name: str) -> List[tuple]:
    """Extract (print_stmt, expected_output) pairs."""
    tests = []
    for line in code.split("\n"):
        line = line.strip()
        if line.startswith(f"print({func_name}(") and "#" in line:
            code_part, comment = line.split("#", 1)
            expected = comment.strip()
            tests.append((code_part.strip(), expected))
        elif line.startswith("print(") and "#" in line:
            code_part, comment = line.split("#", 1)
            tests.append((code_part.strip(), comment.strip()))
    # Deduplicate
    seen = set()
    unique = []
    for stmt, exp in tests:
        key = re.sub(r"\s+", "", stmt)
        if key not in seen:
            seen.add(key)
            unique.append((stmt, exp))
    return unique[:5]


def execution_reward(
    completions_a: List[str],
    completions_b: List[str],
) -> List[float]:
    """Reward: +0.5 syntax valid, +0.5 runs, +0.1 per passing test."""
    rewards = []
    TIMEOUT = 300

    for c1, c2 in zip(completions_a, completions_b):
        reward = 0.0
        code_a = cleanup_code(c1)
        code_b = cleanup_code(c2)

        # Find function in A
        func_match = re.search(r"def\s+(\w+)\s*\(", code_a)
        if not func_match:
            rewards.append(0.0)
            continue
        func_name = func_match.group(1)

        # B must use A's function (not redefine it)
        if func_name not in code_b:
            rewards.append(0.0)
            continue
        if re.search(r"def\s+" + re.escape(func_name) + r"\s*\(", code_b):
            rewards.append(0.0)
            continue

        combined = f"{code_a}\n\n{code_b}"
        try:
            ast.parse(combined)
            reward += 0.5  # syntax valid

            tests = extract_test_cases(combined, func_name)

            local_vars = {}
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):

                # Timeout protection
                signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))
                signal.alarm(TIMEOUT)
                try:
                    exec(combined, local_vars)
                except TimeoutError:
                    signal.alarm(0)
                    rewards.append(reward)
                    continue
                except RecursionError:
                    signal.alarm(0)
                    rewards.append(reward)
                    continue
                except Exception:
                    signal.alarm(0)
                    rewards.append(reward)
                    continue
                signal.alarm(0)

            reward += 0.5  # runs

            for stmt, expected in tests:
                try:
                    test_env = dict(local_vars)
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        exec(stmt, test_env)
                    actual = buf.getvalue().strip()
                    # Compare numbers
                    actual_num = re.search(r"-?\d+(?:\.\d+)?", actual)
                    expected_num = re.search(r"-?\d+(?:\.\d+)?", expected)
                    if actual_num and expected_num and actual_num.group() == expected_num.group():
                        reward += 0.1
                except Exception:
                    pass

        except SyntaxError:
            pass

        rewards.append(reward)

    return rewards


# ─── Prompt formatters ────────────────────────────────────────

def load_humaneval(split_size: int = 100):
    """Load HumanEval dataset from HuggingFace."""
    ds = load_dataset("openai_humaneval", trust_remote_code=True, split="test")
    total = min(split_size, len(ds))
    return ds.select(range(total))


def helper_formatter(example: Dict[str, Any]) -> str:
    """Agent A: write a helper function."""
    prompt_text = example.get("prompt", "")
    # HumanEval prompt format: "def func_name(args)\n    \"\"\"docstring\"\"\"\n"
    func_match = re.match(r'(def\s+\w+\(.*?\).*?\n\s*)"""(.*?)"""', prompt_text, re.DOTALL)
    if func_match:
        signature = func_match.group(1).strip()
        docstring = func_match.group(2).strip()
    else:
        signature = ""
        docstring = prompt_text

    return (
        "You are a Python coding assistant. Write a SINGLE self-contained helper function.\n"
        "Output ONLY the function code — no explanations, no markdown, no main block.\n\n"
        f"Requirement: {docstring}\n\n"
        + (f"Function signature: {signature}\n\n" if signature else "")
        + "Your helper function:"
    )


def main_formatter(example: Dict[str, Any]) -> str:
    """Agent B: write main function + tests using the helper."""
    prompt_text = example.get("prompt", "")
    docstring = prompt_text.split('"""')[1] if '"""' in prompt_text else prompt_text
    entry_point = example.get("entry_point", "solution")

    return (
        "You are a Python coding assistant. A helper function has already been defined.\n"
        "Write a MAIN function that USES the helper to solve the task.\n"
        "Include print() statements with test cases and expected outputs as comments.\n"
        "Output ONLY the code — no explanations, no markdown.\n\n"
        f"Task: {docstring}\n\n"
        f"The helper function '{entry_point}' is already available.\n"
        "Your main function with tests:"
    )


# ─── Main ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="IAC code generation on HumanEval")
    p.add_argument("--model-name", default="Qwen/Qwen3-0.6B")
    p.add_argument("--output-dir", default="./outputs/iac_code")
    p.add_argument("--dataset-size", type=int, default=100)
    p.add_argument("--num-train-epochs", type=int, default=20)
    p.add_argument("--agent-lr", type=float, default=5e-6)
    p.add_argument("--value-loss-coef", type=float, default=0.6)
    p.add_argument("--rollout-buffer-size", type=int, default=2)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.6)
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load HumanEval
    dataset = load_humaneval(args.dataset_size)
    print(f"Dataset: {len(dataset)} HumanEval problems")

    # Load two agents, each on its own GPU
    num_gpus = max(1, torch.cuda.device_count())
    agents = []
    for i in range(2):
        gpu = i % num_gpus
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            device_map={"": gpu},
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        agents.append(model)
        print(f"Agent {i} loaded on GPU {gpu}")

    config = MAGRPOConfig(
        num_train_epochs=args.num_train_epochs,
        agent_learning_rate=args.agent_lr,
        num_generations=2,  # group size for relative advantage
        rollout_buffer_size=args.rollout_buffer_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=0.9,
        num_agents=2,
        num_turns=1,
        parallel_training="mp" if num_gpus >= 2 else "none",
        agent_devices=["cuda:0", "cuda:1"] if num_gpus >= 2 else "cpu",
    )

    trainer = MAGRPOTrainer(
        agents=agents,
        tokenizer=tokenizer,
        reward_func=execution_reward,
        formatters=[helper_formatter, main_formatter],
        args=config,
        train_dataset=dataset,
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
