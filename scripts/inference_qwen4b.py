"""Run Interactive or Single-Prompt Inference on Qwen 4B with Optional LoRA Adapter.

Usage:
    # Single Prompt
    python scripts/inference_qwen4b.py --prompt "Explain how to secure a Linux SSH server."

    # With LoRA Adapter
    python scripts/inference_qwen4b.py --adapter_path outputs/qwen3_4b_cyber/final_adapter --prompt "How do you detect a SYN flood?"

    # Interactive Mode
    python scripts/inference_qwen4b.py --interactive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.qwen4b.inference import generate_response, load_lora_adapter
from src.models.qwen4b.loader import load_qwen_model, load_qwen_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference with Qwen 4B")
    parser.add_argument("--config", default="configs/qwen3_4b_t4.yaml", help="Path to config file")
    parser.add_argument("--adapter_path", default=None, help="Optional path to LoRA adapter")
    parser.add_argument("--prompt", default=None, help="Single prompt to evaluate")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive CLI session")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2, help="Generation temperature")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_id = cfg["model"]["base_model_id"]
    print(f"Loading tokenizer from '{model_id}'...")
    tokenizer = load_qwen_tokenizer(model_id)

    print(f"Loading model '{model_id}'...")
    model = load_qwen_model(
        model_id_or_path=model_id,
        load_in_4bit=cfg["model"].get("load_in_4bit", True),
        compute_dtype=cfg["model"].get("bnb_4bit_compute_dtype", "float16"),
        device_map=cfg["model"].get("device_map", "auto"),
    )

    if args.adapter_path:
        print(f"Loading LoRA adapter from '{args.adapter_path}'...")
        model = load_lora_adapter(model, args.adapter_path)

    system_prompt = "You are a specialized cybersecurity AI assistant providing rigorous, defensively grounded technical explanations."

    if args.prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": args.prompt},
        ]
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            messages_or_prompt=messages,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        print("\n--- Model Response ---")
        print(response)
        print("----------------------\n")

    if args.interactive or not args.prompt:
        print("\nEntering interactive mode. Type 'exit' or 'quit' to end.\n")
        messages = [{"role": "system", "content": system_prompt}]
        while True:
            try:
                user_input = input("\nYou: ").strip()
                if user_input.lower() in ("exit", "quit"):
                    break
                if not user_input:
                    continue

                messages.append({"role": "user", "content": user_input})
                response = generate_response(
                    model=model,
                    tokenizer=tokenizer,
                    messages_or_prompt=messages,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                print(f"\nAssistant: {response}")
                messages.append({"role": "assistant", "content": response})
            except (KeyboardInterrupt, EOFError):
                break


if __name__ == "__main__":
    main()
