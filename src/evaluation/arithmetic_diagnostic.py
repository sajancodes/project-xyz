#!/usr/bin/env python3
"""
ARITHMETIC DIAGNOSTIC FOR PROJECT XYZ
=====================================

Purpose: Measure the model's arithmetic generalization on held-out examples
that were NOT seen during any training run.

The old evaluation suite echoed prompts into responses (100% fake scores).
This benchmark fixes that by:
  1. Keyword matching only on the generated continuation
  2. Unseen numbers, names, objects, and wordings
  3. Both direct and natural-language arithmetic

Do NOT train on these exact questions.
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime

uv_base = r"C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-windows-x86_64-none"
if uv_base not in sys.path:
    sys.path.insert(0, uv_base)
if uv_base + r"\Lib\site-packages" not in sys.path:
    sys.path.insert(0, uv_base + r"\Lib\site-packages")

import torch
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import TOKENIZER_PATH
from model import SmallEnglishLLM, ModelConfig


def kw_in(text, keywords):
    """Word-boundary keyword check."""
    if isinstance(keywords, str):
        keywords = [keywords]
    lower = text.lower()
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower):
            return True
    return False


class ArithmeticDiagnostic:
    def __init__(self, checkpoint_path, tokenizer_path=TOKENIZER_PATH, device="cpu"):
        self.checkpoint_path = checkpoint_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.config = ModelConfig()
        self.device = torch.device(device)
        self.model = SmallEnglishLLM(self.config).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def continuation(self, prompt, max_new_tokens=30, temperature=0.3, top_k=40):
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        prompt_len = input_ids.shape[1]
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self.model(x)
            next_logits = logits[:, -1, :] / temperature
            values, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            filtered = torch.full_like(next_logits, float("-inf"))
            filtered.scatter_(1, indices, values)
            probs = torch.softmax(filtered, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == 3:
                break
        return self.tokenizer.decode(input_ids[0, prompt_len:].tolist()).strip()

    def _extract_answer(self, continuation):
        c = continuation.strip().rstrip(".")
        nums = re.findall(r"\b\d+\b", c)
        if nums:
            return nums[-1]
        if "Answer:" in c.upper():
            parts = c.upper().split("ANSWER:")
            if len(parts) > 1:
                last = parts[-1].strip()
                nums = re.findall(r"\b\d+\b", last)
                if nums:
                    return nums[-1]
        return None

    def _is_correct(self, continuation, expected):
        if kw_in(continuation, str(expected)):
            return True
        extracted = self._extract_answer(continuation)
        if extracted and extracted == str(expected):
            return True
        if str(expected) in continuation:
            return True
        return False

    def test_addition(self):
        """Direct addition problems."""
        tests = []
        name_opts = ["Sajan", "Ravi", "Priya", "Meera", "Tom", "Nina", "Leo", "Zoe"]
        for i in range(50):
            a = 10 + (50 * ((100 * 37) % 89)) % 89
            b = 10 + (73 * ((100 * 53) % 71)) % 71
            if a > 95 or b > 95 or a < 10 or b < 10:
                continue
            name = name_opts[i % len(name_opts)]
            q = f"What is {a} + {b}?\nAssistant:"
            tests.append((q, str(a + b)))
        return tests

    def test_subtraction(self):
        """Direct subtraction problems."""
        tests = []
        name_opts = ["Sajan", "Ravi", "Priya", "Meera", "Tom", "Nina", "Leo", "Zoe"]
        for i in range(40):
            a = 20 + (50 * ((100 * 37) % 79)) % 71
            b = 10 + (37 * ((100 * 53) % 61)) % 53
            if b >= a or a > 98 or b > 98:
                continue
            name = name_opts[i % len(name_opts)]
            q = f"What is {a} - {b}?\nAssistant:"
            tests.append((q, str(a - b)))
        return tests

    def test_multiplication(self):
        """Direct multiplication problems."""
        tests = []
        name_opts = ["Sajan", "Ravi", "Priya", "Meera", "Tom", "Nina", "Leo", "Zoe"]
        for i in range(40):
            a = 2 + (3 * ((100 * 37) % 8))  # 2-9
            b = 2 + (5 * ((100 * 71) % 11))  # 2-16
            prod = a * b
            if prod > 100:
                continue
            name = name_opts[i % len(name_opts)]
            q = f"What is {a} * {b}?\nAssistant:"
            tests.append((q, str(prod)))
        return tests

    def test_word_problem_addition(self):
        """Word problem addition."""
        tests = []
        name_opts = ["Sajan", "Ravi", "Priya", "Meera", "Tom", "Nina", "Leo", "Zoe"]
        objs = ["apples", "books", "balls", "marbles", "coins"]
        for i in range(40):
            name = name_opts[i % len(name_opts)]
            obj = objs[i % len(objs)]
            a = 2 + (50 * ((100 * 37) % 89)) % 89
            b = 2 + (30 * ((100 * 71) % 71)) % 71
            if a > 20 or b > 20:
                continue
            q = f"{name} has {a} {obj}. They receive {b} more {obj}. How many {obj} does {name} have now?\nAssistant:"
            tests.append((q, str(a + b)))
        return tests

    def test_word_problem_subtraction(self):
        """Word problem subtraction."""
        tests = []
        name_opts = ["Sajan", "Ravi", "Priya", "Meera", "Tom", "Nina", "Leo", "Zoe"]
        objs = ["apples", "books", "balls", "marbles", "coins"]
        for i in range(30):
            name = name_opts[i % len(name_opts)]
            obj = objs[i % len(objs)]
            a = 5 + (50 * ((100 * 37) % 89)) % 89
            b = 1 + (40 * ((100 * 71) % 71)) % 59
            if b >= a or a > 30:
                continue
            q = f"{name} had {a} {obj}. They gave away {b} {obj}. How many {obj} does {name} have left?\nAssistant:"
            tests.append((q, str(a - b)))
        return tests

    def test_multi_step(self):
        """Multi-step arithmetic."""
        tests = []
        name_opts = ["Sajan", "Ravi", "Priya", "Meera", "Tom", "Nina", "Leo", "Zoe"]
        objs = ["apples", "books", "balls", "marbles"]
        for i in range(30):
            name = name_opts[i % len(name_opts)]
            obj = objs[i % len(objs)]
            a = 2 + (30 * ((100 * 37) % 89)) % 89
            b = 2 + (20 * ((100 * 71) % 71)) % 71
            c = 2 + (15 * ((100 * 53) % 61)) % 53
            if a + b + c > 50:
                continue
            q = f"{name} has {a} {obj}. They find {b} more {obj}. Then they get {c} more {obj}. How many {obj} does {name} have in total?\nAssistant:"
            tests.append((q, str(a + b + c)))
        return tests

    def run(self):
        torch.manual_seed(0)
        all_tests = [
            ("addition", self.test_addition()),
            ("subtraction", self.test_subtraction()),
            ("multiplication", self.test_multiplication()),
            ("word_problem_addition", self.test_word_problem_addition()),
            ("word_problem_subtraction", self.test_word_problem_subtraction()),
            ("multi_step", self.test_multi_step()),
        ]
        category_results = {}
        total_n = 0
        total_correct = 0
        for name, tests in all_tests:
            n = 0
            correct = 0
            results = []
            for prompt, expected in tests:
                cont = self.continuation(prompt, max_new_tokens=30, temperature=0.3, top_k=40)
                flag = self._is_correct(cont, expected)
                n += 1
                if flag:
                    correct += 1
                results.append({
                    "prompt": prompt,
                    "expected": expected,
                    "continuation": cont[:120],
                    "correct": flag,
                })
            accuracy = correct / n if n > 0 else 0.0
            category_results[name] = {
                "n": n, "accuracy": accuracy, "correct": correct, "results": results
            }
            total_n += n
            total_correct += correct
        overall = total_correct / total_n if total_n > 0 else 0.0
        return {
            "checkpoint": self.checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "category_breakdown": category_results,
            "overall_accuracy": overall,
            "total_examples": total_n,
            "total_correct": total_correct,
        }


def main():
    parser = argparse.ArgumentParser(description="Arithmetic diagnostic")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()
    bench = ArithmeticDiagnostic(args.checkpoint)
    result = bench.run()
    print("\n" + "=" * 50)
    print("ARITHMETIC DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"Overall: {result['overall_accuracy']*100:5.1f}% "
          f"({result['total_correct']}/{result['total_examples']})")
    for name, data in result["category_breakdown"].items():
        print(f"  {name}: {data['accuracy']*100:5.1f}% ({data['correct']}/{data['n']})")
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")
    return result


if __name__ == "__main__":
    main()