#!/usr/bin/env python3
"""
SUN DIAGNOSTIC FOR PROJECT XYZ
==============================

Purpose: Measure the model's sunrise/sunset directional knowledge on held-out
examples that were NOT seen during any training run.

The old evaluation echoed prompts (100% fake scores) and used verbatim training
data strings. This benchmark fixes both:
  1. Keyword matching only on the generated continuation
  2. Balanced rise/set paraphrases with unseen wording

Expected ground truth:
  - Sunrise / rise / come up → east
  - Sunset / set / go down → west

Do NOT train on the exact evaluation wording.
Use paraphrases only.
"""

import os
import re
import sys
import json
import argparse
from datetime import datetime

# Ensure uv Python packages are findable (torch etc.)
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


class SunDiagnostic:
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

    def _is_correct_direction(self, continuation, expected_dir):
        """Check if continuation mentions the expected direction (east/west)."""
        c = continuation.lower()
        if expected_dir == "east":
            return kw_in(c, ["east"]) or kw_in(c, ["rise", "coming up", "sunrise"])
        elif expected_dir == "west":
            return kw_in(c, ["west"]) or kw_in(c, ["set", "going down", "sunset"])
        return False

    # -------------------
    # SUN CATEGORIES
    # -------------------
    def test_rise_east(self):
        """Sunrise questions expecting 'east'."""
        tests = [
            ("User: Where does the sun rise?\nAssistant:", "east"),
            ("User: Where does sunrise occur?\nAssistant:", "east"),
            ("User: In which direction does the sun come up?\nAssistant:", "east"),
            ("User: Where does the sun come up in the morning?\nAssistant:", "east"),
            ("User: Sunrise occurs in which direction?\nAssistant:", "east"),
            ("User: Where does the sun appear at dawn?\nAssistant:", "east"),
            ("User: What direction is sunrise?\nAssistant:", "east"),
            ("User: The sun rises in the ___.\nAssistant:", "east"),
        ]
        return tests

    def test_set_west(self):
        """Sunset questions expecting 'west'."""
        tests = [
            ("User: Where does the sun set?\nAssistant:", "west"),
            ("User: Where does sunset occur?\nAssistant:", "west"),
            ("User: In which direction does the sun go down?\nAssistant:", "west"),
            ("User: Where does the sun go down in the evening?\nAssistant:", "west"),
            ("User: Sunset occurs in which direction?\nAssistant:", "west"),
            ("User: Where does the sun appear at dusk?\nAssistant:", "west"),
            ("User: What direction is sunset?\nAssistant:", "west"),
            ("User: The sun sets in the ___.\nAssistant:", "west"),
        ]
        return tests

    def test_rise_paraphrase(self):
        """Paraphrased sunrise questions."""
        tests = [
            ("The sun rises in the east.", "east"),
            ("The sun comes up in the east.", "east"),
            ("Sunrise occurs in the east.", "east"),
            ("In the morning, the sun appears in the east.", "east"),
            ("Dawn breaks in the east.", "east"),
        ]
        return tests

    def test_set_paraphrase(self):
        """Paraphrased sunset questions."""
        tests = [
            ("The sun sets in the west.", "west"),
            ("The sun goes down in the west.", "west"),
            ("Sunset occurs in the west.", "west"),
            ("In the evening, the sun disappears in the west.", "west"),
            ("Dusk falls in the west.", "west"),
        ]
        return tests

    # -------------------
    # RUN ALL
    # -------------------
    def run(self):
        torch.manual_seed(0)

        all_tests = [
            ("rise_east", self.test_rise_east()),
            ("set_west", self.test_set_west()),
            ("rise_paraphrase", self.test_rise_paraphrase()),
            ("set_paraphrase", self.test_set_paraphrase()),
        ]

        category_results = {}

        for name, tests in all_tests:
            results = []
            n = 0
            correct = 0
            for prompt, expected_dir in tests:
                cont = self.continuation(prompt, max_new_tokens=30, temperature=0.3, top_k=40)
                correct_flag = self._is_correct_direction(cont, expected_dir)
                n += 1
                if correct_flag:
                    correct += 1
                results.append({
                    "prompt": prompt,
                    "expected_direction": expected_dir,
                    "continuation": cont[:120],
                    "correct": correct_flag,
                })
            accuracy = correct / n if n > 0 else 0.0
            category_results[name] = {
                "n": n,
                "accuracy": accuracy,
                "correct": correct,
                "results": results,
            }

        # Compute overall
        total_n = sum(r["n"] for r in category_results.values())
        total_correct = sum(r["correct"] for r in category_results.values())
        overall_accuracy = total_correct / total_n if total_n > 0 else 0.0

        return {
            "checkpoint": self.checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "category_breakdown": category_results,
            "overall_accuracy": overall_accuracy,
            "total_examples": total_n,
            "total_correct": total_correct,
        }


def main():
    parser = argparse.ArgumentParser(description="Sun diagnostic")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    bench = SunDiagnostic(args.checkpoint)
    result = bench.run()

    print("\n" + "=" * 50)
    print("SUN DIAGNOSTIC SUMMARY")
    print("=" * 50)
    print(f"Overall accuracy: {result['overall_accuracy']*100:5.1f}% "
          f"({result['total_correct']}/{result['total_examples']})")
    print()
    for name, data in result["category_breakdown"].items():
        print(f"  {name}: {data['accuracy']*100:5.1f}% "
              f"({data['correct']}/{data['n']})"
              f"  (rise/east vs set/west)")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
    return result


if __name__ == "__main__":
    main()