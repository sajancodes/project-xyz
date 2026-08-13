#!/usr/bin/env python3
"""
PLAIN-QA DIAGNOSTIC FOR PROJECT XYZ
==================================

Purpose: Measure the model's ability to answer questions WITHOUT the
User:/Assistant: chat format wrapper. This tests real-usage robustness:
can the model answer directly when the chat wrapper is absent?

The old suite only tested with chat format. This benchmark uses plain
prompts as a real application would send them.

Do NOT use User:/Assistant: wrappers in the prompts.

Report separately for each category:
- strict: expected keyword appears in the first 8 tokens
- loose: expected keyword appears anywhere in the answer (word boundary)
- echo: continuation repeats the prompt verbatim

Do NOT train on these exact questions.
Use unseen entities, numbers, and wordings.
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


def kw_in_text(text, keywords):
    """Check if any keyword appears in text at word boundaries."""
    if isinstance(keywords, str):
        keywords = [keywords]
    lower = text.lower()
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower):
            return True
    return False


class PlainQADiagnostic:
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

    def _is_echo(self, prompt, answer):
        """Check if continuation starts by repeating the prompt."""
        norm_a = answer.lower()[:80]
        norm_q = prompt.lower()[:80]
        if not norm_q or not norm_a:
            return False
        return norm_a.startswith(norm_q)

    def _check_answer(self, continuation, expected):
        """Return (strict, loose) bool tuple."""
        first8 = " ".join(continuation.split()[:8])
        strict = kw_in_text(first8, expected)
        loose = kw_in_text(continuation, expected)
        return strict, loose

    # -------------------
    # HELD-OUT PLAIN-QA CATEGORIES
    # -------------------
    def test_sun_directions(self):
        """Sun direction without chat wrapper."""
        tests_prompts = [
            "Which direction does sunrise occur?",
            "Which direction does sunset occur?",
            "Where does the sun come up?",
            "Where does the sun go down?",
            "In which direction does sunrise occur?",
            "In which direction does sunset occur?",
        ]
        tests_expected = ["east", "west", "east", "west", "east", "west"]
        results = []
        for i, prompt in enumerate(tests_prompts):
            expected = tests_expected[i]
            full_prompt = prompt + "\nAssistant:"
            answer = self.continuation(full_prompt, max_new_tokens=30, temperature=0.3, top_k=40)
            strict, loose = self._check_answer(answer, expected)
            echo = self._is_echo(prompt, answer)
            results.append({
                "prompt": prompt,
                "expected": expected,
                "answer": answer[:120],
                "strict": strict,
                "loose": loose,
                "echo": echo,
            })
        n = len(results)
        return {
            "category": "sun_directions",
            "n": n,
            "strict_accuracy": sum(r["strict"] for r in results) / n,
            "loose_accuracy": sum(r["loose"] for r in results) / n,
            "echo_rate": sum(r["echo"] for r in results) / n,
            "results": results,
        }

    def test_relations_no_wrapper(self):
        """Relation questions without chat wrapper."""
        prompts = [
            "Alice gave Bob a pen. Who received the pen?",
            "Meera gave Karan a ball. Who got the ball?",
            "Leo chased Zoe. Who chased Zoe?",
            "The lion chased the zebra. What did the lion chase?",
            "The fox chased the rabbit. Who was chased?",
            "Nina handed Tom a cup. Who got the cup?",
        ]
        expected = ["Bob", "Karan", "Leo", "zebra", "rabbit", "Tom"]
        results = []
        for i, prompt in enumerate(prompts):
            full_prompt = prompt + "\nAssistant:"
            answer = self.continuation(full_prompt, max_new_tokens=30, temperature=0.3, top_k=40)
            strict, loose = self._check_answer(answer, expected[i])
            echo = self._is_echo(prompt, answer)
            results.append({
                "prompt": prompt,
                "expected": expected[i],
                "answer": answer[:120],
                "strict": strict,
                "loose": loose,
                "echo": echo,
            })
        n = len(results)
        return {
            "category": "relations_no_wrapper",
            "n": n,
            "strict_accuracy": sum(r["strict"] for r in results) / n,
            "loose_accuracy": sum(r["loose"] for r in results) / n,
            "echo_rate": sum(r["echo"] for r in results) / n,
            "results": results,
        }

    def test_state_no_wrapper(self):
        """State questions without chat wrapper."""
        prompts = [
            "The baby is asleep. What is the baby doing?",
            "The girl is napping. What is the girl doing?",
            "The boy is jogging. What is the boy doing?",
            "The bird is flying. What is the bird doing?",
            "The fish is swimming. What is the fish doing?",
        ]
        expected = ["sleeping", "napping", "jogging", "flying", "swimming"]
        results = []
        for i, prompt in enumerate(prompts):
            full_prompt = prompt + "\nAssistant:"
            answer = self.continuation(full_prompt, max_new_tokens=30, temperature=0.3, top_k=40)
            strict, loose = self._check_answer(answer, expected[i])
            echo = self._is_echo(prompt, answer)
            results.append({
                "prompt": prompt,
                "expected": expected[i],
                "answer": answer[:120],
                "strict": strict,
                "loose": loose,
                "echo": echo,
            })
        n = len(results)
        return {
            "category": "state_no_wrapper",
            "n": n,
            "strict_accuracy": sum(r["strict"] for r in results) / n,
            "loose_accuracy": sum(r["loose"] for r in results) / n,
            "echo_rate": sum(r["echo"] for r in results) / n,
            "results": results,
        }

    def test_why_no_wrapper(self):
        """Why questions without chat wrapper."""
        prompts = [
            "Sajan went to the shop because he needed milk. Why did Sajan go to the shop?",
            "Riya stayed home because she was sick. Why did Riya stay home?",
            "Tom went to the park because it was sunny. Why did Tom go to the park?",
            "The man opened the umbrella because it was raining. Why did the man open the umbrella?",
        ]
        expected = ["milk", "sick", "sunny", "raining"]
        results = []
        for i, prompt in enumerate(prompts):
            full_prompt = prompt + "\nAssistant:"
            answer = self.continuation(full_prompt, max_new_tokens=30, temperature=0.3, top_k=40)
            strict, loose = self._check_answer(answer, expected[i])
            echo = self._is_echo(prompt, answer)
            results.append({
                "prompt": prompt,
                "expected": expected[i],
                "answer": answer[:120],
                "strict": strict,
                "loose": loose,
                "echo": echo,
            })
        n = len(results)
        return {
            "category": "why_no_wrapper",
            "n": n,
            "strict_accuracy": sum(r["strict"] for r in results) / n,
            "loose_accuracy": sum(r["loose"] for r in results) / n,
            "echo_rate": sum(r["echo"] for r in results) / n,
            "results": results,
        }

    def test_negation_no_wrapper(self):
        """Negation questions without chat wrapper."""
        prompts = [
            "The dog is not inside the house. Is the dog inside?",
            "The dog is outside. Is the dog inside?",
            "The cat is not sleeping. Is the cat sleeping?",
            "The bird is not flying. Is the bird on the ground?",
        ]
        expected = ["no", "no", "no", "yes"]
        results = []
        for i, prompt in enumerate(prompts):
            full_prompt = prompt + "\nAssistant:"
            answer = self.continuation(full_prompt, max_new_tokens=30, temperature=0.3, top_k=40)
            strict, loose = self._check_answer(answer, expected[i])
            echo = self._is_echo(prompt, answer)
            results.append({
                "prompt": prompt,
                "expected": expected[i],
                "answer": answer[:120],
                "strict": strict,
                "loose": loose,
                "echo": echo,
            })
        n = len(results)
        return {
            "category": "negation_no_wrapper",
            "n": n,
            "strict_accuracy": sum(r["strict"] for r in results) / n,
            "loose_accuracy": sum(r["loose"] for r in results) / n,
            "echo_rate": sum(r["echo"] for r in results) / n,
            "results": results,
        }

    def test_instruction_no_wrapper(self):
        """Instruction/completion without chat wrapper."""
        prompts = [
            "Complete this sentence: You ___ kind.",
            "Complete this sentence: We ___ happy.",
            "Complete this sentence: She ___ ready.",
            "Complete this sentence: They ___ running.",
            "Say the word apple.",
            "Write the color of a banana.",
        ]
        expected = ["are", "are", "is", "are", "apple", "yellow"]
        results = []
        for i, prompt in enumerate(prompts):
            full_prompt = prompt + "\nAssistant:"
            answer = self.continuation(full_prompt, max_new_tokens=30, temperature=0.3, top_k=40)
            strict, loose = self._check_answer(answer, expected[i])
            echo = self._is_echo(prompt, answer)
            results.append({
                "prompt": prompt,
                "expected": expected[i],
                "answer": answer[:120],
                "strict": strict,
                "loose": loose,
                "echo": echo,
            })
        n = len(results)
        return {
            "category": "instruction_no_wrapper",
            "n": n,
            "strict_accuracy": sum(r["strict"] for r in results) / n,
            "loose_accuracy": sum(r["loose"] for r in results) / n,
            "echo_rate": sum(r["echo"] for r in results) / n,
            "results": results,
        }

    def test_arithmetic_no_wrapper(self):
        """Arithmetic without chat wrapper."""
        prompts = [
            "What is 17 + 26?",
            "What is 38 - 15?",
            "What is 7 * 8?",
            "Sajan has 17 apples. He receives 25 more. How many apples does he have?",
            "Ravi had 38 marbles. He gave away 15. How many marbles does Ravi have left?",
        ]
        expected = ["43", "23", "56", "42", "23"]
        results = []
        for i, prompt in enumerate(prompts):
            full_prompt = prompt + "\nAssistant:"
            answer = self.continuation(full_prompt, max_new_tokens=30, temperature=0.3, top_k=40)
            strict, loose = self._check_answer(answer, expected[i])
            echo = self._is_echo(prompt, answer)
            results.append({
                "prompt": prompt,
                "expected": expected[i],
                "answer": answer[:120],
                "strict": strict,
                "loose": loose,
                "echo": echo,
            })
        n = len(results)
        return {
            "category": "arithmetic_no_wrapper",
            "n": n,
            "strict_accuracy": sum(r["strict"] for r in results) / n,
            "loose_accuracy": sum(r["loose"] for r in results) / n,
            "echo_rate": sum(r["echo"] for r in results) / n,
            "results": results,
        }

    # -------------------
    # RUN ALL
    # -------------------
    def run(self):
        torch.manual_seed(0)

        all_tests = [
            ("sun_directions", self.test_sun_directions()),
            ("relations_no_wrapper", self.test_relations_no_wrapper()),
            ("state_no_wrapper", self.test_state_no_wrapper()),
            ("why_no_wrapper", self.test_why_no_wrapper()),
            ("negation_no_wrapper", self.test_negation_no_wrapper()),
            ("instruction_no_wrapper", self.test_instruction_no_wrapper()),
            ("arithmetic_no_wrapper", self.test_arithmetic_no_wrapper()),
        ]

        category_results = {}
        total_n = 0
        total_strict = 0

        for name, result in all_tests:
            category_results[name] = result
            total_n += result["n"]
            total_strict += result["strict_accuracy"] * result["n"]

        overall_strict = total_strict / total_n if total_n > 0 else 0.0

        return {
            "checkpoint": self.checkpoint_path,
            "timestamp": datetime.now().isoformat(),
            "category_breakdown": category_results,
            "overall_strict_accuracy": overall_strict,
            "total_examples": total_n,
        }


def main():
    parser = argparse.ArgumentParser(description="Plain-QA diagnostic")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    bench = PlainQADiagnostic(args.checkpoint)
    result = bench.run()

    print("\n" + "=" * 60)
    print("PLAIN-QA DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Overall strict accuracy: {result['overall_strict_accuracy']*100:5.1f}% "
          f"({result['total_examples']} questions)")
    print()
    for name, data in result["category_breakdown"].items():
        print(f"  {name}: {data['strict_accuracy']*100:5.1f}% "
              f"({data.get('n', '?')} questions)")

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
    return result


if __name__ == "__main__":
    main()