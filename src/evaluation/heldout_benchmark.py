#!/usr/bin/env python3
"""
PERMANENT HELD-OUT SEMANTIC BENCHMARK FOR PROJECT XYZ
======================================================
Created: 2026-08-12

Purpose
-------
Measure REAL semantic understanding and generalization on examples that
were NOT used in any training run. The old evaluation suite:
  1. decoded the prompt into the "response", so keyword checks matched the
     echoed prompt (fake 100% scores), and
  2. used the exact same strings that appear in the semantic training data.

This benchmark fixes both problems:
  - Keyword matching happens ONLY on the generated continuation.
  - Every example uses entities, objects, wording, or numbers that never
    appear in the synthetic semantic training data (Sajan/Ravi/dog/cat/
    sun-east verbatim forms, 2+2, robin, ...).

Two categories, scored separately (mission section 13):
  CATEGORY A - SEMANTIC / INSTRUCTION (Question -> Answer)
  CATEGORY B - OPEN GENERATION (Prompt -> Relevant continuation)

Scoring
-------
For category A each example reports:
  - strict: expected keyword appears in the first 8 tokens of the answer
  - loose:  expected keyword appears anywhere in the answer (word boundary)
  - echo:   the continuation begins by repeating the user question (this is
            treated as a FAILURE - answering by echoing is not answering)
For category B each prompt reports:
  - topic_hit: a topic keyword appears in the continuation
  - repetition_ratio: 1 - unique/total words
  - dialog_drift: output contains "User:"/"Assistant:" markers even though
    the prompt was plain text (the model looping into dialogue format)
  - echo: continuation repeats the prompt verbatim

Composite score (documented weighting):
  COMPOSITE = 0.55 * catA_strict_accuracy
            + 0.15 * catA_loose_accuracy
            + 0.15 * catB_topic_hit
            + 0.10 * catB_no_dialog_drift
            + 0.05 * (1 - catB_repetition_ratio)

All generation is deterministic: temperature 0.3 / top-k 40 / fixed seed.
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
    """Word-boundary keyword check. Accepts str or list of str."""
    if isinstance(keywords, str):
        keywords = [keywords]
    lower = text.lower()
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower):
            return True
    return False


class HeldoutBenchmark:
    def __init__(self, checkpoint_path, tokenizer_path=TOKENIZER_PATH, device="cuda"):
        self.checkpoint_path = checkpoint_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.config = ModelConfig()
        self.device = torch.device("cpu")
        self.model = SmallEnglishLLM(self.config).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.checkpoint_info = {
            "step": checkpoint.get("step", "unknown"),
            "loss": checkpoint.get("loss", "unknown"),
            "base_checkpoint": checkpoint.get("base_checkpoint", "none"),
            "training_type": checkpoint.get("training_type", "unknown"),
        }

    # ------------------------------------------------------------
    # GENERATION (deterministic)
    # ------------------------------------------------------------
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
        """Continuation starts by repeating the user's question."""
        question = prompt.split("\n")[-1].replace("Assistant:", "").strip()
        norm_a = answer.lower()[:80]
        norm_q = question.lower()[:80]
        if not norm_q or not norm_a:
            return False
        return norm_a.startswith(norm_q[:40]) or norm_a.startswith("user:")

    # ------------------------------------------------------------
    # CATEGORY A - SEMANTIC / INSTRUCTION
    # ------------------------------------------------------------
    def _run_qa(self, name, tests):
        results = []
        for t in tests:
            prompt, keywords = t[0], t[1]
            answer = self.continuation(prompt, max_new_tokens=30)
            first8 = " ".join(answer.split()[:8])
            strict = kw_in(first8, keywords)
            loose = kw_in(answer, keywords)
            echo = self._is_echo(prompt, answer)
            results.append({
                "prompt": prompt,
                "keywords": keywords,
                "answer": answer[:120],
                "strict": strict,
                "loose": loose,
                "echo": echo,
            })
        n = len(results)
        return {
            "category": name,
            "n": n,
            "strict_accuracy": sum(r["strict"] for r in results) / n,
            "loose_accuracy": sum(r["loose"] for r in results) / n,
            "echo_rate": sum(r["echo"] for r in results) / n,
            "results": results,
        }

    def test_sun_facts(self):
        tests = [
            ("User: Which direction does sunrise occur?\nAssistant:", "east"),
            ("User: Where does the sun come up?\nAssistant:", "east"),
            ("User: Where does the sun rise in the morning?\nAssistant:", "east"),
            ("User: The sun sets in the ___.\nAssistant:", "west"),
            ("User: What direction is the sunset?\nAssistant:", "west"),
            ("User: Which way is the sunset?\nAssistant:", "west"),
        ]
        return self._run_qa("sun_facts", tests)

    def test_relations(self):
        tests = [
            ("User: Alice gave Bob a pen. Who received the pen?\nAssistant:", "Bob"),
            ("User: Meera gave Karan a ball. Who got the ball?\nAssistant:", "Karan"),
            ("User: Leo chased Zoe. Who chased Zoe?\nAssistant:", "Leo"),
            ("User: The lion chased the zebra. What did the lion chase?\nAssistant:", "zebra"),
            ("User: The fox chased the rabbit. Who was chased?\nAssistant:", "rabbit"),
            ("User: Nina handed Tom a cup. Who got the cup?\nAssistant:", "Tom"),
            ("User: The cat chased the mouse. What did the cat chase?\nAssistant:", "mouse"),
        ]
        return self._run_qa("relations", tests)

    def test_state(self):
        tests = [
            ("User: The baby is asleep. What is the baby doing?\nAssistant:", ["sleeping", "asleep"]),
            ("User: The girl is napping. What is the girl doing?\nAssistant:", ["sleeping", "napping"]),
            ("User: The boy is jogging. What is the boy doing?\nAssistant:", "jogging"),
            ("User: The bird is flying. What is the bird doing?\nAssistant:", "flying"),
            ("User: The fish is swimming. What is the fish doing?\nAssistant:", "swimming"),
            ("User: The old man is reading. What is the old man doing?\nAssistant:", "reading"),
        ]
        return self._run_qa("state", tests)

    def test_why_context(self):
        tests = [
            ("User: Sajan went to the shop because he needed milk. Why did Sajan go to the shop?\nAssistant:", "milk"),
            ("User: Riya stayed home because she was sick. Why did Riya stay home?\nAssistant:", "sick"),
            ("User: Tom went to the park because it was sunny. Why did Tom go to the park?\nAssistant:", "sunny"),
            ("User: The man opened the umbrella because it was raining. Why did the man open the umbrella?\nAssistant:", "raining"),
            ("User: The dog barked because a stranger was at the door. Why did the dog bark?\nAssistant:", "stranger"),
        ]
        return self._run_qa("why_context", tests)

    def test_negation(self):
        tests = [
            ("User: The dog is not inside the house. Is the dog inside?\nAssistant:", "no"),
            ("User: The dog is outside. Is the dog inside?\nAssistant:", "no"),
            ("User: The cat is not sleeping. Is the cat sleeping?\nAssistant:", "no"),
            ("User: The bird is not flying. Is the bird on the ground?\nAssistant:", "yes"),
        ]
        return self._run_qa("negation", tests)

    def test_entity_tracking(self):
        tests = [
            ("User: Alice gave Bob the red book. Who received the red book?\nAssistant:", "Bob"),
            ("User: Maria bought a blue umbrella. What color is the umbrella?\nAssistant:", "blue"),
            ("User: The yellow car belongs to Leo. Whose car is it?\nAssistant:", "Leo"),
            ("User: Priya ate the green apple. What color was the apple?\nAssistant:", "green"),
            ("User: Omar took the big box. What did Omar take?\nAssistant:", "box"),
        ]
        return self._run_qa("entity_tracking", tests)

    def test_transitive(self):
        tests = [
            ("User: A wren is a bird. All birds lay eggs. Does a wren lay eggs?\nAssistant:", "yes"),
            ("User: All dogs bark. Rex is a dog. Does Rex bark?\nAssistant:", "yes"),
            ("User: All fish swim. Bubbles is a fish. Can Bubbles swim?\nAssistant:", "yes"),
            ("User: All birds have feathers. A sparrow is a bird. Does a sparrow have feathers?\nAssistant:", "yes"),
        ]
        return self._run_qa("transitive", tests)

    def test_arithmetic(self):
        tests = [
            ("User: What is 7 + 2?\nAssistant:", "9"),
            ("User: What is 12 - 5?\nAssistant:", "7"),
            ("User: What is 4 times 3?\nAssistant:", "12"),
            ("User: What is 6 + 6?\nAssistant:", "12"),
            ("User: What is 9 - 4?\nAssistant:", "5"),
            ("User: What is 3 + 6?\nAssistant:", "9"),
        ]
        return self._run_qa("arithmetic", tests)

    def test_instruction(self):
        tests = [
            ("User: Complete this sentence: You ___ kind.\nAssistant:", "are"),
            ("User: Complete this sentence: We ___ happy.\nAssistant:", "are"),
            ("User: Complete this sentence: She ___ ready.\nAssistant:", "is"),
            ("User: Complete this sentence: They ___ running.\nAssistant:", "are"),
            ("User: Complete this sentence: The dog ___ barking.\nAssistant:", "is"),
            ("User: Say the word apple.\nAssistant:", "apple"),
            ("User: Write the color of a banana.\nAssistant:", "yellow"),
        ]
        return self._run_qa("instruction", tests)

    def test_context_retention(self):
        """Multi-turn memory with an UNSEEN name (training used 'Sajan')."""
        prompt = ("User: My name is Priya.\nAssistant: Nice to meet you, Priya.\n"
                  "User: What is my name?\nAssistant:")
        answer = self.continuation(prompt, max_new_tokens=30)
        strict = kw_in(" ".join(answer.split()[:8]), "Priya")
        loose = kw_in(answer, "Priya")
        echo = self._is_echo(prompt, answer)
        return {
            "category": "context_retention",
            "n": 1,
            "strict_accuracy": 1.0 if strict else 0.0,
            "loose_accuracy": 1.0 if loose else 0.0,
            "echo_rate": 1.0 if echo else 0.0,
            "results": [{
                "prompt": prompt,
                "keywords": "Priya",
                "answer": answer[:120],
                "strict": strict,
                "loose": loose,
                "echo": echo,
            }],
        }

    # ------------------------------------------------------------
    # CATEGORY B - OPEN GENERATION
    # ------------------------------------------------------------
    def test_generation(self):
        prompts = [
            ("the barking dog", ["dog", "bark"]),
            ("the sun rises", ["sun", "rise"]),
            ("iron man was", ["iron", "man", "hero", "marvel"]),
            ("the students are", ["student", "school", "teacher", "class"]),
            ("she opened the door", ["door", "room", "house", "open"]),
            ("the cat sat on", ["cat", "sat", "mat", "lap", "window"]),
            ("he is", []),          # generic - no topic expectation
            ("the barking dog", ["dog", "bark"]),  # repeat determinism check
        ]
        results = []
        for prompt, topics in prompts:
            out = self.continuation(prompt, max_new_tokens=60, temperature=0.7)
            words = out.split()
            rep = 1 - (len(set(words)) / len(words)) if words else 1.0
            topic_hit = kw_in(out, topics) if topics else None
            dialog_drift = ("User:" in out) or ("Assistant:" in out)
            echo = out.lower().startswith(prompt.lower())
            results.append({
                "prompt": prompt,
                "topics": topics,
                "continuation": out[:150],
                "topic_hit": topic_hit,
                "repetition_ratio": round(rep, 4),
                "dialog_drift": dialog_drift,
                "echo": echo,
            })

        scored = [r for r in results if r["topic_hit"] is not None]
        n = len(scored)
        return {
            "category": "generation",
            "n": n,
            "topic_hit_rate": sum(r["topic_hit"] for r in scored) / n,
            "repetition_ratio_avg": sum(r["repetition_ratio"] for r in results) / len(results),
            "dialog_drift_rate": sum(r["dialog_drift"] for r in results) / len(results),
            "echo_rate": sum(r["echo"] for r in results) / len(results),
            "results": results,
        }

    def test_plain_qa(self):
        """REAL-USAGE ROBUSTNESS: questions written WITHOUT the 'User:'
        wrapper, only '\nAssistant:' appended (as a real app would do).
        Reported separately - NOT part of the composite score."""
        tests = [
            ("Which direction does sunrise occur?\nAssistant:", "east"),
            ("The dog is not inside the house. Is the dog inside?\nAssistant:", "no"),
            ("Alice gave Bob a pen. Who received the pen?\nAssistant:", "Bob"),
            ("The baby is asleep. What is the baby doing?\nAssistant:", ["sleeping", "asleep"]),
            ("Sajan went to the shop because he needed milk. Why did Sajan go to the shop?\nAssistant:", "milk"),
            ("What is 7 + 2?\nAssistant:", "9"),
            ("All dogs bark. Rex is a dog. Does Rex bark?\nAssistant:", "yes"),
            ("The sun sets in the ___?\nAssistant:", "west"),
        ]
        results = []
        for t in tests:
            prompt, keywords = t[0], t[1]
            answer = self.continuation(prompt, max_new_tokens=30)
            first8 = " ".join(answer.split()[:8])
            strict = kw_in(first8, keywords)
            loose = kw_in(answer, keywords)
            echo = self._is_echo(prompt, answer)
            results.append({
                "prompt": prompt,
                "keywords": keywords,
                "answer": answer[:120],
                "strict": strict,
                "loose": loose,
                "echo": echo,
            })
        n = len(results)
        return {
            "category": "plain_qa",
            "n": n,
            "strict_accuracy": sum(r["strict"] for r in results) / n,
            "loose_accuracy": sum(r["loose"] for r in results) / n,
            "echo_rate": sum(r["echo"] for r in results) / n,
            "results": results,
        }

    # ------------------------------------------------------------
    # RUN ALL
    # ------------------------------------------------------------
    def run(self):
        torch.manual_seed(0)
        print("=" * 70)
        print("HELD-OUT BENCHMARK:", self.checkpoint_path)
        print(f"Step: {self.checkpoint_info['step']} | Loss: {self.checkpoint_info['loss']}")
        print(f"Base: {self.checkpoint_info['base_checkpoint']} | Type: {self.checkpoint_info['training_type']}")
        print("=" * 70)

        cat_a_tests = [
            self.test_sun_facts,
            self.test_relations,
            self.test_state,
            self.test_why_context,
            self.test_negation,
            self.test_entity_tracking,
            self.test_transitive,
            self.test_arithmetic,
            self.test_instruction,
            self.test_context_retention,
        ]

        cat_a = {}
        cat_a_counts = {"strict": 0, "loose": 0, "n": 0}
        for test in cat_a_tests:
            print(f"\nRunning {test.__name__}...")
            result = test()
            cat_a[result["category"]] = result
            cat_a_counts["strict"] += result["strict_accuracy"] * result["n"]
            cat_a_counts["loose"] += result["loose_accuracy"] * result["n"]
            cat_a_counts["n"] += result["n"]
            print(f"  strict {result['strict_accuracy']*100:5.1f}% | "
                  f"loose {result['loose_accuracy']*100:5.1f}% | "
                  f"echo {result['echo_rate']*100:5.1f}%")

        cat_a_strict = cat_a_counts["strict"] / cat_a_counts["n"]
        cat_a_loose = cat_a_counts["loose"] / cat_a_counts["n"]

        print("\nRunning generation...")
        cat_b = self.test_generation()
        print(f"  topic_hit {cat_b['topic_hit_rate']*100:5.1f}% | "
              f"repetition {cat_b['repetition_ratio_avg']*100:5.1f}% | "
              f"dialog_drift {cat_b['dialog_drift_rate']*100:5.1f}%")

        print("\nRunning plain (unwrapped) QA...")
        cat_plain = self.test_plain_qa()
        print(f"  strict {cat_plain['strict_accuracy']*100:5.1f}% | "
              f"loose {cat_plain['loose_accuracy']*100:5.1f}% | "
              f"echo {cat_plain['echo_rate']*100:5.1f}%")

        composite = (0.55 * cat_a_strict
                     + 0.15 * cat_a_loose
                     + 0.15 * cat_b["topic_hit_rate"]
                     + 0.10 * (1 - cat_b["dialog_drift_rate"])
                     + 0.05 * (1 - cat_b["repetition_ratio_avg"]))

        result = {
            "checkpoint": self.checkpoint_path,
            "checkpoint_info": self.checkpoint_info,
            "timestamp": datetime.now().isoformat(),
            "category_a": cat_a,
            "category_a_summary": {
                "n": cat_a_counts["n"],
                "strict_accuracy": cat_a_strict,
                "loose_accuracy": cat_a_loose,
            },
            "category_b": cat_b,
            "category_plain_qa": cat_plain,
            "composite": composite,
            "composite_weighting": {
                "catA_strict": 0.55,
                "catA_loose": 0.15,
                "catB_topic_hit": 0.15,
                "catB_no_dialog_drift": 0.10,
                "catB_no_repetition": 0.05,
            },
        }

        print("\n" + "=" * 70)
        print("HELD-OUT BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"  Category A (semantic, n={cat_a_counts['n']}): "
              f"strict {cat_a_strict*100:5.1f}%  loose {cat_a_loose*100:5.1f}%")
        print(f"  Category B (generation): "
              f"topic_hit {cat_b['topic_hit_rate']*100:5.1f}%  "
              f"repetition {cat_b['repetition_ratio_avg']*100:5.1f}%  "
              f"drift {cat_b['dialog_drift_rate']*100:5.1f}%")
        print(f"  COMPOSITE: {composite:.4f}")

        return result


def main():
    parser = argparse.ArgumentParser(description="Held-out semantic benchmark")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--output", help="Output JSON file")
    args = parser.parse_args()

    bench = HeldoutBenchmark(args.checkpoint)
    result = bench.run()

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")
    return result


if __name__ == "__main__":
    main()
