#!/usr/bin/env python3
"""
CANONICAL SEMV2E EVALUATION (deterministic)
============================================
Fixes the measurement problem where stochastic decoding (temp 0.3
multinomial) gave inconsistent results (13/15 vs 5/15).

The core semantic benchmark uses GREEDY decoding (temperature 0 =>
argmax) so capability is not decided by random sampling. A separate
stochastic generation-quality pass (temp 0.7) is kept for fluency.

Reuses ALL existing test definitions by subclassing:
  - HeldoutBenchmark   (Cat A semantic + Cat B generation + plain QA)
  - SunDiagnostic
  - ArithmeticDiagnostic
  - PlainQADiagnostic

Records full metadata for reproducibility (mission section 4/19).

Usage:
  python src/evaluation/canonical_eval.py <checkpoint> --output out.json
  python src/evaluation/canonical_eval.py <checkpoint> --temperature 0.0
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

PROJ_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_SRC)
sys.path.insert(0, os.path.join(PROJ_SRC, "models"))
sys.path.insert(0, os.path.join(PROJ_SRC, "config"))

from paths import TOKENIZER_PATH
from model import SmallEnglishLLM, ModelConfig
from heldout_benchmark import HeldoutBenchmark, kw_in
from sun_diagnostic import SunDiagnostic
from arithmetic_diagnostic import ArithmeticDiagnostic
from plain_qa_diagnostic import PlainQADiagnostic


class DeterministicHarness:
    """Shared deterministic generation core used by all subclasses."""

    def __init__(self, checkpoint_path, tokenizer_path=TOKENIZER_PATH, device="cpu"):
        self.checkpoint_path = checkpoint_path
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.config = ModelConfig()
        self.device = torch.device(device)
        self.model = SmallEnglishLLM(self.config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.ckpt = ckpt

    @torch.no_grad()
    def continuation(self, prompt, max_new_tokens=30, temperature=0.0, top_k=None, top_p=None, seed=0):
        """Greedy (temperature=0 -> argmax) or sampled decoding."""
        rng = torch.Generator(device=self.device)
        rng.manual_seed(seed)
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        prompt_len = input_ids.shape[1]

        for _ in range(max_new_tokens):
            x = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self.model(x)
            next_logits = logits[:, -1, :]
            if temperature > 0:
                next_logits = next_logits / temperature
                if top_k is not None:
                    values, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    filtered = torch.full_like(next_logits, float("-inf"))
                    filtered.scatter_(1, indices, values)
                    next_logits = filtered
                if top_p is not None and top_p < 1.0:
                    sorted_l, sorted_i = torch.sort(next_logits, descending=True)
                    cum = torch.cumsum(torch.softmax(sorted_l, dim=-1), dim=-1)
                    mask = cum > top_p
                    mask[..., 1:] = mask[..., :-1].clone()
                    mask[..., 0] = False
                    sorted_l[mask] = float("-inf")
                    next_logits = next_logits.scatter(1, sorted_i, sorted_l)
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1, generator=rng)
            else:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_token], dim=1)
            if next_token.item() == 3:
                break
        return self.tokenizer.decode(input_ids[0, prompt_len:].tolist()).strip()


# Ensure subclasses share the deterministic continuation signature used
# internally by the original classes (they call continuation(prompt,
# max_new_tokens=..., temperature=..., top_k=...)).
class CanonicalHeldout(DeterministicHarness, HeldoutBenchmark):
    def __init__(self, checkpoint_path, temperature=0.0, seed=0, max_new_tokens=30, **kw):
        DeterministicHarness.__init__(self, checkpoint_path, device=kw.get("device", "cpu"))
        self._ev_temperature = temperature
        self._ev_seed = seed
        self._ev_max_tokens = max_new_tokens
        self.checkpoint_info = {
            "step": self.ckpt.get("step", "unknown"),
            "loss": self.ckpt.get("loss", "unknown"),
            "base_checkpoint": self.ckpt.get("base_checkpoint", "none"),
            "training_type": self.ckpt.get("training_type", "unknown"),
        }

    def continuation(self, prompt, max_new_tokens=30, temperature=0.0, top_k=None):
        return DeterministicHarness.continuation(
            self, prompt,
            max_new_tokens=max_new_tokens,
            temperature=self._ev_temperature,
            top_k=top_k,
            seed=self._ev_seed,
        )


class CanonicalSun(DeterministicHarness, SunDiagnostic):
    def __init__(self, checkpoint_path, temperature=0.0, seed=0, **kw):
        DeterministicHarness.__init__(self, checkpoint_path, device=kw.get("device", "cpu"))
        self._ev_temperature = temperature
        self._ev_seed = seed

    def continuation(self, prompt, max_new_tokens=30, temperature=0.3, top_k=40):
        return DeterministicHarness.continuation(
            self, prompt, max_new_tokens=max_new_tokens,
            temperature=self._ev_temperature, seed=self._ev_seed)


class CanonicalArithmetic(DeterministicHarness, ArithmeticDiagnostic):
    def __init__(self, checkpoint_path, temperature=0.0, seed=0, **kw):
        DeterministicHarness.__init__(self, checkpoint_path, device=kw.get("device", "cpu"))
        self._ev_temperature = temperature
        self._ev_seed = seed

    def continuation(self, prompt, max_new_tokens=30, temperature=0.3, top_k=40):
        return DeterministicHarness.continuation(
            self, prompt, max_new_tokens=max_new_tokens,
            temperature=self._ev_temperature, seed=self._ev_seed)


class CanonicalPlainQA(DeterministicHarness, PlainQADiagnostic):
    def __init__(self, checkpoint_path, temperature=0.0, seed=0, **kw):
        DeterministicHarness.__init__(self, checkpoint_path, device=kw.get("device", "cpu"))
        self._ev_temperature = temperature
        self._ev_seed = seed

    def continuation(self, prompt, max_new_tokens=30, temperature=0.3, top_k=40):
        return DeterministicHarness.continuation(
            self, prompt, max_new_tokens=max_new_tokens,
            temperature=self._ev_temperature, seed=self._ev_seed)


def run_canonical(checkpoint_path, temperature=0.0, seed=0, max_new_tokens=30, device="cpu"):
    torch.manual_seed(seed)
    bench = CanonicalHeldout(checkpoint_path, temperature=temperature, seed=seed,
                             max_new_tokens=max_new_tokens, device=device)

    print("=" * 70)
    print("CANONICAL SEMV2E EVALUATION (deterministic)")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Temperature: {temperature}  Seed: {seed}  max_new_tokens: {max_new_tokens}")
    print("=" * 70)

    # ----- Category A: existing heldout semantic categories -----
    cat_a_tests = [
        bench.test_sun_facts,
        bench.test_relations,
        bench.test_state,
        bench.test_why_context,
        bench.test_negation,
        bench.test_entity_tracking,
        bench.test_transitive,
        bench.test_arithmetic,
        bench.test_instruction,
        bench.test_context_retention,
    ]
    cat_a = {}
    strict_sum = loose_sum = n = 0
    for t in cat_a_tests:
        r = t()
        cat_a[r["category"]] = r
        strict_sum += r["strict_accuracy"] * r["n"]
        loose_sum += r["loose_accuracy"] * r["n"]
        n += r["n"]
        print(f"  {r['category']:<18} strict {r['strict_accuracy']*100:5.1f}%  "
              f"loose {r['loose_accuracy']*100:5.1f}%  echo {r['echo_rate']*100:4.1f}%")
    cat_a_strict = strict_sum / n
    cat_a_loose = loose_sum / n

    # ----- Category B: generation (stochastic fluencing kept at 0.7) -----
    candle_b = CanonicalHeldout(checkpoint_path, temperature=0.7, seed=seed * 2 + 1,
                                max_new_tokens=60, device=device)
    cat_b = candle_b.test_generation()

    # ----- Plain QA (unwrapped) -----
    plain = bench.test_plain_qa()

    print(f"\n  CAT A aggregate: strict {cat_a_strict*100:.1f}%  loose {cat_a_loose*100:.1f}%  (n={n})")
    print(f"  CAT B generation: topic {cat_b['topic_hit_rate']*100:.1f}%  rep {cat_b['repetition_ratio_avg']*100:.1f}%  drift {cat_b['dialog_drift_rate']*100:.1f}%")
    print(f"  PLAIN QA: strict {plain['strict_accuracy']*100:.1f}%")

    composite = (0.55 * cat_a_strict + 0.15 * cat_a_loose
                 + 0.15 * cat_b["topic_hit_rate"]
                 + 0.10 * (1 - cat_b["dialog_drift_rate"])
                 + 0.05 * (1 - cat_b["repetition_ratio_avg"]))

    # ----- Standalone diagnostics (greedy) -----
    print("\nRunning standalone diagnostics (greedy)...")
    sun = CanonicalSun(checkpoint_path, temperature=temperature, seed=seed, device=device)
    sun_r = sun.run()
    arith = CanonicalArithmetic(checkpoint_path, temperature=temperature, seed=seed, device=device)
    arith_r = arith.run()
    plainqa = CanonicalPlainQA(checkpoint_path, temperature=temperature, seed=seed, device=device)
    plainqa_r = plainqa.run()

    print(f"  sun overall        {sun_r['overall_accuracy']*100:5.1f}% ({sun_r['total_correct']}/{sun_r['total_examples']})")
    print(f"  arithmetic overall {arith_r['overall_accuracy']*100:5.1f}% ({arith_r['total_correct']}/{arith_r['total_examples']})")
    print(f"  plain-qa strict    {plainqa_r['overall_strict_accuracy']*100:5.1f}% ({plainqa_r['total_examples']} q)")

    params = sum(p.numel() for p in bench.model.parameters())
    result = {
        "protocol": "SEMV2E-canonical-v1",
        "checkpoint": checkpoint_path,
        "checkpoint_info": bench.checkpoint_info,
        "metadata": {
            "parameter_count": params,
            "device": device,
            "seed": seed,
            "temperature": temperature,
            "top_k": None,
            "top_p": None,
            "decoding": "greedy(argmax)" if temperature == 0 else "temp/{}".format(temperature),
            "max_new_tokens": max_new_tokens,
            "chat_wrapper": "ON for CatA; CatB plain; plain-QA unwrapped",
            "generation_quality_temperature": 0.7,
        },
        "timestamp": datetime.now().isoformat(),
        "category_a": cat_a,
        "category_a_summary": {"n": n, "strict_accuracy": cat_a_strict, "loose_accuracy": cat_a_loose},
        "category_b": cat_b,
        "category_plain_qa": plain,
        "sun_diagnostic": sun_r,
        "arithmetic_diagnostic": arith_r,
        "plain_qa_diagnostic": plainqa_r,
        "composite": composite,
        "composite_weighting": {
            "catA_strict": 0.55, "catA_loose": 0.15,
            "catB_topic_hit": 0.15, "catB_no_dialog_drift": 0.10,
            "catB_no_repetition": 0.05,
        },
    }
    print("\n" + "=" * 70)
    print(f"COMPOSITE (greedy, deterministic): {composite:.4f}")
    print("=" * 70)
    return result


def main():
    ap = argparse.ArgumentParser(description="Canonical deterministic SEMV2E evaluation")
    ap.add_argument("checkpoint", help="path to checkpoint .pt")
    ap.add_argument("--output", help="output JSON file")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="core benchmark temperature (0 = greedy)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    result = run_canonical(args.checkpoint, temperature=args.temperature, seed=args.seed)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")
    return result


if __name__ == "__main__":
    main()