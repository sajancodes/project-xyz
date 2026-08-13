#!/usr/bin/env python3
"""
SEMV2E LEAKAGE-SAFE HELD-OUT BENCHMARK
========================================
A permanent evaluation set for semantic binding, generated procedurally
with a FIXED seed so the exact same questions are used every run.

LEAKAGE RULES (mission section 9):
  1. All entity names in this benchmark are DISJOINT from the SEMV2E
     training name pool and from the existing semantic generator pool.
  2. Entity/object/color combinations in this benchmark are freshly
     generated with a fixed seed - none of the exact (name, object,
     color, relation, question-wording) tuples are used in training.
  3. The dataset is persisted to data/semv2e_benchmark.json and is
     NEVER used by any optimizer. It is evaluation-only.
  4. Question templates differ from the training templates.

Categories (mission section 5):
  A  single_fact        one fact, one question
  B  multi_fact         two facts, retrieve one binding
  C  distractor         >=3 facts + irrelevant sentence, retrieve one
  D  relation_reversal  entity <-> owned object
  E  paraphrase         different verb/wording for same binding
  F  pronoun            coreference resolution
  G  multi_hop          two-hop color/object binding
  H  sentence_order     scrambled sentence order robustness
  I  entity_generalization  unseen entities/combinations
  J  distractor_binding symmetric red-car/blue-car question pair

Usage:
  python src/evaluation/semv2e_benchmark.py generate
  python src/evaluation/semv2e_benchmark.py eval <checkpoint> --output out.json
"""

import os
import re
import sys
import json
import argparse
import random
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

from paths import TOKENIZER_PATH, PROJECT_ROOT
from model import SmallEnglishLLM, ModelConfig


# ------------------------------------------------------------
# ENTITY SEPARATION
# ------------------------------------------------------------
# Existing SEMANTIC V2 TRAINING name pool (semantic_data.NAME_POOL):
TRAIN_NAMES = {
    "Emma", "Noah", "Olivia", "Liam", "Ava", "Ethan", "Sofia", "Daniel",
    "Mia", "Lucas", "Grace", "Henry", "Amara", "Dev", "Ines", "Luna",
    "Ivy", "Jack", "Ruby", "Max", "Nora", "Kai", "Zara", "Arun",
    "Dara", "Nia", "Owen", "Tara", "Hugo", "Vera",
}
# Existing OLD held-out benchmark names (heldout_benchmark.py + plain_qa):
OLD_BENCH_NAMES = {
    "Alice", "Bob", "Meera", "Karan", "Leo", "Zoe", "Nina", "Tom",
    "Maria", "Omar", "Priya", "Sajan", "Riya", "Rex", "Bubbles",
    "Tweety", "Sky", "Fluff", "Pip", "Coco", "Robin", "Wren",
    "Sparrow", "Eagle", "Owl", "Parrot", "Pigeon", "Crow",
}
# SEMV2E evaluation-only names - disjoint from BOTH sets above:
EVAL_NAMES = [
    "Maya", "Ravi", "Sita", "Theo", "Ana", "Leon", "Iris", "Milo",
    "June", "Ola", "Ben", "Chloe", "Zane", "Frida", "Ash", "Bria",
    "Cato", "Dina", "Elon", "Faye", "Gia", "Hana", "Ivo", "Jules",
    "Kim", "Leif", "Mina", "Nemo", "Oona", "Pio",
]

_ASSERT_OK = (set(EVAL_NAMES) & (TRAIN_NAMES | OLD_BENCH_NAMES))
if _ASSERT_OK:
    raise RuntimeError(f"EVAL_NAMES overlap with training/old benchmark: {_ASSERT_OK}")

OBJECTS = ["car", "bicycle", "umbrella", "boat", "watch", "jacket",
           "bag", "hat", "phone", "book", "pen", "lamp"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange"]
OWN_VERBS = ["owns", "has", "possesses", "keeps"]
ACQ_VERBS = ["bought", "got", "received", "acquired"]
PRONOUNS = [("He", "his"), ("She", "her"), ("They", "their")]

DATASET_PATH = os.path.join(PROJECT_ROOT, "data", "semv2e_benchmark.json")


# ------------------------------------------------------------
# TEMPLATE-BASED GENERATION (fixed seed)
# ------------------------------------------------------------
class Semv2eBenchmarkGen:
    def __init__(self, seed=20260813):
        self.rng = random.Random(seed)

    def _name(self):
        return self.rng.choice(EVAL_NAMES)

    def _color(self):
        return self.rng.choice(COLORS)

    def _obj(self):
        return self.rng.choice(OBJECTS)

    def _pron(self):
        return self.rng.choice(PRONOUNS)

    # ---------------- A: single fact ----------------
    def gen_single_fact(self):
        name, color, obj = self._name(), self._color(), self._obj()
        verb = self.rng.choice(OWN_VERBS)
        q = self.rng.choice([
            f"What does {name} own?",
            f"What does {name} have?",
            f"What is {name}'s {obj}?",
            f"What belongs to {name}?",
        ])
        kw = [color, obj]
        prompt = f"{name} {verb} a {color} {obj}. {q}"
        return prompt, kw

    # ---------------- B: multi fact ----------------
    def gen_multi_fact(self):
        a, b = self.rng.sample(EVAL_NAMES, 2)
        ca, oa = self._color(), self._obj()
        cb, ob = self._color(), self._obj()
        verb = self.rng.choice(OWN_VERBS)
        # avoid identical (color,obj) pairs
        tries = 0
        while (ca, oa) == (cb, ob) and tries < 20:
            cb, ob = self._color(), self._obj()
            tries += 1
        target = self.rng.choice([a, b])
        if target == a:
            kw = [ca, oa]
        else:
            kw = [cb, ob]
        prompt = (f"{a} {verb} a {ca} {oa}. {b} {verb} a {cb} {ob}. "
                  f"What does {target} {verb.split() and 'own' or 'own'}?")
        # normalize question to 'own' for consistency
        prompt = f"{a} {verb} a {ca} {oa}. {b} {verb} a {cb} {ob}. What does {target} own?"
        return prompt, kw

    # ---------------- C: distractor ----------------
    def gen_distractor(self):
        names = self.rng.sample(EVAL_NAMES, 3)
        triples = [(self._color(), self._obj()) for _ in range(3)]
        verb = self.rng.choice(OWN_VERBS)
        target = self.rng.choice(names)
        c, o = triples[names.index(target)]
        kw = [c, o]
        extra = self.rng.choice([
            f"{self.rng.choice([n for n in names if n != target])} is a teacher.",
            f"{target} lives near the river.",
            f"{self.rng.choice([n for n in names if n != target])} likes to read.",
        ])
        facts = " ".join(
            f"{n} {verb} a {c} {o}." for n, (c, o) in zip(names, triples)
        )
        prompt = f"{facts} {extra} What does {target} own?"
        return prompt, kw

    # ---------------- D: relation reversal ----------------
    def gen_relation_reversal(self):
        name, color, obj = self._name(), self._color(), self._obj()
        q = self.rng.choice([
            f"Who owns the {obj}?",
            f"Who has the {color} {obj}?",
            f"Whose {obj} is it?",
            f"Who does the {color} {obj} belong to?",
        ])
        prompt = f"{name} owns a {color} {obj}. {q}"
        return prompt, [name]

    # ---------------- E: paraphrase ----------------
    def gen_paraphrase(self):
        name, color, obj = self._name(), self._color(), self._obj()
        fact_verb = self.rng.choice(["possesses", "has", "owns", "keeps"])
        q_verb = self.rng.choice(["own", "have", "possess"])
        prompt = f"{name} {fact_verb} a {color} {obj}. What does {name} {q_verb}?"
        return prompt, [color, obj]

    # ---------------- F: pronoun ----------------
    def gen_pronoun(self):
        name, color, obj = self._name(), self._color(), self._obj()
        Subj, Poss = self._pron()
        if self.rng.random() < 0.5:
            prompt = (f"{name} bought a {color} {obj}. {Subj} parked it outside. "
                      f"What color is {Poss} {obj}?")
            kw = [color]
        else:
            prompt = (f"{name} has a {color} {obj}. {Subj} left it at home. "
                      f"What did {name} leave at home?")
            kw = [color, obj]
        return prompt, kw

    # ---------------- G: multi-hop ----------------
    def gen_multi_hop(self):
        name, color, obj = self._name(), self._color(), self._obj()
        if self.rng.random() < 0.5:
            prompt = (f"{name} owns a {obj}. The {obj} is {color}. "
                      f"What color is {name}'s {obj}?")
            kw = [color]
        else:
            prompt = (f"A {obj} belongs to {name}. The {obj} is {color}. "
                      f"What does {name} own?")
            kw = [color, obj]
        return prompt, kw

    # ---------------- H: sentence order ----------------
    def gen_sentence_order(self):
        name, color, obj = self._name(), self._color(), self._obj()
        order = self.rng.random()
        if order < 0.33:
            prompt = (f"A {color} {obj} belongs to {name}. Who owns the {obj}?")
            kw = [name]
        elif order < 0.66:
            prompt = (f"A {color} {obj} belongs to {name}. What does {name} own?")
            kw = [color, obj]
        else:
            prompt = (f"The {obj} is parked outside. {name} owns the {color} {obj}. "
                      f"Who owns the {obj}?")
            kw = [name]
        return prompt, kw

    # ---------------- I: entity generalization ----------------
    def gen_entity_generalization(self):
        # unique name + random object/color; unusual pairing
        name = self._name()
        color, obj = self._color(), self._obj()
        verb = self.rng.choice(OWN_VERBS)
        if self.rng.random() < 0.5:
            prompt = f"{name} {verb} a {color} {obj}. What does {name} own?"
            kw = [color, obj]
        else:
            prompt = f"The {color} {obj} belongs to {name}. Whose {obj} is it?"
            kw = [name]
        return prompt, kw

    # ---------------- J: distractor binding (symmetric pair) ----------------
    def gen_distractor_binding(self):
        a, b = self.rng.sample(EVAL_NAMES, 2)
        obj = self.rng.choice(["car", "bicycle", "boat", "umbrella"])
        c1, c2 = self.rng.sample(COLORS, 2)
        verb = self.rng.choice(OWN_VERBS)
        p1 = f"{a} {verb} a {c1} {obj}. {b} {verb} a {c2} {obj}. What color is {b}'s {obj}?"
        p2 = f"{a} {verb} a {c1} {obj}. {b} {verb} a {c2} {obj}. What color is {a}'s {obj}?"
        return (p1, [c2]), (p2, [c1])

    def generate(self, per_category=60):
        items = []
        cat = "A_single_fact"
        for _ in range(per_category):
            p, kw = self.gen_single_fact()
            items.append({"category": cat, "prompt": p, "keywords": kw})

        cat = "B_multi_fact"
        for _ in range(per_category):
            p, kw = self.gen_multi_fact()
            items.append({"category": cat, "prompt": p, "keywords": kw})

        cat = "C_distractor"
        for _ in range(per_category):
            p, kw = self.gen_distractor()
            items.append({"category": cat, "prompt": p, "keywords": kw})

        cat = "D_relation_reversal"
        for _ in range(per_category):
            p, kw = self.gen_relation_reversal()
            items.append({"category": cat, "prompt": p, "keywords": kw})

        cat = "E_paraphrase"
        for _ in range(per_category):
            p, kw = self.gen_paraphrase()
            items.append({"category": cat, "prompt": p, "keywords": kw})

        cat = "F_pronoun"
        for _ in range(per_category):
            p, kw = self.gen_pronoun()
            items.append({"category": cat, "prompt": p, "keywords": kw})

        cat = "G_multi_hop"
        for _ in range(per_category):
            p, kw = self.gen_multi_hop()
            items.append({"category": cat, "prompt": p, "keywords": kw})

        cat = "H_sentence_order"
        for _ in range(per_category):
            p, kw = self.gen_sentence_order()
            items.append({"category": cat, "prompt": p, "keywords": kw})

        cat = "I_entity_generalization"
        for _ in range(per_category):
            p, kw = self.gen_entity_generalization()
            items.append({"category": cat, "prompt": p, "keywords": kw})

        cat = "J_distractor_binding"
        for _ in range(per_category // 2):
            (p1, kw1), (p2, kw2) = self.gen_distractor_binding()
            items.append({"category": cat, "prompt": p1, "keywords": kw1})
            items.append({"category": cat, "prompt": p2, "keywords": kw2})

        return items


def generate_dataset():
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    gen = Semv2eBenchmarkGen(seed=20260813)
    items = gen.generate(per_category=60)
    counts = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    doc = {
        "name": "SEMV2E held-out semantic binding benchmark",
        "generator_seed": 20260813,
        "created": datetime.now().isoformat(),
        "n": len(items),
        "per_category": counts,
        "entity_pool": EVAL_NAMES,
        "disjoint_from": {
            "semantic_v2_train_names": sorted(TRAIN_NAMES),
            "old_bench_names": sorted(OLD_BENCH_NAMES),
        },
        "leakage_rule": "eval-only, never used by any optimizer; "
                        "entities and (name,object,color,relation,question) tuples "
                        "disjoint from training",
        "items": items,
    }
    with open(DATASET_PATH, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"Generated {len(items)} held-out benchmark questions -> {DATASET_PATH}")
    for cat, n in counts.items():
        print(f"  {cat:<24} {n}")
    return doc


# ------------------------------------------------------------
# EVALUATION (greedy, deterministic)
# ------------------------------------------------------------
def kw_in(text, keywords):
    if isinstance(keywords, str):
        keywords = [keywords]
    lower = text.lower()
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower):
            return True
    return False


def kw_all(text, keywords):
    """All keywords must appear (word boundary)."""
    lower = text.lower()
    for kw in keywords:
        if not re.search(r"\b" + re.escape(kw.lower()) + r"\b", lower):
            return False
    return True


class Semv2eBenchmarkEval:
    def __init__(self, checkpoint_path, device="cpu"):
        self.checkpoint_path = checkpoint_path
        self.tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
        self.config = ModelConfig()
        self.device = torch.device(device)
        self.model = SmallEnglishLLM(self.config).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.ckpt = ckpt
        self.checkpoint_info = {
            "step": ckpt.get("step", "unknown"),
            "loss": ckpt.get("loss", "unknown"),
            "base_checkpoint": ckpt.get("base_checkpoint", "none"),
            "training_type": ckpt.get("training_type", "unknown"),
        }

    @torch.no_grad()
    def continuation(self, prompt, max_new_tokens=20, seed=0):
        encoded = self.tokenizer.encode(prompt)
        input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=self.device)
        prompt_len = input_ids.shape[1]
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.config.max_seq_len:]
            logits, _ = self.model(x)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, nxt], dim=1)
            if nxt.item() == 3:
                break
        return self.tokenizer.decode(input_ids[0, prompt_len:].tolist()).strip()

    def evaluate_item(self, item, wrapped):
        prompt = item["prompt"]
        if wrapped:
            full = f"User: {prompt}\nAssistant:"
        else:
            full = prompt + "\nAssistant:"
        answer = self.continuation(full)
        first8 = " ".join(answer.split()[:8])
        all_ok = kw_all(first8, item["keywords"]) or kw_all(answer, item["keywords"])
        return {
            "category": item["category"],
            "prompt": prompt,
            "keywords": item["keywords"],
            "answer": answer[:120],
            "correct": all_ok,
            "strict": kw_all(first8, item["keywords"]),
        }

    def run(self):
        with open(DATASET_PATH) as f:
            doc = json.load(f)
        items = doc["items"]

        chat_results = {}
        plain_results = {}
        chat_n = plain_n = chat_c = plain_c = 0
        for it in items:
            cr = self.evaluate_item(it, wrapped=True)
            pr = self.evaluate_item(it, wrapped=False)
            chat_results.setdefault(it["category"], []).append(cr)
            plain_results.setdefault(it["category"], []).append(pr)
            if cr["correct"]:
                chat_c += 1
            if pr["correct"]:
                plain_c += 1
            chat_n += 1
            plain_n += 1

        def summarize(results, n, c):
            per_cat = {}
            for cat, rs in results.items():
                per_cat[cat] = {
                    "n": len(rs),
                    "correct": sum(r["correct"] for r in rs),
                    "strict": sum(r["strict"] for r in rs),
                    "accuracy": sum(r["correct"] for r in rs) / len(rs),
                }
            return {
                "per_category": per_cat,
                "n": n,
                "correct": c,
                "accuracy": c / n,
            }

        params = sum(p.numel() for p in self.model.parameters())
        return {
            "protocol": "SEMV2E-benchmark-v1",
            "checkpoint": self.checkpoint_path,
            "checkpoint_info": self.checkpoint_info,
            "metadata": {
                "parameter_count": params,
                "device": str(self.device),
                "decoding": "greedy(argmax)",
                "temperature": 0.0,
                "max_new_tokens": 20,
                "dataset": DATASET_PATH,
                "n_total": len(items),
                "categories": sorted(set(i["category"] for i in items)),
            },
            "chat_wrapped": summarize(chat_results, chat_n, chat_c),
            "plain": summarize(plain_results, plain_n, plain_c),
            "timestamp": datetime.now().isoformat(),
        }


def main():
    ap = argparse.ArgumentParser(description="SEMV2E held-out benchmark")
    ap.add_argument("mode", choices=["generate", "eval"])
    ap.add_argument("checkpoint", nargs="?", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    if args.mode == "generate":
        generate_dataset()
        return

    if not args.checkpoint:
        print("eval requires a checkpoint path")
        return

    ev = Semv2eBenchmarkEval(args.checkpoint)
    result = ev.run()

    print("=" * 70)
    print("SEMV2E BENCHMARK (greedy, deterministic)")
    print(f"Checkpoint : {args.checkpoint}")
    print("=" * 70)
    for label in ("chat_wrapped", "plain"):
        s = result[label]
        print(f"\n[{label}] overall accuracy {s['accuracy']*100:.1f}% ({s['correct']}/{s['n']})")
        for cat, d in sorted(s["per_category"].items()):
            print(f"  {cat:<24} {d['accuracy']*100:5.1f}%  ({d['correct']}/{d['n']})")
    print("=" * 70)

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved to {args.output}")
    return result


if __name__ == "__main__":
    main()