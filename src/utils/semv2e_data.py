#!/usr/bin/env python3
"""
SEMV2E PROCEDURAL TRAINING DATA GENERATOR
===========================================
Generates leakage-safe training examples focused on semantic binding.

LEAKAGE SAFETY (mission section 9):
  - Training names are DISJOINT from the SEMV2E held-out benchmark
    entity pool (data/semv2e_benchmark.json uses Maya/Ravi/Sita/Theo/...).
    See EVAL_BENCH_NAMES below - never generated here.
  - The (name, object, color, relation, question) tuples differ from the
    benchmark's fixed-seed generated set.
  - This file produces ONLY training data.

Mixture (mission section 10) - the trainer applies ratios; this generator
produces a large pool per category so sampling is never exhausted.

Categories (mirror the benchmark to teach the underlying capability):
  single_fact, multi_fact, distractor, relation_reversal, paraphrase,
  pronoun, multi_hop, sentence_order, generalization, negation,
  contrastive_binding, arithmetic, sun, instruction
"""

import random
from semantic_data import SemanticGenerator, load_instruction_subset

# SEMV2E held-out benchmark names - NEVER used for training:
EVAL_BENCH_NAMES = {
    "Maya", "Ravi", "Sita", "Theo", "Ana", "Leon", "Iris", "Milo",
    "June", "Ola", "Ben", "Chloe", "Zane", "Frida", "Ash", "Bria",
    "Cato", "Dina", "Elon", "Faye", "Gia", "Hana", "Ivo", "Jules",
    "Kim", "Leif", "Mina", "Nemo", "Oona", "Pio",
}

# Training-only names (disjoint from benchmark + old benchmark pools).
TRAIN_NAMES = [
    "Aaron", "Bella", "Caleb", "Daisy", "Elena", "Felix", "Greta", "Hector",
    "Isla", "Jonas", "Kira", "Lena", "Mateo", "Nina", "Oscar", "Paula",
    "Quinn", "Rosa", "Sam", "Tessa", "Umar", "Val", "Wade", "Xena",
    "Yara", "Zia", "Adam", "Briana", "Cole", "Dara",
]

# Overlap guard: if training names ever collide with benchmark names, fail loudly.
_overlap = set(TRAIN_NAMES) & EVAL_BENCH_NAMES
if _overlap:
    raise RuntimeError(f"TRAIN_NAMES overlap with benchmark: {_overlap}")

# Deterministic gender for pronoun consistency (grammar correctness).
_MALE = {"Aaron", "Caleb", "Felix", "Hector", "Jonas", "Mateo", "Oscar",
         "Quinn", "Sam", "Umar", "Wade", "Adam", "Cole"}
_FEMALE = {"Bella", "Daisy", "Elena", "Greta", "Isla", "Kira", "Lena",
           "Paula", "Rosa", "Tessa", "Xena", "Yara", "Zia", "Briana", "Dara"}
GENDER = {}
for n in TRAIN_NAMES:
    GENDER[n] = "he" if n in _MALE else ("she" if n in _FEMALE else "they")

PRONOUN_POOL = {
    "he": ("He", "his"),
    "she": ("She", "her"),
    "they": ("They", "their"),
}

OBJECTS = ["car", "bicycle", "umbrella", "boat", "watch", "jacket",
           "bag", "hat", "phone", "book", "pen", "lamp",
           "scooter", "guitar", "drum", "camera", "radio", "pillow"]
COLORS = ["red", "blue", "green", "yellow", "black", "white", "purple", "orange",
          "pink", "brown", "gray", "gold"]
OWN_VERBS = ["owns", "has", "possesses", "keeps", "has got"]
ACQ_VERBS = ["bought", "got", "received", "acquired", "purchased"]
PRONOUNS = [("He", "his"), ("She", "her"), ("They", "their")]
LOCATIONS = ["outside", "inside", "in the garage", "at home", "in the yard",
             "in the driveway", "near the park", "under the tree"]


class Semv2eDataGen:
    def __init__(self, seed=20260813):
        self.rng = random.Random(seed)

    def _name(self):
        return self.rng.choice(TRAIN_NAMES)

    def _color(self):
        return self.rng.choice(COLORS)

    def _obj(self):
        return self.rng.choice(OBJECTS)

    def _loc(self):
        return self.rng.choice(LOCATIONS)

    def _pron(self, name):
        return PRONOUN_POOL[GENDER[name]]
    def single_fact(self):
        name, color, obj = self._name(), self._color(), self._obj()
        verb = self.rng.choice(OWN_VERBS)
        q = self.rng.choice([
            f"What does {name} own?",
            f"What does {name} have?",
            f"What belongs to {name}?",
            f"Tell me what {name} owns.",
        ])
        return f"{name} {verb} a {color} {obj}. {q}", [color, obj]

    # ---------------- multi fact ----------------
    def multi_fact(self):
        a, b = self.rng.sample(TRAIN_NAMES, 2)
        ca, oa = self._color(), self._obj()
        cb, ob = self._color(), self._obj()
        verb = self.rng.choice(OWN_VERBS)
        target = self.rng.choice([a, b])
        kw = [ca, oa] if target == a else [cb, ob]
        return (f"{a} {verb} a {ca} {oa}. {b} {verb} a {cb} {ob}. "
                f"What does {target} own?"), kw

    # ---------------- distractor ----------------
    def distractor(self):
        names = self.rng.sample(TRAIN_NAMES, 3)
        triples = [(self._color(), self._obj()) for _ in range(3)]
        verb = self.rng.choice(OWN_VERBS)
        target = self.rng.choice(names)
        c, o = triples[names.index(target)]
        facts = " ".join(f"{n} {verb} a {c} {o}." for n, (c, o) in zip(names, triples))
        distractor = self.rng.choice([
            f"{self.rng.choice([n for n in names if n != target])} is a teacher.",
            f"{target} lives near the river.",
            f"{self.rng.choice([n for n in names if n != target])} likes to read.",
            f"{target} walked to the store yesterday.",
        ])
        return f"{facts} {distractor} What does {target} own?", [c, o]

    # ---------------- relation reversal ----------------
    def relation_reversal(self):
        name, color, obj = self._name(), self._color(), self._obj()
        q = self.rng.choice([
            f"Who owns the {obj}?",
            f"Who has the {color} {obj}?",
            f"Whose {obj} is it?",
            f"Who does the {color} {obj} belong to?",
        ])
        return f"{name} owns a {color} {obj}. {q}", [name]

    # ---------------- paraphrase ----------------
    def paraphrase(self):
        name, color, obj = self._name(), self._color(), self._obj()
        fact_verb = self.rng.choice(["possesses", "has", "owns", "keeps"])
        q_verb = self.rng.choice(["own", "have", "possess"])
        return (f"{name} {fact_verb} a {color} {obj}. "
                f"What does {name} {q_verb}?"), [color, obj]

    # ---------------- pronoun ----------------
    def pronoun(self):
        name, color, obj = self._name(), self._color(), self._obj()
        Subj, Poss = self._pron(name)
        if self.rng.random() < 0.5:
            return (f"{name} bought a {color} {obj}. {Subj} parked it outside. "
                    f"What color is {Poss} {obj}?"), [color]
        return (f"{name} has a {color} {obj}. {Subj} left it at home. "
                f"What did {name} leave at home?"), [color, obj]

    # ---------------- multi hop ----------------
    def multi_hop(self):
        name, color, obj = self._name(), self._color(), self._obj()
        if self.rng.random() < 0.5:
            return (f"{name} owns a {obj}. The {obj} is {color}. "
                    f"What color is {name}'s {obj}?"), [color]
        return (f"A {obj} belongs to {name}. The {obj} is {color}. "
                f"What does {name} own?"), [color, obj]

    # ---------------- sentence order ----------------
    def sentence_order(self):
        name, color, obj = self._name(), self._color(), self._obj()
        r = self.rng.random()
        if r < 0.33:
            return (f"A {color} {obj} belongs to {name}. Who owns the {obj}?"), [name]
        if r < 0.66:
            return (f"A {color} {obj} belongs to {name}. What does {name} own?"), [color, obj]
        return (f"The {obj} is parked outside. {name} owns the {color} {obj}. "
                f"Who owns the {obj}?"), [name]

    # ---------------- generalization (fresh objects/verbs) ----------------
    def generalization(self):
        name = self._name()
        color, obj = self._color(), self._obj()
        verb = self.rng.choice(["owns", "has", "keeps"])
        if self.rng.random() < 0.5:
            return f"{name} {verb} a {color} {obj}. What does {name} own?", [color, obj]
        return f"The {color} {obj} belongs to {name}. Whose {obj} is it?", [name]

    # ---------------- negation ----------------
    def negation(self):
        name, color, obj = self._name(), self._color(), self._obj()
        verb = self.rng.choice(OWN_VERBS)
        if self.rng.random() < 0.5:
            # "does not own" - answer should reference who owns it or that nobody does
            other = self._name()
            return (f"{name} does not own a {obj}. {other} owns a {color} {obj}. "
                    f"Does {name} own the {color} {obj}?"), ["no"]
        return (f"{name} owns a {color} {obj}. Is it true that {name} owns a "
                f"{self.rng.choice([c for c in COLORS if c != color])} {obj}?"), ["no"]

    # ---------------- contrastive binding (like benchmark J) ----------------
    def contrastive_binding(self):
        a, b = self.rng.sample(TRAIN_NAMES, 2)
        obj = self.rng.choice(["car", "bicycle", "boat", "umbrella", "watch", "phone"])
        c1, c2 = self.rng.sample(COLORS, 2)
        verb = self.rng.choice(OWN_VERBS)
        if self.rng.random() < 0.5:
            return (f"{a} {verb} a {c1} {obj}. {b} {verb} a {c2} {obj}. "
                    f"What color is {b}'s {obj}?"), [c2]
        return (f"{a} {verb} a {c1} {obj}. {b} {verb} a {c2} {obj}. "
                f"What color is {a}'s {obj}?"), [c1]

    # ---------------- two-location state ----------------
    def location_state(self):
        name, color, obj = self._name(), self._color(), self._obj()
        loc = self._loc()
        return (f"{name} parked the {color} {obj} {loc}. "
                f"Where did {name} park the {obj}?"), [loc.split()[0] if loc.startswith(("in ", "under ", "near ")) else loc]

    # ---------------- category dispatch ----------------
    CATS = [
        ("single_fact", 14), ("multi_fact", 14), ("distractor", 12),
        ("relation_reversal", 12), ("paraphrase", 12), ("pronoun", 10),
        ("multi_hop", 10), ("sentence_order", 8), ("generalization", 8),
        ("negation", 6), ("contrastive_binding", 12), ("location_state", 6),
    ]

    def generate(self, n, seed=None):
        rng = random.Random(seed if seed is not None else self.rng.randint(0, 2**31))
        cats = [c for c, w in self.CATS for _ in range(w)]
        examples = []
        seen = set()
        attempts = 0
        while len(examples) < n and attempts < n * 200:
            attempts += 1
            cat = rng.choice(cats)
            text, kw = getattr(self, cat)()
            user, assistant = text.split(". ", 1) if ". " in text else (text, "")
            # build User:/Assistant: with the QUESTION as the user part
            # find the question (last sentence)
            sentences = [s.strip() for s in text.split(".") if s.strip()]
            if len(sentences) >= 2:
                question = sentences[-1]
                context = ". ".join(sentences[:-1])
                answer = None
                # derive a natural answer keyword-based completion
                if cat in ("relation_reversal", "sentence_order") and "Who" in question:
                    answer = kw[0]
                elif cat == "negation":
                    answer = kw[0]
                elif kw and cat in ("single_fact", "multi_fact", "distractor",
                                    "paraphrase", "multi_hop", "generalization",
                                    "contrastive_binding", "location_state"):
                    answer = f"a {kw[0]} {kw[1]}" if len(kw) == 2 else kw[0]
                elif cat == "pronoun":
                    answer = kw[0]
                elif kw:
                    answer = kw[0]
                if answer is None:
                    continue
                if len(kw) == 2 and isinstance(answer, str) and answer.startswith("a "):
                    first = kw[0]
                    answer = f"an {kw[0]} {kw[1]}" if first[0] in "aeiou" else f"a {kw[0]} {kw[1]}"
                ex = f"User: {context}. {question}\nAssistant: {answer}"
            else:
                ex = text
            if ex not in seen:
                seen.add(ex)
                examples.append(ex)
        return examples


def main():
    gen = Semv2eDataGen(seed=20260813)
    ex = gen.generate(3000, seed=20260813)
    print(f"Generated {len(ex)} SEMV2E training examples")
    for e in ex[:20]:
        print("---")
        print(e)


if __name__ == "__main__":
    main()