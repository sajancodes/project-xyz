import torch
from tokenizers import Tokenizer

from model import SmallEnglishLLM
from model_config import ModelConfig


# ============================================================
# EXPERIMENT 1 — GRAMMAR EVALUATION
# ============================================================

CHECKPOINT_PATH = "checkpoint_fineweb.pt"
TOKENIZER_PATH = "tokenizer.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)


# ============================================================
# LOAD MODEL
# ============================================================

config = ModelConfig()

model = SmallEnglishLLM(config)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()


print("=" * 70)
print("EXPERIMENT 1 — GRAMMAR EVALUATION")
print("=" * 70)

print("Checkpoint:", CHECKPOINT_PATH)
print("Training step:", checkpoint["step"])
print("Training loss:", checkpoint["loss"])
print("Device:", DEVICE)
print()


# ============================================================
# HELPER
# ============================================================

def get_next_token_probabilities(text):

    encoded = tokenizer.encode(text)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.no_grad():

        logits, _ = model(
            input_ids
        )

    # Last position predicts the next token
    next_token_logits = logits[0, -1]

    probabilities = torch.softmax(
        next_token_logits,
        dim=-1
    )

    return probabilities


def token_probability(text, candidate):

    probabilities = get_next_token_probabilities(text)

    candidate_ids = tokenizer.encode(
        candidate
    ).ids

    if len(candidate_ids) != 1:
        return None

    token_id = candidate_ids[0]

    return probabilities[token_id].item()


def evaluate_question(prefix, candidates, expected):

    probabilities = get_next_token_probabilities(
        prefix
    )

    results = []

    for candidate in candidates:

        ids = tokenizer.encode(
            candidate
        ).ids

        if len(ids) != 1:
            results.append(
                (candidate, None)
            )
            continue

        token_id = ids[0]

        probability = probabilities[
            token_id
        ].item()

        results.append(
            (candidate, probability)
        )

    valid_results = [
        x for x in results
        if x[1] is not None
    ]

    prediction = max(
        valid_results,
        key=lambda x: x[1]
    )[0]

    correct = prediction == expected

    print(f"Sentence: {prefix}___")

    for candidate, probability in results:

        if probability is None:

            print(
                f"  {candidate:>5} : "
                f"NOT SINGLE TOKEN"
            )

        else:

            print(
                f"  {candidate:>5} : "
                f"{probability:.6f}"
            )

    print(
        f"Prediction: {prediction}"
    )

    print(
        f"Expected:   {expected}"
    )

    print(
        f"Result:     {'PASS ✓' if correct else 'FAIL ✗'}"
    )

    print()

    return correct


# ============================================================
# BASIC GRAMMAR TEST
# ============================================================

tests = [

    # Prefix, candidates, expected

    (
        "I ",
        ["am", "is", "are"],
        "am"
    ),

    (
        "He ",
        ["am", "is", "are"],
        "is"
    ),

    (
        "She ",
        ["am", "is", "are"],
        "is"
    ),

    (
        "They ",
        ["am", "is", "are"],
        "are"
    ),

    (
        "We ",
        ["am", "is", "are"],
        "are"
    ),

    (
        "You ",
        ["am", "is", "are"],
        "are"
    ),

    (
        "It ",
        ["am", "is", "are"],
        "is"
    ),
]


# ============================================================
# RUN TESTS
# ============================================================

correct = 0
total = len(tests)


for prefix, candidates, expected in tests:

    if evaluate_question(
        prefix,
        candidates,
        expected
    ):

        correct += 1


accuracy = (
    correct / total
) * 100


# ============================================================
# SUMMARY
# ============================================================

print("=" * 70)
print("GRAMMAR EVALUATION SUMMARY")
print("=" * 70)

print(
    f"Correct: {correct}/{total}"
)

print(
    f"Accuracy: {accuracy:.2f}%"
)

print("=" * 70)