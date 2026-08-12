# ============================================================
# SEMANTIC + INSTRUCTION FINE-TUNING
# ============================================================
#
# Starts from your pretrained FineWeb checkpoint.
#
# Goal:
#   Teach the model to:
#       1. understand simple literal relationships
#       2. answer questions
#       3. follow instructions
#       4. recognize paraphrases
#       5. perform simple reasoning
#
# IMPORTANT:
# This is NOT a replacement for pretraining.
# It is a second training stage.
#
# INPUT:
#     checkpoint_fineweb_106k.pt
#
# OUTPUT:
#     checkpoint_semantic_5k.pt
#
# ============================================================

import os
import time
import torch
from torch.optim import AdamW
from tokenizers import Tokenizer

from model import SmallEnglishLLM
from model_config import ModelConfig


# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 16
SEQ_LEN = 512

LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01

MAX_STEPS = 5000

LOG_EVERY = 50
CHECKPOINT_EVERY = 50000

BASE_CHECKPOINT = "checkpoint-106k.pt"
OUTPUT_CHECKPOINT = "checkpoint_semantic_5k.pt"

TOKENIZER_PATH = "tokenizer.json"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_BF16 = torch.cuda.is_available()


# ============================================================
# TRAINING DATA
# ============================================================
#
# These are intentionally simple.
#
# We use multiple formulations of the same concepts so that
# the model cannot simply memorize one exact sentence.
#
# ============================================================

EXAMPLES = [

    # --------------------------------------------------------
    # IDENTITY
    # --------------------------------------------------------

    (
        "User: My name is Sajan.\n"
        "Assistant: Your name is Sajan."
    ),

    (
        "User: I am Sajan.\n"
        "Assistant: You are Sajan."
    ),

    (
        "User: What is my name if I say 'I am Sajan'?\n"
        "Assistant: Your name is Sajan."
    ),


    # --------------------------------------------------------
    # SUN / EAST
    # --------------------------------------------------------

    (
        "User: Where does the sun rise?\n"
        "Assistant: The sun rises in the east."
    ),

    (
        "User: What direction does the sun rise from?\n"
        "Assistant: The sun rises from the east."
    ),

    (
        "User: In which direction does sunrise occur?\n"
        "Assistant: Sunrise occurs in the east."
    ),

    (
        "User: The sun comes up in the ___\n"
        "Assistant: east."
    ),

    (
        "User: Every morning the sun appears in the eastern sky. "
        "What direction is that?\n"
        "Assistant: East."
    ),


    # --------------------------------------------------------
    # SIMPLE FACTS
    # --------------------------------------------------------

    (
        "User: What color is grass usually?\n"
        "Assistant: Grass is usually green."
    ),

    (
        "User: What color is the sky on a clear day?\n"
        "Assistant: The sky is usually blue."
    ),

    (
        "User: What do humans breathe?\n"
        "Assistant: Humans breathe air."
    ),

    (
        "User: What animal says meow?\n"
        "Assistant: A cat says meow."
    ),

    (
        "User: What animal says bark?\n"
        "Assistant: A dog says bark."
    ),


    # --------------------------------------------------------
    # SIMPLE RELATIONSHIPS
    # --------------------------------------------------------

    (
        "User: Sajan gave Ravi an apple. Who received the apple?\n"
        "Assistant: Ravi received the apple."
    ),

    (
        "User: Ravi gave Sajan a book. Who gave the book?\n"
        "Assistant: Ravi gave the book."
    ),

    (
        "User: The dog chased the cat. Who chased the cat?\n"
        "Assistant: The dog chased the cat."
    ),

    (
        "User: The cat chased the dog. What did the cat chase?\n"
        "Assistant: The cat chased the dog."
    ),


    # --------------------------------------------------------
    # PARAPHRASES
    # --------------------------------------------------------

    (
        "User: The boy is running. What is the boy doing?\n"
        "Assistant: The boy is running."
    ),

    (
        "User: The child is running. What action is the child doing?\n"
        "Assistant: The child is running."
    ),

    (
        "User: The dog is sleeping. What is the dog doing?\n"
        "Assistant: The dog is sleeping."
    ),

    (
        "User: The dog is asleep. What is happening to the dog?\n"
        "Assistant: The dog is sleeping."
    ),


    # --------------------------------------------------------
    # BASIC LOGIC
    # --------------------------------------------------------

    (
        "User: All birds have wings. A robin is a bird. "
        "Does a robin have wings?\n"
        "Assistant: Yes, a robin has wings."
    ),

    (
        "User: All cats are animals. Milo is a cat. "
        "Is Milo an animal?\n"
        "Assistant: Yes, Milo is an animal."
    ),

    (
        "User: If something is bigger than a box, "
        "and the box is bigger than a cup, "
        "what is bigger: the first thing or the cup?\n"
        "Assistant: The first thing is bigger than the cup."
    ),


    # --------------------------------------------------------
    # BASIC ARITHMETIC
    # --------------------------------------------------------

    (
        "User: What is 2 + 2?\n"
        "Assistant: 4."
    ),

    (
        "User: What is 5 + 3?\n"
        "Assistant: 8."
    ),

    (
        "User: What is 10 - 4?\n"
        "Assistant: 6."
    ),

    (
        "User: What is 3 multiplied by 4?\n"
        "Assistant: 12."
    ),


    # --------------------------------------------------------
    # INSTRUCTION FOLLOWING
    # --------------------------------------------------------

    (
        "User: Complete this sentence: I ___ Sajan.\n"
        "Assistant: am."
    ),

    (
        "User: Complete this sentence: He ___ happy.\n"
        "Assistant: is."
    ),

    (
        "User: Complete this sentence: They ___ happy.\n"
        "Assistant: are."
    ),

    (
        "User: Complete this sentence: We ___ students.\n"
        "Assistant: are."
    ),

    (
        "User: Complete this sentence: She ___ here.\n"
        "Assistant: is."
    ),


    # --------------------------------------------------------
    # QUESTION UNDERSTANDING
    # --------------------------------------------------------

    (
        "User: Who is running if I say 'The boy is running'?\n"
        "Assistant: The boy is running."
    ),

    (
        "User: What is happening in 'The dog is sleeping'?\n"
        "Assistant: The dog is sleeping."
    ),

    (
        "User: Where is the apple if Sajan puts the apple on the table?\n"
        "Assistant: The apple is on the table."
    ),


    # --------------------------------------------------------
    # CONVERSATION
    # --------------------------------------------------------

    (
        "User: Hello.\n"
        "Assistant: Hello! How can I help you?"
    ),

    (
        "User: Good morning.\n"
        "Assistant: Good morning!"
    ),

    (
        "User: How are you?\n"
        "Assistant: I am doing well. How are you?"
    ),

    (
        "User: Thank you.\n"
        "Assistant: You're welcome."
    ),

    (
        "User: Good night.\n"
        "Assistant: Good night!"
    ),
]


# ============================================================
# REPEAT DATA
# ============================================================
#
# We repeat examples to make the tiny dataset large enough
# for a small fine-tuning experiment.
#
# This is intentionally NOT 10,000 copies of one fact.
#
# ============================================================

TRAIN_TEXTS = []

for example in EXAMPLES:
    TRAIN_TEXTS.append(example)

print("=" * 70)
print("SEMANTIC + INSTRUCTION FINE-TUNING")
print("=" * 70)

print("Examples:", len(TRAIN_TEXTS))
print("Base checkpoint:", BASE_CHECKPOINT)
print("Output checkpoint:", OUTPUT_CHECKPOINT)
print()


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA is required for this experiment."
    )

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "VRAM:",
    round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024**3,
        2
    ),
    "GB"
)

print()


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = Tokenizer.from_file(
    TOKENIZER_PATH
)

print(
    "Tokenizer vocabulary:",
    tokenizer.get_vocab_size()
)

print()


# ============================================================
# MODEL
# ============================================================

config = ModelConfig()

model = SmallEnglishLLM(
    config
).to(DEVICE)


# ============================================================
# LOAD PRETRAINED CHECKPOINT
# ============================================================

if not os.path.exists(BASE_CHECKPOINT):

    raise FileNotFoundError(
        f"\nCould not find {BASE_CHECKPOINT}\n\n"
        "Make a copy of your original 106k FineWeb "
        "checkpoint first.\n\n"
        "Example:\n"
        "Copy-Item checkpoint_fineweb.pt "
        "checkpoint_fineweb_106k.pt"
    )


checkpoint = torch.load(
    BASE_CHECKPOINT,
    map_location=DEVICE
)


if "model_state_dict" not in checkpoint:

    raise RuntimeError(
        "Checkpoint does not contain model_state_dict."
    )


model.load_state_dict(
    checkpoint["model_state_dict"]
)


base_step = checkpoint.get(
    "step",
    "unknown"
)

print(
    "Loaded pretrained checkpoint."
)

print(
    "Base training step:",
    base_step
)

print()


# ============================================================
# PARAMETER COUNT
# ============================================================

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Parameters:",
    f"{parameter_count:,}"
)

print(
    "Parameters:",
    f"{parameter_count / 1e6:.2f}M"
)

print()


# ============================================================
# TOKENIZE TRAINING DATA
# ============================================================

print(
    "Tokenizing semantic dataset..."
)

tokenized_examples = []

for text in TRAIN_TEXTS:

    encoded = tokenizer.encode(
        text
    )

    ids = encoded.ids

    if len(ids) >= 2:

        tokenized_examples.append(
            ids
        )


print(
    "Tokenized examples:",
    len(tokenized_examples)
)


total_tokens = sum(
    len(x)
    for x in tokenized_examples
)

print(
    "Total dataset tokens:",
    f"{total_tokens:,}"
)

print()


# ============================================================
# CREATE OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# IMPORTANT:
# DO NOT LOAD THE OLD FINEWEB OPTIMIZER STATE
# ============================================================
#
# We intentionally create a fresh optimizer.
#
# FineWeb pretraining and semantic fine-tuning are different
# training stages.
#
# ============================================================


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=False
)


# ============================================================
# BATCH CREATION
# ============================================================

def create_training_sequence():

    # Randomly choose an example.
    index = torch.randint(
        0,
        len(tokenized_examples),
        (1,)
    ).item()

    ids = tokenized_examples[index]

    # Repeat if too short.
    while len(ids) < 2:

        index = torch.randint(
            0,
            len(tokenized_examples),
            (1,)
        ).item()

        ids = tokenized_examples[index]


    # --------------------------------------------------------
    # If the sequence is shorter than SEQ_LEN,
    # repeat it so the model gets a fixed-length sequence.
    #
    # IMPORTANT:
    # This keeps the example's language pattern.
    # --------------------------------------------------------

    if len(ids) < SEQ_LEN + 1:

        repeats = (
            (SEQ_LEN + 1)
            // len(ids)
        ) + 1

        ids = (
            ids * repeats
        )


    # Random starting position.

    if len(ids) > SEQ_LEN + 1:

        max_start = (
            len(ids)
            - SEQ_LEN
            - 1
        )

        start = torch.randint(
            0,
            max_start + 1,
            (1,)
        ).item()

    else:

        start = 0


    sequence = ids[
        start:
        start + SEQ_LEN + 1
    ]


    # Safety padding by repetition.

    while len(sequence) < SEQ_LEN + 1:

        sequence.append(
            ids[
                len(sequence)
                % len(ids)
            ]
        )


    return (
        sequence[:-1],
        sequence[1:]
    )


def get_batch():

    inputs = []
    targets = []

    for _ in range(BATCH_SIZE):

        x, y = create_training_sequence()

        inputs.append(x)
        targets.append(y)


    x = torch.tensor(
        inputs,
        dtype=torch.long,
        device=DEVICE
    )

    y = torch.tensor(
        targets,
        dtype=torch.long,
        device=DEVICE
    )

    return x, y


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    step,
    loss_value
):

    checkpoint_data = {

        "step": int(step),

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "config":
            config.__dict__,

        "loss":
            float(loss_value),

        "base_checkpoint":
            BASE_CHECKPOINT,

        "training_type":
            "semantic_instruction_finetuning",

        "examples":
            len(TRAIN_TEXTS),
    }


    torch.save(
        checkpoint_data,
        OUTPUT_CHECKPOINT
    )


    # Permanent archive.

    archive_name = (
        f"checkpoint-semantic-{step}"
        ".pt"
    )

    torch.save(
        checkpoint_data,
        archive_name
    )


    print()
    print(
        "Checkpoint saved:",
        OUTPUT_CHECKPOINT
    )

    print(
        "Archive saved:",
        archive_name
    )

    print()


# ============================================================
# TRAINING
# ============================================================

model.train()

print("=" * 70)
print("STARTING SEMANTIC TRAINING")
print("=" * 70)

print(
    "Base step:",
    base_step
)

print(
    "Fine-tuning steps:",
    MAX_STEPS
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Sequence length:",
    SEQ_LEN
)

print(
    "Learning rate:",
    LEARNING_RATE
)

print()


start_time = time.time()


for step in range(
    1,
    MAX_STEPS + 1
):

    step_start = time.time()


    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    input_ids, target_ids = get_batch()


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    optimizer.zero_grad(
        set_to_none=True
    )


    with torch.amp.autocast(
        "cuda",
        dtype=torch.bfloat16,
        enabled=USE_BF16
    ):

        logits, loss = model(
            input_ids,
            target_ids
        )


    # --------------------------------------------------------
    # Backward
    # --------------------------------------------------------

    loss.backward()


    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0
    )


    optimizer.step()


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    if (
        step % LOG_EVERY == 0
        or step == 1
    ):

        torch.cuda.synchronize()

        elapsed = (
            time.time()
            - step_start
        )

        tokens_per_sec = (
            BATCH_SIZE
            * SEQ_LEN
            / elapsed
        )

        vram = (
            torch.cuda.max_memory_allocated()
            / 1024**3
        )


        print(
            f"Step {step:5d}/{MAX_STEPS} | "
            f"Loss {loss.item():.4f} | "
            f"{tokens_per_sec:,.0f} tok/s | "
            f"VRAM {vram:.2f} GB"
        )


    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------

    if (
        step % CHECKPOINT_EVERY == 0
    ):

        save_checkpoint(
            step,
            loss.item()
        )


# ============================================================
# FINAL CHECKPOINT
# ============================================================

save_checkpoint(
    MAX_STEPS,
    loss.item()
)


# ============================================================
# SUMMARY
# ============================================================

total_time = (
    time.time()
    - start_time
)

total_tokens = (
    MAX_STEPS
    * BATCH_SIZE
    * SEQ_LEN
)

print()
print("=" * 70)
print("SEMANTIC TRAINING COMPLETE")
print("=" * 70)

print(
    "Base checkpoint:",
    BASE_CHECKPOINT
)

print(
    "Base step:",
    base_step
)

print(
    "Fine-tuning steps:",
    MAX_STEPS
)

print(
    "Final loss:",
    f"{loss.item():.4f}"
)

print(
    "Fine-tuning tokens:",
    f"{total_tokens:,}"
)

print(
    "Training time:",
    f"{total_time / 60:.2f} minutes"
)

print(
    "Checkpoint:",
    OUTPUT_CHECKPOINT
)

print("=" * 70)
