
# visualize_network.py
#
# REAL MODEL VISUALIZER
#
# This does NOT simulate a neural network.
#
# It loads:
#     checkpoint_fineweb.pt
#
# and runs:
#     SmallEnglishLLM
#
# The displayed nodes are selected dimensions from the
# REAL tensors produced by the model.
#
# The visualization is a projection of the real network,
# because displaying millions of individual parameters
# would be unusable.
#
# ============================================================

import os
import sys

import torch
import matplotlib.pyplot as plt
import networkx as nx

from tokenizers import Tokenizer

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from paths import CHECKPOINT_106K, TOKENIZER_PATH as PATHS_TOKENIZER

from model import SmallEnglishLLM
from model_config import ModelConfig


# ============================================================
# SETTINGS
# ============================================================

CHECKPOINT = CHECKPOINT_106K
TOKENIZER_PATH = PATHS_TOKENIZER
# Number of real dimensions shown per layer.
#
# Increase this later if desired.
DISPLAY_NEURONS = 16

# Prompt
PROMPT = "i am"


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
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

model = SmallEnglishLLM(
    config
).to(device)


checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print(
    "Checkpoint step:",
    checkpoint.get("step", "unknown")
)

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

print()


# ============================================================
# TOKENIZE PROMPT
# ============================================================

encoded = tokenizer.encode(
    PROMPT
)

token_ids = encoded.ids

print("Prompt:", PROMPT)

print(
    "Tokens:",
    encoded.tokens
)

print(
    "IDs:",
    token_ids
)

print()


x = torch.tensor(
    [token_ids],
    dtype=torch.long,
    device=device
)


# ============================================================
# CAPTURE REAL ACTIVATIONS
# ============================================================

activations = {}

hooks = []


def make_hook(name):

    def hook(module, inputs, output):

        # Some transformer modules return tuples.
        if isinstance(output, tuple):
            output = output[0]

        activations[name] = (
            output.detach()
            .float()
            .cpu()
        )

    return hook


# ------------------------------------------------------------
# Find transformer blocks
# ------------------------------------------------------------
#
# We inspect the model to locate the layers automatically.
#

print("Model modules:")

for name, module in model.named_modules():

    print(
        name,
        type(module).__name__
    )

print()


# ------------------------------------------------------------
# Register hooks on modules that look like transformer blocks
# ------------------------------------------------------------

for name, module in model.named_modules():

    class_name = type(module).__name__.lower()

    if (
        "block" in name.lower()
        or "block" in class_name
        or "transformer" in name.lower()
    ):

        hooks.append(
            module.register_forward_hook(
                make_hook(name)
            )
        )


# ============================================================
# REAL FORWARD PASS
# ============================================================

with torch.no_grad():

    logits, loss = model(
        x
    )


# ============================================================
# REMOVE HOOKS
# ============================================================

for hook in hooks:
    hook.remove()


# ============================================================
# FIND USEFUL ACTIVATIONS
# ============================================================

print()
print("=" * 70)
print("REAL ACTIVATIONS CAPTURED")
print("=" * 70)

for name, activation in activations.items():

    print(
        name,
        "shape =",
        tuple(activation.shape)
    )

print()


# ============================================================
# FALLBACK
# ============================================================
#
# If the automatic block detection didn't find anything,
# inspect the model's parameters instead.
#

if not activations:

    print(
        "No transformer blocks were automatically detected."
    )

    print(
        "We will inspect model architecture."
    )

    raise RuntimeError(
        "No activations captured. "
        "Send the printed model modules so the hooks "
        "can be connected to the exact architecture."
    )


# ============================================================
# SELECT ACTIVATION LAYERS
# ============================================================

layer_names = list(
    activations.keys()
)

print(
    "Visualization layers:"
)

for i, name in enumerate(
    layer_names
):

    print(
        i,
        name
    )

print()


# ============================================================
# CREATE GRAPH
# ============================================================

G = nx.DiGraph()

positions = {}

node_values = {}


# ------------------------------------------------------------
# Input nodes
# ------------------------------------------------------------

input_layer_name = "INPUT"

for i, token_id in enumerate(
    token_ids
):

    node = (
        f"input_{i}"
    )

    G.add_node(
        node,
        layer=0,
        value=0
    )

    positions[node] = (
        0,
        -i
    )


# ------------------------------------------------------------
# Activation layers
# ------------------------------------------------------------

layer_index = 1

previous_nodes = [
    f"input_{i}"
    for i in range(
        len(token_ids)
    )
]


for layer_name in layer_names:

    activation = activations[
        layer_name
    ]

    # --------------------------------------------------------
    # Expected shape:
    #
    # [batch, sequence, hidden]
    # --------------------------------------------------------

    if activation.ndim != 3:
        continue

    # Use the LAST token because this is the representation
    # used to predict the next token.

    values = activation[
        0,
        -1
    ]

    values = values[
        :DISPLAY_NEURONS
    ]

    values = values.numpy()

    current_nodes = []

    for neuron_index, value in enumerate(
        values
    ):

        node = (
            f"L{layer_index}_N{neuron_index}"
        )

        G.add_node(
            node,
            layer=layer_index,
            value=float(value)
        )

        positions[node] = (
            layer_index,
            -neuron_index
        )

        node_values[node] = float(
            value
        )

        current_nodes.append(
            node
        )

    # --------------------------------------------------------
    # Connect previous layer → current layer
    #
    # These connections represent the displayed pathway.
    #
    # The actual model has far more connections.
    # --------------------------------------------------------

    for previous in previous_nodes:

        for current in current_nodes:

            G.add_edge(
                previous,
                current
            )

    previous_nodes = current_nodes

    layer_index += 1


# ============================================================
# OUTPUT LAYER
# ============================================================

probabilities = torch.softmax(
    logits[0, -1],
    dim=-1
)

top_values, top_indices = torch.topk(
    probabilities,
    5
)

print("=" * 70)
print("REAL NEXT-TOKEN PREDICTIONS")
print("=" * 70)

for probability, token_id in zip(
    top_values,
    top_indices
):

    token_id = int(
        token_id
    )

    text = tokenizer.decode(
        [token_id]
    )

    print(
        repr(text),
        f"{float(probability) * 100:.3f}%"
    )

print()


# ============================================================
# DRAW NETWORK
# ============================================================

plt.figure(
    figsize=(18, 10)
)

# ------------------------------------------------------------
# Extract node values
# ------------------------------------------------------------

values = []

for node in G.nodes:

    values.append(
        node_values.get(
            node,
            0.0
        )
    )


# ------------------------------------------------------------
# Normalize activation values
# ------------------------------------------------------------

if values:

    max_abs = max(
        abs(v)
        for v in values
    )

    if max_abs == 0:
        max_abs = 1.0

    node_sizes = [
        150
        + 500
        * abs(v)
        / max_abs
        for v in values
    ]

else:

    node_sizes = 300


# ------------------------------------------------------------
# Draw
# ------------------------------------------------------------

nx.draw_networkx_edges(
    G,
    positions,
    alpha=0.08,
    width=0.5
)

nx.draw_networkx_nodes(
    G,
    positions,
    node_size=node_sizes,
    alpha=0.9
)


# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

labels = {}

for node in G.nodes:

    if node.startswith(
        "input_"
    ):

        index = int(
            node.split("_")[1]
        )

        if index < len(
            encoded.tokens
        ):

            labels[node] = (
                encoded.tokens[index]
            )


nx.draw_networkx_labels(
    G,
    positions,
    labels=labels,
    font_size=8
)


# ============================================================
# TITLE
# ============================================================

step = checkpoint.get(
    "step",
    "unknown"
)

plt.title(
    "REAL SmallEnglishLLM Neural Network Projection\n"
    f"Prompt: {PROMPT!r} | "
    f"Checkpoint step: {step} | "
    f"Parameters: "
    f"{sum(p.numel() for p in model.parameters()):,}"
)

plt.axis("off")

plt.tight_layout()

plt.show()

