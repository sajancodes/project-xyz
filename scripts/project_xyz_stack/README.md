# PROJECT XYZ — Master Training, Research, Reproducibility, and Scale-Up Manual

> Project XYZ is an independent language-model research project.
> This document is the execution manual for building, training, evaluating, scaling, and studying Project XYZ.
> The neural architecture is intentionally replaceable.
> The training system is designed to survive architecture changes.
> The model you ultimately train is yours to define.

---

## 0. Document purpose

This README is deliberately detailed.
It is meant to be followed while doing the work.
It is not a marketing page.
It is not a benchmark claim.
It is not a promise that a particular model size will work.
It is the engineering contract for Project XYZ.

Use it to answer five questions at every stage.

1. What are we trying to prove?
2. What exact inputs are being used?
3. What exact code and configuration are being used?
4. What evidence tells us the experiment worked?
5. What artifacts must be saved so the result can be reproduced?

If an experiment cannot answer those questions, it is not finished.

---

## 1. Project identity

Project name: **Project XYZ**.
Repository: `sajancodes/project-xyz`.
Primary research direction: train language models from scratch and study how useful behavior emerges.
Primary engineering direction: make every important training decision inspectable and reproducible.
Primary model rule: the architecture is not hard-coded into the training framework.
Primary data rule: every serious run must have a data manifest.
Primary checkpoint rule: intermediate checkpoints are research artifacts, not disposable files.
Primary evaluation rule: training loss alone never decides whether a model is good.
Primary scaling rule: prove correctness at small scale before buying or renting larger compute.

---

## 2. Non-negotiable rules

- Call the project Project XYZ.
- Keep model architecture decisions separate from trainer infrastructure.
- Keep tokenizer revisions versioned.
- Keep dataset provenance recorded.
- Keep train and validation data logically separated.
- Keep benchmark data out of training data where possible.
- Keep experiment configs in source control.
- Keep intermediate checkpoints for important runs.
- Keep exact global step in every checkpoint.
- Keep optimizer and scheduler state in resumable checkpoints.
- Keep random-number-generator state where practical.
- Keep package and hardware information in run manifests.
- Evaluate multiple checkpoints, not only the last checkpoint.
- Run deterministic evaluations before subjective generation tests.
- Treat free-form chat quality and benchmark accuracy as different measurements.
- Treat scripts as plans until a real run produces artifacts.
- Treat checkpoints as untrusted until they reload successfully.
- Treat data as untrusted until samples are inspected.
- Treat a low loss as suspicious when it drops too quickly.
- Treat sudden benchmark jumps as suspicious until leakage is ruled out.
- Treat a successful launch as only the beginning of validation.
- Never silently change architecture parameters to make a backend accept the model.
- Never silently change tokenizer special-token IDs.
- Never silently change dataset ordering between resume runs.
- Never overwrite the official baseline checkpoint without a verified replacement.
- Never claim a capability without an evaluation that measures it.

---

## 3. Start here

### 3.1 Clone and update

```bash
git clone https://github.com/sajancodes/project-xyz.git
cd project-xyz
git pull
```

### 3.2 Create an environment

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3.3 Install CUDA-enabled PyTorch

Install the PyTorch build that matches the CUDA runtime supported by your system.
Do not install a CPU-only build by accident.
Verify before doing anything else.

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA GPU')"
```

Pass condition:

```text
CUDA available = True
GPU name is your NVIDIA GPU
```

Fail condition:

```text
CUDA available = False
```

If it fails, fix PyTorch before debugging Project XYZ.

### 3.4 Install Project XYZ stack dependencies

```bash
python -m pip install -r project_xyz_stack/requirements.txt
```

### 3.5 Validate the model bridge

```bash
python project_xyz_stack/validate_stack.py
```

The validator must:

- instantiate the configured Project XYZ model;
- create random token inputs;
- execute a forward pass;
- compute a finite loss;
- execute the model-output adapter;
- verify the logits shape;
- print the parameter count.

Do not begin corpus training until this passes.

---

## 4. Repository map

```text
project-xyz/
├─ README.md
├─ PROJECT_SUMMARY.md
├─ TRAINING_HISTORY.md
├─ BENCHMARK_HISTORY.md
├─ tokenizer.json
├─ docs/
├─ experiments/
├─ src/
│  ├─ config/
│  ├─ evaluation/
│  ├─ models/
│  ├─ training/
│  └─ utils/
└─ project_xyz_stack/
   ├─ README.md
   ├─ PROJECT_XYZ_TRAINING.md
   ├─ PROJECT_XYZ_DISTRIBUTED_BACKEND.md
   ├─ bootstrap_project_xyz.py
   ├─ checkpointing.py
   ├─ evaluate.py
   ├─ model_adapter.py
   ├─ prepare_data.py
   ├─ requirements.txt
   ├─ train.py
   ├─ validate_stack.py
   └─ config/
      └─ project_xyz.json
```

The `src/` tree contains the existing Project XYZ research implementation.
The `project_xyz_stack/` tree contains the architecture-independent training framework.

---

## 5. Project XYZ architecture contract

The trainer must not assume one exact architecture.
The model adapter loads a model file and a config file dynamically.
The default points at the current Project XYZ implementation.

Configuration fields:

```json
{
  "model_file": "src/models/model.py",
  "model_class": "SmallEnglishLLM",
  "config_file": "src/config/model_config.py",
  "config_class": "ModelConfig",
  "config_kwargs": {}
}
```

To train a new model architecture:

1. create the new model implementation;
2. create its configuration dataclass or class;
3. point `model_file` at it;
4. point `model_class` at the class name;
5. point `config_file` at its configuration;
6. point `config_class` at the configuration class;
7. add architecture values under `config_kwargs` if required;
8. run `validate_stack.py`;
9. count parameters independently;
10. run a tiny overfit test before a real corpus run.

Expected model behavior:

- input is a tensor of token IDs with shape `[batch, sequence]`;
- output may be `(logits, loss)`;
- output may be a framework object exposing `.logits` and optionally `.loss`;
- output may be logits only;
- logits shape should be `[batch, sequence, vocab]`;
- if loss is missing, the adapter calculates cross-entropy against targets.

---

## 6. Current small-model reference

The existing repository contains a small decoder-only Transformer used for experiments.
It is a reference implementation, not a permanent architecture requirement.

Current historical configuration:

```text
vocabulary: 16,000
maximum sequence length: 512
model dimension: 256
layers: 8
attention heads: 8
feed-forward dimension: 1,024
parameter count: approximately 14.66M
```

The current model has been useful for:

- learning the training pipeline;
- verifying FineWeb ingestion;
- studying catastrophic forgetting;
- testing semantic fine-tuning;
- testing checkpoint lineage;
- testing held-out evaluation;
- studying differences between benchmark scores and free generation;
- preparing future weight-map and activation experiments.

Do not assume the final Project XYZ model must retain these dimensions.

---

## 7. Define the next model before training it

Create a model specification before writing code.

Record:

- model name;
- architecture family;
- target parameter count;
- vocabulary size;
- maximum context length;
- embedding dimension;
- layer count;
- attention-head count;
- KV-head count;
- head dimension;
- FFN dimension;
- activation function;
- normalization type;
- normalization epsilon;
- positional encoding;
- positional encoding base/frequency settings;
- embedding tying policy;
- output-head tying policy;
- bias policy;
- dropout policy;
- initialization rule;
- attention implementation;
- dense versus expert layers;
- expert count if applicable;
- active experts per token if applicable;
- router loss if applicable;
- expected parameter count;
- expected parameter memory;
- expected optimizer-state memory;
- expected activation memory at target batch/context;
- minimum hardware target;
- scale-up hardware target.

A model without a written specification is too easy to change accidentally.

---

## 8. Parameter-count verification

Count parameters from the instantiated model.
Do not trust a spreadsheet alone.
Do not trust a model name alone.

Use:

```python
sum(p.numel() for p in model.parameters())
```

Also record trainable parameters:

```python
sum(p.numel() for p in model.parameters() if p.requires_grad)
```

Validate:

- embeddings are counted;
- attention projections are counted;
- FFN projections are counted;
- normalization weights are counted;
- biases are counted if used;
- output head is counted;
- tied weights are not double-counted in memory estimates;
- expert parameters are counted correctly;
- router parameters are counted correctly.

---

## 9. Memory planning

Before training, estimate memory.

Approximate components:

```text
parameters
parameter gradients
optimizer states
master weights if used
activations
attention intermediates
temporary kernel buffers
CUDA allocator fragmentation
dataloader pinned memory
```

For AdamW in ordinary mixed-precision training, optimizer state can dominate parameter memory.

Measure actual peak allocation after launch:

```python
torch.cuda.max_memory_allocated()
```

Also inspect reserved memory:

```python
torch.cuda.max_memory_reserved()
```

A model that fits with 50 MB free is not a stable training configuration.
Leave headroom.

---

## 10. RTX 4050 6 GB development strategy

The laptop GPU is useful for Project XYZ development.
It is not a datacenter cluster.
Use it where it has leverage.

Good laptop tasks:

- tokenizer experiments;
- tiny model pretraining;
- current small-model training;
- short architecture sweeps;
- data-pipeline verification;
- overfit tests;
- checkpoint tests;
- evaluation development;
- activation collection;
- weight-difference analysis;
- layer-output analysis;
- inference benchmarks;
- low-rank fine-tuning experiments on compatible pretrained models;
- debugging distributed-ready code on a single device.

Poor laptop tasks:

- training multi-billion-parameter dense models from scratch;
- long-context giant-batch experiments;
- large expert models with all weights resident;
- full optimizer-state training of models far beyond VRAM;
- cluster-level reinforcement-learning workloads.

The correct strategy is to prove ideas cheaply, then scale only proven ideas.

---

## 11. Tokenizer policy

The tokenizer is part of the model.
Changing it changes the meaning of embedding rows.
Version it like code.

For every tokenizer record:

- tokenizer type;
- vocabulary size;
- training corpus;
- corpus revision;
- number of training documents;
- normalizer;
- pre-tokenizer;
- decoder;
- special tokens;
- special-token IDs;
- byte fallback behavior;
- unknown-token behavior;
- BOS policy;
- EOS policy;
- padding policy;
- tokenizer file hash.

Project XYZ currently tracks `tokenizer.json`.

---

## 12. Tokenizer acceptance tests

Create a tokenizer test suite.
Include:

- ordinary English;
- punctuation;
- numbers;
- decimals;
- code;
- URLs;
- Unicode;
- Nepali text if multilingual behavior is planned;
- emoji;
- repeated whitespace;
- newlines;
- tabs;
- markdown;
- JSON;
- Python;
- JavaScript;
- mathematical notation;
- long words;
- rare names;
- malformed UTF-8 inputs where the library allows testing.

Measure:

- average tokens per character;
- average tokens per word;
- code tokenization efficiency;
- number tokenization behavior;
- special-token correctness.

---

## 13. Tokenizer freeze rule

Once a serious pretraining run starts, freeze the tokenizer for that model lineage.
If a new tokenizer is required, create a new lineage.

Do not:

- reorder vocabulary IDs;
- change BOS ID;
- change EOS ID;
- change padding ID;
- replace merges silently;
- retrain the tokenizer and reuse old embeddings as if nothing changed.

---

## 14. Data philosophy

Model quality is constrained by data quality.
More tokens do not automatically mean better tokens.

Project XYZ data work has five goals:

1. legality and provenance;
2. useful domain coverage;
3. quality filtering;
4. duplicate control;
5. reproducible mixture construction.

The training stack should know exactly what it consumed.

---

## 15. Data source registry

Maintain a table for every source.

Fields:

```text
source_name
source_type
source_url_or_path
source_revision
license_or_terms
language
estimated_documents
estimated_tokens
quality_notes
known_biases
known_duplicates
PII_risk
benchmark_overlap_risk
processing_version
included_or_excluded
```

Do not rely on memory for this registry.

---

## 16. Data ingestion

The lightweight Project XYZ preparer supports:

- streaming Hugging Face datasets;
- local JSONL files;
- configurable text field;
- EOS insertion;
- tokenizer hashing;
- token-count recording;
- document-count recording;
- compact binary token storage.

Example:

```bash
python project_xyz_stack/prepare_data.py \
  --dataset HuggingFaceFW/fineweb \
  --dataset-config sample-10BT \
  --split train \
  --tokenizer tokenizer.json \
  --output data/project_xyz/train.bin \
  --max-docs 10000
```

Windows PowerShell:

```powershell
python project_xyz_stack/prepare_data.py `
  --dataset HuggingFaceFW/fineweb `
  --dataset-config sample-10BT `
  --split train `
  --tokenizer tokenizer.json `
  --output data/project_xyz/train.bin `
  --max-docs 10000
```

---

## 17. Prepared-data format

The local pipeline writes a flat token stream.
Each document may end with EOS.
The binary dtype is selected from vocabulary size.

Typical dtype:

```text
uint16 when vocab <= 65,535
uint32 otherwise
```

Each binary file should have a sidecar manifest:

```text
train.bin
train.bin.meta.json
```

The manifest is required evidence for serious runs.

---

## 18. Data manifest fields

Current manifest fields include:

- format version;
- source type;
- source value;
- dataset configuration;
- split;
- text field;
- tokenizer path;
- tokenizer SHA-256;
- vocabulary size;
- binary dtype;
- EOS token ID;
- document count;
- token count;
- maximum-document limit.

Future large-scale manifests should add:

- source revision hash;
- filter revision;
- dedupe revision;
- mixture ID;
- shard list;
- shard hashes;
- language distribution;
- domain distribution;
- quality distribution;
- benchmark decontamination results.

---

## 19. Raw-data inspection

Before tokenizing millions of documents, inspect raw samples.

Inspect at least:

- first 100 samples;
- random 100 samples;
- shortest samples;
- longest samples;
- high-symbol samples;
- high-number samples;
- repeated samples;
- malformed samples;
- non-target-language samples;
- HTML-heavy samples;
- navigation-page samples;
- SEO-spam samples;
- machine-generated boilerplate;
- code blocks if code is a target domain.

Write findings down.

---

## 20. Quality filtering

Potential quality signals:

- document length;
- alphabetic-character ratio;
- punctuation ratio;
- repeated-line ratio;
- repeated-ngram ratio;
- language confidence;
- boilerplate score;
- URL density;
- profanity policy if relevant;
- code-to-text ratio;
- entropy-like repetition metrics;
- classifier-based quality score;
- educational-content score;
- citation/reference density;
- formatting integrity.

Do not apply filters blindly.
Measure what they remove.

---

## 21. Deduplication

Duplicate control matters because repetition changes effective data weighting.

Consider:

- exact document hashes;
- normalized document hashes;
- MinHash;
- SimHash;
- n-gram fingerprints;
- URL canonicalization;
- near-duplicate clustering;
- repeated boilerplate stripping before similarity comparison.

Track duplicate rates per source.

---

## 22. Benchmark decontamination

If Project XYZ is evaluated on public benchmarks, reduce contamination risk.

Procedure:

1. collect benchmark questions and reference text;
2. normalize text;
3. create n-gram fingerprints;
4. search corpus shards;
5. flag exact matches;
6. flag near matches;
7. remove contaminated training documents when feasible;
8. record the method;
9. report residual uncertainty.

A high score is less useful if the test was in training data.

---

## 23. Language policy

Decide whether a model is:

- English-only;
- English-primary multilingual;
- balanced multilingual;
- domain multilingual.

Then measure actual token distribution.
Do not call a model multilingual because a few non-English documents leaked into the corpus.

---

## 24. Code-data policy

If coding ability matters, define code data explicitly.

Record:

- languages;
- repository-license rules;
- generated-file filters;
- vendor-directory filters;
- minified-code filters;
- duplicate-file handling;
- test-code handling;
- documentation handling;
- notebook handling;
- issue/commit-message handling if included.

Evaluate coding separately from general language modeling.

---

## 25. Math-data policy

Math data should include more than final answers.
Possible categories:

- arithmetic;
- algebra;
- geometry;
- probability;
- calculus;
- discrete math;
- word problems;
- proofs;
- symbolic transformations;
- worked examples;
- textbook exposition.

Track answer leakage and duplicated benchmark questions.

---

## 26. Data mixture design

A mixture is an optimization decision.
It changes what gradients the model sees.

For each component record:

```text
source
weight
token budget
sampling method
quality threshold
language/domain label
```

Do not only report percentages.
Report actual consumed tokens when the run finishes.

---

## 27. Mixture experiments

When comparing mixtures:

- keep architecture fixed;
- keep tokenizer fixed;
- keep optimizer fixed;
- keep token budget fixed where possible;
- change one mixture factor at a time;
- evaluate identical checkpoints;
- compare both aggregate and per-domain metrics.

A mixture can improve one domain while silently hurting another.

---

## 28. Train/validation split

Validation data must not be sampled from the training stream after training starts.
Create a stable validation split.
Version it.
Keep it fixed across comparable experiments.

Validation should represent:

- target language;
- major domains;
- normal document lengths;
- rare but important domains;
- code/math if those are important.

---

## 29. Validation leakage test

Check:

- exact overlap;
- near-duplicate overlap;
- same URL appearing in both splits;
- mirrored pages;
- repeated syndicated content;
- templated pages differing only by a few fields.

Validation loss is misleading if validation documents are duplicates of training documents.

---

## 30. Packing policy

The local trainer uses contiguous token blocks.
For large-scale training, efficient packing reduces padding waste.

Record:

- whether documents can cross sequence boundaries;
- whether multiple documents share a sequence;
- whether EOS separates documents;
- whether attention can cross document boundaries;
- how long documents are truncated or split;
- whether segment IDs are used.

Packing semantics can affect training behavior.

---

## 31. Sequence length

Context length is both a model and training decision.

Longer context:

- increases activation memory;
- increases attention cost;
- changes tokens per batch;
- can reduce batch size;
- may require different positional settings;
- may require different data packing;
- may change learning dynamics.

Do not increase context length just because a kernel supports it.

---

## 32. Context curriculum

Possible strategy:

1. pretrain mostly at shorter context;
2. verify language quality;
3. extend positional settings if architecture supports it;
4. continue training with longer sequences;
5. add long-document data;
6. evaluate retrieval and long-range dependencies;
7. verify short-context regressions.

Treat context extension as its own experiment.

---

## 33. Training configuration

Main configuration:

```text
project_xyz_stack/config/project_xyz.json
```

Current sections:

- run name;
- seed;
- model;
- data;
- training;
- output.

Every production run should copy the final resolved configuration into its run directory.

---

## 34. Run naming

Recommended pattern:

```text
project_xyz_<model>_<data>_<context>_<tokens>_<date>_<revision>
```

Example:

```text
project_xyz_dense120m_webcode_2048_8b_20261016_r1
```

A good run name helps identify the experiment without opening its config.

---

## 35. Random seeds

Record the seed.
Use more than one seed when comparing small differences.

One seed can exaggerate an apparent win.

Seed affects:

- initialization;
- data order;
- dropout;
- sampling;
- some kernel paths;
- distributed worker behavior.

Exact bitwise reproducibility is not always possible across hardware, but seed control still matters.

---

## 36. Initialization

Initialization can decide whether a deep model trains at all.

Record:

- distribution;
- standard deviation;
- layer-specific scaling;
- residual scaling;
- embedding initialization;
- router initialization for experts;
- output-head initialization.

Check initial activation statistics before long training.

---

## 37. Initialization diagnostics

Before optimizer steps, inspect:

- mean activation;
- activation standard deviation;
- max absolute activation;
- logits standard deviation;
- initial loss;
- gradient norm after one backward pass;
- layer-by-layer gradient norm;
- fraction of zero gradients;
- NaN/Inf count.

Large outliers at step zero usually become expensive failures later.

---

## 38. Optimizer

Default local optimizer: AdamW.

Current configurable fields:

- learning rate;
- beta1;
- beta2;
- epsilon;
- weight decay.

Current starting defaults:

```text
beta1 = 0.9
beta2 = 0.95
weight_decay = 0.1
epsilon = 1e-8
```

These are starting points, not universal constants.

---

## 39. Weight-decay policy

Consider excluding some parameters from weight decay:

- normalization weights;
- biases;
- router biases;
- embeddings depending on the chosen recipe.

If Project XYZ changes from uniform decay to parameter groups, document it explicitly.

---

## 40. Learning-rate search

Do short sweeps before long runs.

Example candidates:

```text
1e-4
2e-4
3e-4
5e-4
8e-4
```

The correct range depends on:

- model size;
- initialization;
- optimizer;
- global batch;
- data quality;
- context length;
- precision;
- normalization.

Compare early loss slope and stability.

---

## 41. Warmup

Warmup reduces shock at the beginning of training.

Record warmup in both:

- optimizer steps;
- approximate tokens.

For experiments, compare warmup fractions rather than copying one fixed number to every token budget.

---

## 42. Cosine decay

The local trainer uses warmup followed by cosine decay.

The minimum learning-rate ratio is configurable.

Questions to test:

- should LR reach near zero?
- should it end at 10% of peak?
- should there be a constant tail?
- should continued pretraining restart the schedule?

These are experimental choices.

---

## 43. Effective global batch

Calculate it explicitly.

```text
effective_batch = micro_batch
                × gradient_accumulation
                × data_parallel_world_size
```

Tokens per optimizer step:

```text
tokens_per_step ≈ effective_batch × sequence_length
```

Record both numbers.

---

## 44. Gradient accumulation

Use accumulation when the desired effective batch does not fit in memory.

Tradeoffs:

- more optimizer-step latency;
- fewer synchronization events in distributed training;
- same approximate effective batch if implemented correctly;
- micro-batch-dependent kernel efficiency.

Verify that loss is divided correctly before backward.

---

## 45. Gradient clipping

The local default clips global gradient norm.

Current starting value:

```text
1.0
```

Log pre-clipping norm during diagnostics if possible.
Frequent clipping may indicate:

- learning rate too high;
- unstable architecture;
- bad batch;
- precision problems;
- initialization problems.

---

## 46. Mixed precision

Preferred local precision: BF16 when supported.
Fallback: FP16.

BF16 advantages:

- larger exponent range;
- often fewer overflow problems.

FP16 requirements:

- gradient scaling is commonly necessary;
- overflow monitoring is important.

Always verify actual dtype paths in the profiler.

---

## 47. Precision acceptance test

Run the same short job in:

- FP32 where feasible;
- BF16;
- FP16 if supported.

Compare:

- initial loss;
- loss after N steps;
- gradient norms;
- NaN/Inf behavior;
- memory;
- throughput.

Large quality divergence requires investigation.

---

## 48. Compile mode

`torch.compile` can improve performance in some environments.
It can also complicate debugging.

Policy:

1. get eager mode correct;
2. validate checkpoints in eager mode;
3. benchmark compile mode;
4. compare loss behavior;
5. only then enable it for long runs.

---

## 49. Dataloader design

The local token dataset memory-maps the binary stream.
Benefits:

- low startup memory;
- simple indexing;
- fast sequential access;
- no need to load all tokens into RAM.

Monitor:

- worker stalls;
- page faults;
- storage throughput;
- CPU overhead;
- pinned-memory behavior.

---

## 50. Overfit-a-batch test

Before corpus training, prove the model can memorize one tiny batch.

Procedure:

1. sample one fixed batch;
2. disable data shuffling;
3. train only that batch;
4. run 100–1000 optimizer steps depending on size;
5. verify loss falls strongly;
6. verify generated next-token probabilities concentrate correctly.

If the model cannot overfit one batch, do not train on billions of tokens.

---

## 51. Tiny-corpus test

Next, train on a tiny real corpus.

Goals:

- data decoding works;
- EOS behavior works;
- checkpoint save works;
- checkpoint reload works;
- validation runs;
- throughput logging works;
- loss decreases without exploding.

Keep this run cheap.

---

## 52. Smoke-test token budget

A smoke test should finish quickly enough to rerun often.

Possible scale:

```text
100–1000 steps
small sequence length
tiny data subset
frequent logging
one checkpoint
one evaluation
```

A smoke test is for correctness, not final quality.

---

## 53. First real pretraining run

Only launch after:

- validator passes;
- tokenizer tests pass;
- data samples look correct;
- data manifest exists;
- tiny-batch overfit succeeds;
- tiny-corpus loss falls;
- checkpoint reload succeeds;
- evaluation script succeeds;
- disk space is sufficient;
- thermal behavior is acceptable;
- power settings are stable.

---

## 54. Local training command

```bash
python project_xyz_stack/train.py \
  --config project_xyz_stack/config/project_xyz.json
```

Windows PowerShell:

```powershell
python project_xyz_stack/train.py `
  --config project_xyz_stack/config/project_xyz.json
```

The trainer requires CUDA by default.

---

## 55. Training logs

At minimum log:

- global step;
- loss;
- learning rate;
- tokens per second;
- peak allocated VRAM;
- evaluation loss;
- evaluation perplexity;
- checkpoint path.

For serious runs add:

- gradient norm;
- optimizer-step time;
- data-loading time;
- GPU utilization;
- GPU power;
- GPU temperature;
- CPU utilization;
- host RAM;
- storage throughput;
- tokens consumed.

---

## 56. Throughput interpretation

Tokens per second is useful only with context.
Record:

- sequence length;
- micro batch;
- accumulation;
- precision;
- model size;
- GPU model;
- number of GPUs;
- compile setting;
- attention kernel;
- checkpointing/recomputation setting.

Do not compare throughput numbers with different workloads as if they were equivalent.

---

## 57. Loss interpretation

Training loss should generally decrease.
But a very low loss can mean memorization.

Ask:

- is validation loss also improving?
- is held-out task performance improving?
- is the model repeating training text?
- did data accidentally become repetitive?
- did a bug shift targets incorrectly?
- did the validation split leak?

Loss is necessary evidence, not sufficient evidence.

---

## 58. Perplexity

Perplexity is derived from mean cross-entropy loss.

```text
perplexity = exp(loss)
```

The evaluator caps exponent input for numerical practicality.
Use loss for precise comparison when values are extreme.

Perplexity comparisons require the same tokenizer and evaluation data.

---

## 59. Checkpoint cadence

Checkpoint often enough to study training evolution.
Too frequent:

- wastes storage;
- reduces throughput;
- creates management overhead.

Too sparse:

- loses recovery points;
- hides when capabilities emerged;
- weakens weight-map research.

Choose cadence based on run cost and research goals.

---

## 60. Checkpoint contents

Project XYZ resumable checkpoints store:

- model state;
- optimizer state;
- scheduler state;
- gradient-scaler state when applicable;
- Python RNG state;
- NumPy RNG state;
- PyTorch CPU RNG state;
- CUDA RNG state when available;
- configuration;
- global step.

Format identifier:

```text
project-xyz-checkpoint-v2
```

---

## 61. Atomic checkpointing

Checkpoint files are written to a temporary path first.
Then they replace the destination atomically where supported.

Reason:

- process crashes should be less likely to leave a half-written official checkpoint.

Still verify after save.

---

## 62. Latest checkpoint

`latest.pt` is a convenience pointer implemented as a full saved payload.
Do not assume latest means best.

Best checkpoint selection must come from evaluation.

---

## 63. Checkpoint pruning

The trainer can retain only the last N milestone files.
Use pruning carefully.

For research runs where capability emergence matters, archive milestone checkpoints before pruning.

Recommended categories:

- recovery checkpoints;
- research milestone checkpoints;
- official baseline checkpoints;
- release checkpoints.

They should have different retention policies.

---

## 64. Resume test

Mandatory before a long run.

Procedure:

1. train to step A;
2. save;
3. stop the process;
4. restart;
5. confirm resume reports step A;
6. train to step B;
7. inspect LR continuity;
8. inspect loss continuity;
9. confirm checkpoint B loads;
10. compare against an uninterrupted short control run.

---

## 65. Run manifest

Each run directory should contain `run_manifest.json`.

Current manifest includes:

- project name;
- resolved config;
- PyTorch version;
- CUDA runtime;
- CUDA availability;
- GPU name.

Extend it over time with:

- git commit;
- hostname;
- OS;
- Python version;
- package lock hash;
- tokenizer hash;
- data manifest hashes;
- start/end timestamps;
- total tokens;
- final checkpoint hash.

---

## 66. Git revision policy

Every important experiment should point to a git commit.
Do not run major experiments from a dirty working tree unless the diff is saved into the run artifacts.

Recommended process:

```text
edit
unit test
smoke test
commit
launch experiment
record commit SHA
```

---

## 67. Experiment folder

Recommended structure:

```text
runs/
└─ <run_name>/
   ├─ run_manifest.json
   ├─ resolved_config.json
   ├─ data_manifest_refs.json
   ├─ logs/
   ├─ eval/
   ├─ step_00000500.pt
   ├─ step_00001000.pt
   └─ latest.pt
```

For large runs, checkpoints may live on separate storage.
Store references and hashes in the run directory.

---

## 68. Evaluation philosophy

Evaluate what you care about.
Do not infer one capability from another.

Separate:

- language-modeling quality;
- factual recall;
- in-context binding;
- instruction following;
- reasoning;
- arithmetic;
- code;
- long-context behavior;
- free-form generation;
- safety/robustness if relevant.

---

## 69. Deterministic evaluation

Use greedy decoding or fixed seeds for canonical comparisons.

Why:

- sampled decoding adds noise;
- a model can appear better or worse because of randomness;
- regression tracking needs stable outputs.

Keep sampled generation as a separate qualitative test.

---

## 70. Existing Project XYZ evaluation assets

The repository already contains multiple evaluation scripts under `src/evaluation/`.
Use them as historical baselines while designing the next unified suite.

Historical categories include:

- grammar;
- factual QA;
- entity relations;
- paraphrase;
- instruction following;
- reasoning;
- negation;
- entity tracking;
- arithmetic;
- sun-direction facts;
- deterministic canonical scoring;
- free generation.

---

## 71. Evaluation versioning

Benchmarks change too.
Version:

- question set;
- expected answers;
- normalization rules;
- exact-match rules;
- partial-credit rules;
- generation settings;
- metric implementation.

A score without an evaluator revision is incomplete.

---

## 72. Benchmark split discipline

Create:

- train examples;
- validation examples;
- hidden held-out examples.

Do not tune repeatedly on the hidden held-out set.
Otherwise it becomes a validation set.

---

## 73. Free-generation tests

Maintain a stable set of prompts.
Include:

- plain questions;
- ambiguous questions;
- multi-step instructions;
- long prompts;
- code requests;
- refusal/robustness cases if relevant;
- creative generation;
- factual questions;
- entity-binding prompts.

Save raw outputs.

---

## 74. Checkpoint sweep evaluation

For every important run:

1. list milestones;
2. evaluate each milestone;
3. plot validation loss versus step;
4. plot benchmark score versus step;
5. identify capability peaks;
6. identify regressions;
7. select best checkpoint by declared criteria;
8. keep final checkpoint separately from best checkpoint.

---

## 75. Official baseline policy

An official baseline must have:

- checkpoint path;
- checkpoint hash;
- architecture config;
- tokenizer hash;
- data lineage;
- training config;
- evaluation revision;
- metric table;
- generation samples;
- git commit.

Replacing the baseline requires evidence that the replacement is better under the declared protocol.

---

## 76. Regression gates

Define categories that must not fall below thresholds.
Example:

```text
validation loss: no significant regression
factual QA: >= baseline - tolerance
instruction: >= baseline - tolerance
entity tracking: >= baseline - tolerance
free generation: manual regression review
```

A single improved aggregate score should not hide catastrophic domain regressions.

---

## 77. Capability-emergence research

Project XYZ can study when capabilities appear.

Method:

- save checkpoints frequently;
- run the same evaluations at each checkpoint;
- store raw outputs;
- compare weight deltas;
- compare activation deltas;
- correlate metric changes with parameter/layer changes;
- test whether changes persist at later checkpoints.

This is stronger than inspecting only the final model.

---

## 78. Weight-delta analysis

For checkpoints A and B calculate:

```text
ΔW = W_B - W_A
```

Measure per tensor:

- L1 norm;
- L2 norm;
- max absolute change;
- mean absolute change;
- relative change versus weight norm;
- changed-sign fraction;
- cosine similarity.

Rank layers by change.

---

## 79. Weight-map project

Goal:

Identify whether particular layers, heads, neurons, or parameter groups change disproportionately when specific capabilities improve.

Do not assume one capability maps to one neuron.
Modern language models use distributed representations.
The experiment should test localization rather than presuppose it.

---

## 80. Weight-map experimental design

For each capability:

1. choose a stable benchmark;
2. collect baseline checkpoint;
3. train a controlled continuation emphasizing the capability;
4. collect multiple intermediate checkpoints;
5. measure metric change;
6. compute parameter deltas;
7. collect activations on fixed prompts;
8. compare attention/MLP statistics;
9. run ablations on candidate components;
10. test whether ablation selectively damages the capability.

Selective causal effects are stronger evidence than correlation.

---

## 81. Activation collection

Use forward hooks carefully.
Possible targets:

- embedding output;
- attention Q/K/V projections;
- attention output;
- residual stream;
- normalization output;
- MLP input;
- MLP intermediate;
- MLP output;
- final hidden state;
- logits.

Do not store full activations for huge datasets without a storage plan.

---

## 82. Activation statistics

Useful summaries:

- mean;
- standard deviation;
- RMS;
- sparsity;
- max absolute value;
- kurtosis;
- cosine similarity between prompts;
- principal components;
- token-wise norm;
- layer-wise norm.

Use fixed prompt sets for checkpoint comparisons.

---

## 83. Attention-head analysis

Do not call a head a “coding head” from one visualization.

Better evidence:

- head pattern is consistent across many coding prompts;
- pattern differs from matched non-code prompts;
- ablating the head changes coding performance;
- unrelated tasks are less affected;
- result replicates across seeds/checkpoints.

---

## 84. MLP-neuron analysis

Possible analysis:

- activation frequency;
- top activating tokens;
- top activating contexts;
- checkpoint activation drift;
- output-vector similarity;
- targeted ablation;
- activation patching.

Again, expect distributed features.

---

## 85. Activation patching

Concept:

1. run a clean prompt;
2. run a corrupted prompt;
3. replace an internal activation in the corrupted run with the clean activation;
4. measure whether correct behavior is restored.

This can provide causal evidence about where information flows.

---

## 86. Layer ablation

Possible ablations:

- zero one attention output;
- zero one MLP output;
- skip one block;
- mask one head;
- scale one residual branch;
- perturb selected weights.

Always compare multiple tasks to detect nonspecific damage.

---

## 87. Checkpoint interpolation

Research experiment:

Interpolate weights between two checkpoints.

```text
W(alpha) = (1-alpha) * W_A + alpha * W_B
```

Evaluate several alpha values.
This can reveal whether a capability changes smoothly along the optimization path.

Do not use interpolation across unrelated architectures.

---

## 88. Parameter freezing experiments

To test where learning occurs:

- freeze embeddings;
- freeze early layers;
- freeze late layers;
- train only attention;
- train only MLPs;
- train only normalization;
- train selected blocks.

Compare learning speed and final capability.

---

## 89. Gradient analysis

Collect per-layer gradient statistics.

Measure:

- norm;
- mean;
- max absolute value;
- zero fraction;
- sign distribution;
- update-to-weight ratio.

Compare across task-specific batches.

---

## 90. Gradient similarity

For two data domains A and B:

1. compute gradient vector summaries for A;
2. compute gradient vector summaries for B;
3. compare cosine similarity by layer;
4. identify conflicting layers;
5. test mixture weights that reduce destructive interference.

This can help explain catastrophic forgetting.

---

## 91. Catastrophic forgetting protocol

When adding a new capability:

- evaluate old capabilities before training;
- train on new data;
- evaluate new capability;
- re-evaluate all old capabilities;
- inspect validation loss;
- inspect free generation;
- compare weight/gradient changes.

A new capability is not a win if it destroys core language behavior.

---

## 92. Mixed-domain continuation

To reduce forgetting:

- mix new-task data with representative pretraining data;
- tune mixture ratio;
- use lower learning rate;
- use a decay schedule;
- stop at the best checkpoint rather than a fixed final step.

Project XYZ historical experiments already showed why this matters.

---

## 93. Curriculum experiments

Possible curricula:

- general text then code;
- general text then math;
- short context then long context;
- clean high-quality data then broader data;
- easy reasoning then harder reasoning;
- dense model then expert conversion is generally not trivial and requires explicit design.

Test curriculum claims instead of assuming them.

---

## 94. Architecture ablation policy

Change one major architectural factor at a time where practical.

Candidate ablations:

- learned positions versus rotary positions;
- LayerNorm versus RMSNorm;
- GELU versus SwiGLU;
- MHA versus GQA;
- tied versus untied output embeddings;
- bias versus no bias;
- different head dimensions;
- different FFN ratios;
- different depth/width tradeoffs.

Keep token budget comparable.

---

## 95. Depth versus width

At similar parameter count compare:

- deeper/narrower;
- shallower/wider.

Measure:

- training speed;
- memory;
- validation loss;
- reasoning tasks;
- factual tasks;
- generation quality;
- activation stability.

Do not choose based only on parameter count.

---

## 96. Attention-head count

Changing heads changes head dimension unless model dimension changes too.

Check:

- divisibility;
- kernel support;
- attention entropy;
- memory;
- performance.

Too many tiny heads can be inefficient.

---

## 97. Grouped-query attention

GQA can reduce KV-cache cost for inference.
Research questions:

- quality impact at Project XYZ scale;
- training throughput impact;
- memory impact;
- best KV-head ratio;
- compatibility with chosen kernels.

Keep head-dimension semantics correct.

---

## 98. Rotary positions

If using rotary positions, record:

- rotary dimension;
- base/frequency parameter;
- scaling method;
- context-extension method;
- implementation revision.

Context behavior can change dramatically with these settings.

---

## 99. Normalization

Compare normalization choices carefully.

Record:

- pre-norm versus post-norm;
- LayerNorm versus RMSNorm;
- epsilon;
- whether normalization has bias;
- residual scaling.

Inspect activation and gradient stability.

---

## 100. Feed-forward block

Record:

- activation;
- hidden dimension;
- gating method;
- bias policy;
- dropout;
- expert routing if applicable.

FFN parameters often dominate model size.

---

## 101. SwiGLU experiments

If using SwiGLU:

- define exact projection layout;
- define hidden dimension calculation;
- record parameter count;
- benchmark kernel performance;
- compare quality at matched compute.

Do not copy an FFN dimension from another architecture without recalculating parameters.

---

## 102. Weight tying

Possible choices:

- tie token embedding and output head;
- keep them separate.

Measure:

- parameter savings;
- validation loss;
- rare-token behavior;
- checkpoint conversion complexity.

Document the choice.

---

## 103. Bias policy

Some modern architectures remove many linear biases.
If experimenting:

- count parameter difference;
- compare stability;
- compare loss;
- verify conversion tooling.

Bias removal is an architectural choice, not a universal improvement.

---

## 104. Dropout policy

Large pretraining runs often use low or zero dropout, but Project XYZ should test its own regime.

Consider:

- dataset size;
- model size;
- overfitting evidence;
- continued training;
- small supervised stages.

Record all dropout probabilities.

---

## 105. Expert-model research

Do not move to mixture-of-experts until dense training is stable.

Expert models add:

- router parameters;
- routing instability;
- load balancing;
- expert parallelism;
- communication overhead;
- more complicated checkpointing;
- active versus total parameter accounting.

Build dense baselines first.

---

## 106. Expert routing metrics

If experts are introduced, log:

- tokens per expert;
- router entropy;
- expert load variance;
- dropped-token rate;
- auxiliary routing loss;
- active experts per token;
- communication time;
- per-expert gradient norms.

Expert collapse must be detectable.

---

## 107. Long-context research

Long-context ability is not proven by increasing a config number.

Evaluate:

- passkey retrieval;
- multi-document QA;
- position-dependent retrieval;
- long code context;
- long-form consistency;
- distractor resistance;
- performance versus token position.

---

## 108. Long-context data

Use documents that actually require long-range dependencies.
Simply concatenating unrelated short documents creates long sequences, not necessarily long-context learning.

Record document-length distribution.

---

## 109. Position-wise evaluation

Place the answer-bearing information at:

- beginning;
- 25%;
- middle;
- 75%;
- end.

Measure retrieval accuracy by position.
This detects “lost in the middle” behavior.

---

## 110. Scaling laws

Scaling-law experiments require controlled runs.

Vary:

- model parameters;
- training tokens;
- compute.

Keep data quality and tokenizer stable.
Fit curves only after collecting enough points.
Do not infer a law from two runs.

---

## 111. Tokens per parameter

Track total training tokens divided by parameter count.
This ratio is descriptive, not a magic optimum.

Use it to compare runs and plan budgets.

---

## 112. Compute accounting

Record:

- GPU count;
- GPU model;
- wall time;
- average utilization;
- approximate power;
- tokens processed;
- optimizer steps;
- failed-run time;
- checkpoint/eval overhead.

Later, estimate cost per useful improvement.

---

## 113. Laptop thermal policy

For long local runs:

- monitor GPU temperature;
- monitor CPU temperature;
- monitor clock throttling;
- keep vents unobstructed;
- verify fan behavior;
- avoid training if a fan is mechanically failing;
- use stable power settings;
- inspect charger temperature;
- pause hardware experiments if abnormal vibration or grinding occurs.

Hardware health comes before a training run.

---

## 114. Storage planning

Estimate before training.

Storage categories:

- raw data;
- processed token shards;
- checkpoints;
- optimizer states;
- evaluation outputs;
- activation dumps;
- logs;
- exported models.

Activation dumps can become larger than checkpoints.

---

## 115. Checkpoint size planning

Approximate checkpoint size depends on:

- parameter dtype;
- optimizer state;
- scaler state;
- sharding strategy;
- compression;
- whether temporary and latest copies coexist.

Ensure free disk space exceeds several checkpoints plus working headroom.

---

## 116. Data storage layout

Recommended:

```text
data/
├─ raw/
├─ manifests/
├─ project_xyz/
│  ├─ train.bin
│  ├─ train.bin.meta.json
│  ├─ valid.bin
│  └─ valid.bin.meta.json
└─ external_cache/
```

Do not commit huge datasets into git.

---

## 117. Checkpoint storage layout

Recommended:

```text
checkpoints/
├─ pretrain/
├─ continuation/
├─ posttrain/
├─ best/
└─ archive/
```

Store a metadata file beside important checkpoints.

---

## 118. Artifact hashes

For important releases compute SHA-256 for:

- tokenizer;
- config;
- data manifests;
- checkpoint;
- evaluator output;
- exported model shards.

Hashes make accidental changes detectable.

---

## 119. Distributed transition gate

Do not move to distributed training until local correctness is established.

Gate checklist:

- architecture spec frozen for experiment;
- parameter count verified;
- tokenizer frozen;
- data manifest frozen;
- local overfit test passes;
- local tiny-corpus test passes;
- local checkpoint resume passes;
- local evaluator passes;
- distributed backend supports architecture exactly;
- distributed forward parity checked;
- distributed short-run loss curve checked.

---

## 120. Optional backend bootstrap

Run:

```bash
python project_xyz_stack/bootstrap_project_xyz.py
```

This clones optional research backends under:

```text
project_xyz_stack/_upstream/project_xyz_backends/
```

A local lock file records upstream commits.
The `_upstream` directory is ignored by git.

---

## 121. Why upstream code stays upstream

Do not vendor thousands of unrelated backend files without a reason.
Keeping upstreams separate improves:

- attribution;
- updates;
- security review;
- diff clarity;
- repository size;
- reproducibility through pinned commits.

Project XYZ owns its model and experiment definitions.
The backend is replaceable infrastructure.

---

## 122. DDP

Use DDP when each GPU can hold a complete model copy.

Benefits:

- simple;
- mature;
- good scaling for fitting models.

Costs:

- full parameters on every GPU;
- full optimizer state on every GPU;
- gradient synchronization.

The local Project XYZ trainer supports `torchrun` DDP.

---

## 123. DDP launch example

```bash
torchrun --standalone --nproc_per_node=2 \
  project_xyz_stack/train.py \
  --config project_xyz_stack/config/project_xyz.json
```

Before multi-GPU use:

- test one GPU;
- ensure NCCL works;
- verify world size;
- verify effective batch;
- verify only rank zero writes checkpoints.

---

## 124. FSDP

Use FSDP when full model/optimizer replication is too expensive.

Potentially shards:

- parameters;
- gradients;
- optimizer states.

Validate:

- wrapping policy;
- state-dict format;
- resume behavior;
- mixed precision;
- gradient clipping semantics;
- checkpoint conversion.

---

## 125. Tensor/model parallelism

Use when individual layers do not fit efficiently on one GPU or when throughput justifies it.

Costs include communication inside every layer.
Require fast interconnects for good scaling.

Do not add model parallelism to a tiny model merely because it exists.

---

## 126. Context parallelism

Useful for very long sequences.
It partitions sequence/context work across devices.

Validate:

- attention semantics;
- positional encoding;
- sequence divisibility requirements;
- communication cost;
- checkpoint compatibility.

---

## 127. Expert parallelism

If Project XYZ adopts experts, expert parallelism may distribute experts across devices.

Monitor all-to-all communication.
A theoretically sparse model can still be communication-bound.

---

## 128. Interconnect awareness

Scaling depends on:

- PCIe;
- NVLink;
- InfiniBand;
- Ethernet;
- topology;
- NCCL configuration.

Benchmark communication separately from model compute.

---

## 129. Cluster environment record

For cluster runs record:

- node count;
- GPUs per node;
- GPU type;
- GPU memory;
- CPU model;
- RAM per node;
- interconnect;
- storage type;
- scheduler;
- container image;
- driver version;
- CUDA version;
- NCCL version.

---

## 130. Distributed failure modes

Common failures:

- one rank OOMs;
- NCCL timeout;
- data workers hang;
- rank zero exits early;
- checkpoint shard missing;
- filesystem metadata bottleneck;
- mismatched environment across nodes;
- different code revision across nodes;
- network interface misconfiguration.

Build failure recovery before expensive runs.

---

## 131. Distributed smoke test

Use:

- very small step count;
- frequent logging;
- one checkpoint;
- one evaluation;
- full stop/resume cycle.

Pass only if every rank exits cleanly and the checkpoint reloads.

---

## 132. Data-loader scaling

At high GPU throughput, the input pipeline can become the bottleneck.

Measure:

- tokenizer time;
- storage read time;
- CPU parsing time;
- packing time;
- H2D transfer time;
- GPU idle time waiting for data.

Fix the bottleneck you actually measure.

---

## 133. Sequence packing efficiency

Metric:

```text
real_tokens / capacity_tokens
```

Track padding waste.
High padding waste means compute is spent on non-data.

---

## 134. Asynchronous evaluation

For large runs, evaluation can happen separately.

Requirements:

- immutable checkpoint identifier;
- evaluator revision;
- benchmark revision;
- result file tied to checkpoint;
- retry logic;
- no accidental evaluation of a partially written checkpoint.

---

## 135. Training dashboard

A future Project XYZ dashboard should show:

- step;
- tokens;
- training loss;
- validation loss;
- LR;
- gradient norm;
- throughput;
- GPU utilization;
- VRAM;
- temperature;
- checkpoint events;
- evaluation scores;
- ETA based on measured throughput.

Do not let dashboards replace raw logs.

---

## 136. Experiment database

As experiments grow, store metadata in a structured database or table.

Fields:

- run ID;
- parent run;
- git commit;
- model config;
- data config;
- optimizer config;
- hardware;
- start/end time;
- status;
- best checkpoint;
- metrics;
- notes;
- failure reason.

---

## 137. Parent-child lineage

Every continuation or post-training run should identify its parent checkpoint.

Example:

```text
random_init
└─ pretrain_v1
   ├─ continue_math_v1
   ├─ continue_code_v1
   └─ sft_v1
      └─ preference_v1
```

Never lose lineage.

---

## 138. Continued pretraining

Use continued pretraining for domain adaptation when next-token data is available.

Examples:

- code;
- math;
- science;
- Nepal-specific text;
- legal/medical domains where appropriate data and evaluation exist.

Use lower LR than initial training as a starting hypothesis, then test.

---

## 139. SFT

Supervised fine-tuning teaches response formats and instruction behavior.

SFT dataset records should include:

- source;
- license;
- prompt;
- response;
- system context if used;
- quality score;
- domain;
- length;
- dedupe ID.

Do not mix low-quality synthetic responses blindly.

---

## 140. Assistant-only loss

For chat SFT, consider masking prompt tokens so loss applies primarily to the assistant response.

Verify masking visually on token IDs.
A one-token offset bug can invalidate the entire run.

---

## 141. Chat template

A chat template is part of post-training behavior.
Version:

- role tokens;
- separators;
- BOS/EOS use;
- system-message format;
- tool-call format;
- generation prompt suffix.

Training and inference must use compatible templates.

---

## 142. Synthetic data

Synthetic data can help but creates risks:

- model-style homogenization;
- factual errors;
- hidden duplication;
- reward hacking;
- benchmark leakage;
- low diversity.

Use generation filters and human spot checks.

---

## 143. Synthetic-data provenance

Record:

- generator model;
- generator version;
- prompt template;
- temperature;
- decoding parameters;
- generation date;
- filtering model;
- quality threshold;
- dedupe process.

Synthetic data without provenance is hard to debug.

---

## 144. Reasoning data

Separate:

- final-answer supervision;
- worked-solution supervision;
- verification data;
- process labels;
- tool traces.

Evaluate correctness independently of verbosity.

---

## 145. Tool-use data

If Project XYZ learns tools, define a machine-readable schema.

Record:

- tool name;
- arguments;
- tool output;
- assistant decision;
- success/failure;
- environment state;
- final response.

Train on valid tool calls, not just natural-language descriptions of tools.

---

## 146. Preference data

Preference pairs should track:

- prompt;
- chosen response;
- rejected response;
- reason for preference;
- annotator or judge source;
- confidence;
- domain;
- safety label if relevant.

Watch for judge-model bias.

---

## 147. Preference optimization

Before using a preference objective:

- establish SFT baseline;
- validate pair quality;
- measure reward accuracy;
- inspect KL drift;
- evaluate helpfulness and regressions;
- retain parent checkpoint.

Never overwrite the SFT base.

---

## 148. Reinforcement-learning stages

Use RL only when there is a meaningful reward signal.

Good candidates can include:

- verifiable math answers;
- code tests;
- tool-task success;
- environment rewards.

Weak subjective rewards require extra care.

---

## 149. Verifiable rewards

Prefer rewards that can be checked automatically when possible.

Examples:

- unit tests pass;
- exact numeric answer;
- theorem checker accepts;
- task environment reaches goal;
- JSON schema validates;
- tool execution succeeds.

Even verifiable rewards can be gamed.
Inspect trajectories.

---

## 150. Reward hacking

Look for:

- exploiting tests;
- hard-coded answers;
- malformed outputs that fool parsers;
- excessive retries;
- environment shortcuts;
- reward-model quirks.

Add adversarial evaluations before trusting reward improvements.

---

## 151. Post-training evaluation

Compare against the base model on:

- validation perplexity;
- instruction following;
- factual accuracy;
- reasoning;
- code;
- long context;
- style;
- refusal/robustness if relevant;
- free generation.

Post-training can improve task behavior while reducing base-model breadth.

---

## 152. Model export

Export must include:

- weights;
- architecture config;
- tokenizer;
- special tokens;
- chat template if applicable;
- generation defaults only as recommendations;
- license/usage metadata;
- checkpoint lineage.

Test exported model outputs against the source checkpoint.

---

## 153. Safetensors export

If exporting to safetensors:

- verify tensor count;
- verify tensor shapes;
- verify dtypes;
- verify tied-weight handling;
- verify shard index;
- verify reload.

Never delete source checkpoint before reload validation.

---

## 154. Quantization

Quantization is for inference unless a specific training method says otherwise.

Test:

- FP16/BF16 baseline;
- 8-bit;
- 4-bit;
- other formats if needed.

Measure:

- memory;
- tokens/sec;
- benchmark delta;
- generation changes.

---

## 155. Quantization calibration

For methods requiring calibration:

- use representative text;
- include target domains;
- keep calibration data separate from benchmark evaluation;
- record calibration set hash.

---

## 156. Inference benchmark

Record:

- hardware;
- model dtype;
- quantization;
- context length;
- prompt tokens;
- generated tokens;
- batch size;
- time to first token;
- decode tokens/sec;
- peak VRAM.

---

## 157. Sampling settings

Generation quality depends on decoding.
Record:

- temperature;
- top-p;
- top-k;
- repetition penalty if used;
- max tokens;
- stop sequences;
- seed.

Canonical evaluations should minimize sampling noise.

---

## 158. Repetition diagnostics

Measure:

- repeated n-grams;
- loop length;
- repeated sentences;
- EOS failure;
- degeneration by output length.

Repetition can indicate training, decoding, or data problems.

---

## 159. Factual diagnostics

Separate:

- memorized world facts;
- context-provided facts;
- entity binding;
- relation direction;
- uncertainty behavior.

A model can succeed at one and fail at another.

---

## 160. Arithmetic diagnostics

Test by operation and number range.

Categories:

- addition;
- subtraction;
- multiplication;
- division;
- negatives;
- decimals;
- percentages;
- multi-step arithmetic.

Use exact answer checking where possible.

---

## 161. Code diagnostics

Evaluate:

- syntax completion;
- function generation;
- bug fixing;
- unit-test pass rate;
- repository-level edits later;
- explanation separately from executable correctness.

Code that sounds right is not enough.
Run it.

---

## 162. Long-context diagnostics

Record accuracy versus context length.
Example lengths:

```text
512
1,024
2,048
4,096
8,192
16,384
```

Only test lengths supported by the architecture and runtime.

---

## 163. Robustness tests

Possible categories:

- typos;
- unusual capitalization;
- irrelevant distractors;
- contradictory context;
- reordered facts;
- missing information;
- adversarial formatting;
- long whitespace;
- Unicode confusables.

---

## 164. Calibration

A model should not always sound certain.
Where possible measure confidence/calibration using token probabilities or task-specific probability outputs.

Compare confidence against correctness.

---

## 165. Error taxonomy

Label failures.

Suggested labels:

- retrieval failure;
- binding error;
- direction reversal;
- arithmetic error;
- instruction omission;
- hallucination;
- repetition;
- truncation;
- formatting error;
- refusal error;
- tool-selection error;
- context-position error.

Error categories guide data decisions better than one aggregate score.

---

## 166. Evaluation report template

Every major checkpoint report should include:

```text
checkpoint
parent
step
tokens consumed
validation loss
perplexity
benchmark version
aggregate score
per-category scores
free-generation notes
known regressions
hardware
code commit
```

---

## 167. Experiment hypothesis template

Write before training:

```text
Hypothesis:
Change:
Control:
Expected mechanism:
Primary metric:
Secondary metrics:
Regression metrics:
Token budget:
Stop condition:
Decision rule:
```

This reduces hindsight bias.

---

## 168. Experiment conclusion template

Write after evaluation:

```text
Result:
Primary metric delta:
Regression delta:
Best checkpoint:
Unexpected behavior:
Likely explanation:
Confidence:
Next experiment:
Artifacts:
```

---

## 169. One-variable rule

When possible, change one main variable per experiment.

If architecture, data, LR, batch, and tokenizer all change together, a win teaches very little.

Large changes can still be tested, but label them as system changes rather than clean ablations.

---

## 170. Statistical caution

Small evaluations are noisy.
Small models are seed-sensitive.
Sampled decoding is noisy.

Use:

- larger test sets;
- multiple seeds;
- confidence intervals where appropriate;
- paired comparisons on identical examples.

---

## 171. Human evaluation

When automatic metrics miss quality, use structured human review.

Define rubrics such as:

- correctness;
- relevance;
- coherence;
- completeness;
- concision;
- style;
- factuality.

Randomize model order to reduce bias.

---

## 172. Blind evaluation

For serious comparisons, hide model/checkpoint identity from evaluators.
This reduces expectation bias.

---

## 173. Documentation rule

If a result matters, document it while the context is fresh.

Minimum:

- what changed;
- why;
- exact command;
- exact config;
- observed result;
- artifact paths;
- interpretation;
- next step.

---

## 174. Training-history rule

`TRAINING_HISTORY.md` should describe only verified runs.
Do not mark a planned script as completed training.

Use statuses:

```text
PLANNED
RUNNING
COMPLETED_UNEVALUATED
EVALUATED
REJECTED
OFFICIAL_BASELINE
```

---

## 175. Benchmark-history rule

`BENCHMARK_HISTORY.md` should include evaluator revision and checkpoint identity.
If an old score used a different protocol, label it clearly.

---

## 176. Root README versus stack README

Root `README.md` explains Project XYZ history and current research state.
This `project_xyz_stack/README.md` explains how to execute the training system.

Keep both.
Do not duplicate every historical table here.
Link instead.

---

## 177. Code style

Training code should prioritize:

- explicit behavior;
- readable state transitions;
- clear config names;
- deterministic error messages;
- minimal hidden globals;
- testability.

Do not optimize readability away for tiny speed gains.

---

## 178. Configuration validation

Future improvement: add strict schema validation.
Validate:

- positive step counts;
- positive sequence length;
- valid precision;
- nonnegative warmup;
- warmup <= total steps;
- checkpoint cadence > 0;
- eval cadence > 0;
- valid model paths;
- valid data paths;
- batch sizes > 0;
- accumulation > 0.

Fail before allocating the GPU.

---

## 179. Unit tests

Add tests for:

- model adapter;
- dataset indexing;
- dtype detection;
- LR schedule;
- checkpoint save/load;
- RNG restore;
- pruning;
- config validation;
- evaluator math;
- manifest writing.

---

## 180. Integration tests

Create a tiny toy model and tiny binary dataset.
Run:

- 5 training steps;
- checkpoint;
- resume;
- evaluation.

This should run in CI on CPU after trainer CPU-test support is added.

---

## 181. GPU integration test

On a CUDA machine run:

- validator;
- 20-step smoke test;
- BF16 or FP16 path;
- checkpoint reload;
- evaluation.

Record GPU details.

---

## 182. Numerical tests

Check LR schedule values at:

- step 0;
- half warmup;
- end warmup;
- mid training;
- final step.

Check exact expected ranges.

---

## 183. Checkpoint compatibility tests

When checkpoint format changes:

- bump format version;
- write migration code if old checkpoints matter;
- keep read tests for official baseline formats;
- never silently reinterpret fields.

---

## 184. Data-format versioning

The prepared token-stream manifest has a format identifier.
If binary semantics change, bump the identifier.

Changes requiring a bump can include:

- different document separators;
- segment metadata;
- compression;
- shard index semantics;
- dtype interpretation.

---

## 185. Failure: CUDA unavailable

Symptom:

```text
CUDA is required
```

Check:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

If build ends in `+cpu`, install the correct CUDA-enabled PyTorch build.

---

## 186. Failure: out of memory

Try in this order:

1. reduce micro batch;
2. increase gradient accumulation to preserve effective batch;
3. reduce sequence length for debugging;
4. disable unnecessary activation capture;
5. enable activation checkpointing if supported;
6. use more memory-efficient attention;
7. use sharding on multiple GPUs;
8. redesign model size if hardware target is fixed.

Do not immediately use CPU offload for every case; it can destroy throughput.

---

## 187. Failure: NaN loss

Investigate:

- LR too high;
- invalid data tokens;
- logits overflow;
- FP16 overflow;
- bad initialization;
- unstable normalization;
- broken attention mask;
- corrupted checkpoint;
- optimizer-state mismatch.

Reproduce on a tiny fixed batch.

---

## 188. Failure: loss does not decrease

Check:

- targets shifted correctly;
- parameters require gradients;
- optimizer sees parameters;
- backward executes;
- optimizer step executes;
- LR nonzero;
- gradient norm nonzero;
- data not random garbage;
- model output shape correct.

Run overfit-a-batch test.

---

## 189. Failure: loss drops unrealistically fast

Check for:

- repeated tiny dataset;
- target leakage;
- same tokens as inputs without correct shift;
- validation leakage;
- accidentally training on validation;
- data pointer not advancing;
- duplicated documents.

---

## 190. Failure: resume changes LR

Inspect:

- scheduler state loaded;
- optimizer state loaded;
- global step restored;
- scheduler stepped in the same order;
- config unchanged.

Resume bugs can waste long runs.

---

## 191. Failure: checkpoint loads but quality collapses

Check:

- architecture config identical;
- tokenizer identical;
- model state loaded strictly;
- normalization parameters present;
- tied weights restored;
- eval mode enabled;
- chat template identical;
- generation settings identical.

---

## 192. Failure: validation improves but benchmark falls

Possible causes:

- benchmark domain underrepresented;
- catastrophic forgetting;
- evaluation noise;
- decoding mismatch;
- benchmark contamination in earlier checkpoint;
- validation too narrow.

Inspect per-category errors.

---

## 193. Failure: benchmark improves but free generation worsens

Possible causes:

- overfitting benchmark-like formats;
- narrow fine-tuning data;
- chat-template mismatch;
- sampling sensitivity;
- catastrophic forgetting;
- short-answer bias.

Keep both metrics.

---

## 194. Failure: throughput decreases over time

Investigate:

- thermal throttling;
- memory fragmentation;
- background processes;
- storage cache behavior;
- checkpoint/eval intervals;
- dataloader workers;
- GPU power limit;
- Windows power mode.

---

## 195. Failure: one fan sounds abnormal

Do not treat mechanical fan noise as a software optimization problem.
High-RPM vibration or scraping from one fan can indicate debris, blade contact, or bearing wear.
Avoid extended maximum-fan training until hardware is inspected.

---

## 196. Windows notes

Project XYZ local training can run on Windows.
Watch for:

- path escaping;
- long paths;
- PowerShell quoting;
- multiprocessing behavior;
- CUDA/PyTorch wheel compatibility;
- file-locking behavior;
- antivirus scanning large checkpoint files.

Use raw strings for Windows paths in Python.

---

## 197. WSL/Linux notes

Linux/WSL is generally easier for advanced distributed tooling.
Use it when a backend depends on:

- Linux CUDA extensions;
- NCCL features;
- shell scripts;
- Slurm-like workflows;
- container runtimes.

Do not move environments mid-run without recording changes.

---

## 198. Python version policy

Pin a supported Python version for long experiments.
Very new Python releases may lag in support from:

- PyTorch extensions;
- quantization libraries;
- attention kernels;
- distributed packages.

Prefer compatibility over novelty on training machines.

---

## 199. Requirements lock

`requirements.txt` gives minimum dependencies.
For a reproducible major run, also capture exact installed versions.

Example:

```bash
python -m pip freeze > runs/<run>/pip-freeze.txt
```

---

## 200. Environment snapshot

Save:

```bash
python --version
python -m pip freeze
nvidia-smi
```

Also save PyTorch/CUDA check output.

---

## 201. First 10-minute work session

Do exactly this:

1. `git pull`.
2. activate `.venv`.
3. verify CUDA.
4. install stack requirements.
5. run `validate_stack.py`.
6. open `project_xyz_stack/config/project_xyz.json`.
7. confirm current model paths.
8. confirm data paths.
9. decide whether you are testing existing model or new architecture.
10. do not start a long run yet.

---

## 202. First one-hour work session

1. validate tokenizer;
2. prepare 1,000–10,000 documents;
3. inspect manifest;
4. run tiny-corpus training;
5. save one checkpoint;
6. stop process;
7. resume;
8. run evaluator;
9. record throughput;
10. record VRAM;
11. write any failure into notes.

---

## 203. First-day goal

By the end of day one, you should have:

- a working CUDA environment;
- a validated model bridge;
- a prepared data subset;
- a data manifest;
- a short successful training run;
- a reloadable checkpoint;
- an evaluation output;
- a written next experiment.

Do not define success as “the script ran.”

---

## 204. Architecture-development loop

Repeat:

1. write hypothesis;
2. change architecture;
3. count parameters;
4. validate forward/backward;
5. overfit tiny batch;
6. train fixed small token budget;
7. evaluate fixed validation set;
8. evaluate fixed task suite;
9. compare compute and memory;
10. keep or reject change.

---

## 205. Data-development loop

Repeat:

1. inspect source;
2. define filter;
3. measure removal rate;
4. inspect removed samples;
5. inspect retained samples;
6. tokenize subset;
7. measure token distribution;
8. run fixed model experiment;
9. compare validation/task metrics;
10. version filter.

---

## 206. Optimization-development loop

Repeat:

1. fix model;
2. fix data;
3. choose LR candidates;
4. choose warmup candidates;
5. run short sweeps;
6. compare stability;
7. compare early validation;
8. select range;
9. run longer confirmation;
10. document result.

---

## 207. Evaluation-development loop

Repeat:

1. define capability;
2. create held-out examples;
3. create deterministic scorer;
4. test scorer on known answers;
5. test false positives;
6. freeze version;
7. evaluate multiple checkpoints;
8. inspect errors;
9. expand difficult cases;
10. do not train directly on hidden test examples.

---

## 208. Weight-map-development loop

Repeat:

1. select two checkpoints;
2. verify same architecture;
3. compute tensor deltas;
4. rank layers;
5. select fixed prompt set;
6. capture activations;
7. identify correlated components;
8. ablate candidates;
9. test capability selectivity;
10. replicate with another training run.

---

## 209. Experiment priority order

Recommended order:

1. correctness;
2. data quality;
3. stable optimization;
4. architecture efficiency;
5. evaluation breadth;
6. scale;
7. post-training;
8. serving optimization.

Do not optimize serving before the model can learn.

---

## 210. Project XYZ milestone M0 — environment

Exit criteria:

- repository cloned;
- environment created;
- CUDA true;
- dependencies installed;
- validator passes.

Artifacts:

- environment notes;
- PyTorch version;
- CUDA version;
- GPU name.

---

## 211. Project XYZ milestone M1 — tokenizer

Exit criteria:

- tokenizer tests pass;
- special tokens verified;
- tokenizer hash recorded;
- efficiency measured on target domains.

Artifacts:

- tokenizer file;
- tokenizer report;
- hash.

---

## 212. Project XYZ milestone M2 — data subset

Exit criteria:

- source selected;
- raw samples inspected;
- subset prepared;
- manifest generated;
- random tokenized samples decoded correctly.

---

## 213. Project XYZ milestone M3 — tiny overfit

Exit criteria:

- fixed batch loss decreases strongly;
- no NaNs;
- gradients nonzero;
- checkpoint save/load works.

---

## 214. Project XYZ milestone M4 — smoke pretraining

Exit criteria:

- real data;
- 100–1000 steps;
- loss trend plausible;
- validation runs;
- restart works;
- throughput measured.

---

## 215. Project XYZ milestone M5 — baseline architecture

Exit criteria:

- architecture spec frozen;
- parameter count verified;
- short-run comparison beats or matches reference under declared metric;
- memory/throughput acceptable.

---

## 216. Project XYZ milestone M6 — baseline corpus

Exit criteria:

- corpus registry complete;
- filters versioned;
- dedupe measured;
- validation split frozen;
- mixture manifest frozen.

---

## 217. Project XYZ milestone M7 — first serious pretrain

Exit criteria:

- planned token budget consumed;
- intermediate checkpoints saved;
- run manifest complete;
- no unresolved training instability.

---

## 218. Project XYZ milestone M8 — checkpoint sweep

Exit criteria:

- all milestones evaluated;
- best checkpoint selected by declared rule;
- regressions documented;
- raw outputs archived.

---

## 219. Project XYZ milestone M9 — interpretability baseline

Exit criteria:

- weight deltas calculated;
- activation capture works;
- fixed prompt set versioned;
- at least one causal ablation experiment completed.

---

## 220. Project XYZ milestone M10 — distributed parity

Exit criteria:

- architecture registered in backend;
- parameter parity passes;
- forward parity passes;
- short distributed loss curve matches expectations;
- distributed checkpoint reload passes.

---

## 221. Project XYZ milestone M11 — scaled pretrain

Exit criteria:

- compute budget defined;
- storage budget defined;
- recovery tested;
- evaluation cadence defined;
- full run completed or intentionally stopped by rule.

---

## 222. Project XYZ milestone M12 — post-training

Exit criteria:

- base checkpoint frozen;
- SFT dataset versioned;
- chat template versioned;
- SFT evaluated against base;
- regressions documented.

---

## 223. Project XYZ milestone M13 — preference/reward stage

Exit criteria:

- reward signal validated;
- parent checkpoint retained;
- reward hacking tests run;
- benchmark and free-generation regressions checked.

---

## 224. Project XYZ milestone M14 — release candidate

Exit criteria:

- checkpoint hash;
- config;
- tokenizer;
- lineage;
- evaluation report;
- known limitations;
- inference test;
- export reload test.

---

## 225. Model-size ladder

Do not jump directly to the largest model you can imagine.

Suggested research ladder:

```text
~15M reference
~30M
~60M
~120M
~250M
~500M
~1B
larger only when infrastructure and evidence justify it
```

Actual sizes may differ.
The point is controlled scaling.

---

## 226. Matched-token comparisons

When comparing sizes, train on a matched token budget first.
This answers:

- which model learns faster per token?
- which has better validation loss at fixed data?
- which is better per unit compute?

Then run size-appropriate token budgets later.

---

## 227. Matched-compute comparisons

Another useful experiment holds approximate compute constant.
A smaller model can see more tokens under the same compute budget.
This can change which model is best.

---

## 228. Architecture naming

Use clear IDs.

Examples:

```text
project-xyz-dense-15m-v1
project-xyz-dense-60m-v1
project-xyz-dense-120m-gqa-v2
project-xyz-dense-250m-rope-v1
```

Name should encode meaningful design, not hype.

---

## 229. Model config files

As architectures multiply, create separate versioned configs instead of repeatedly editing one config in place.

Example:

```text
configs/models/project_xyz_15m.json
configs/models/project_xyz_60m.json
configs/models/project_xyz_120m.json
```

Future refactor recommended.

---

## 230. Training config files

Likewise separate training recipes:

```text
configs/training/smoke.json
configs/training/pretrain_local.json
configs/training/pretrain_cluster.json
configs/training/continue_code.json
configs/training/sft.json
```

Compose model/data/training configs later if needed.

---

## 231. Data config files

Future structure:

```text
configs/data/fineweb_small.json
configs/data/web_code_mix_v1.json
configs/data/math_mix_v1.json
```

Each config should resolve to immutable manifests for serious runs.

---

## 232. Config immutability

When a run starts, copy the resolved config into the run directory.
Later edits to repository configs must not rewrite history.

---

## 233. Run IDs

A run ID should be unique.
Include timestamp or random suffix if necessary.
Never reuse a run directory for a different experiment.

---

## 234. Resume versus new run

Resume only when continuing the same optimization trajectory.
Create a new run when changing:

- architecture;
- tokenizer;
- major data mixture;
- optimizer type;
- intended experiment hypothesis.

Continued pretraining from a checkpoint is a child run, not a silent resume.

---

## 235. Learning-rate restart

If continued pretraining creates a new run, decide explicitly whether to:

- continue old scheduler;
- restart warmup;
- use a new smaller LR;
- use a short re-warmup.

Document the choice.

---

## 236. Optimizer reset experiments

Sometimes continued training resets optimizer state.
Sometimes it should preserve it.
Test both when important.

Resetting changes the trajectory even from identical weights.

---

## 237. Dataset-order resume

Perfect resume ideally restores the data cursor/order.
The current lightweight loader does not yet implement a fully serialized dataloader cursor.

Treat exact data-order restoration as a future improvement.
Record this limitation.

---

## 238. Determinism limitations

Even with RNG state restored, GPU kernels can be nondeterministic.
Distributed scheduling can also change order.

Goal:

- scientifically reproducible trends;
- clearly recorded sources of nondeterminism.

Do not promise bitwise identity unless tested.

---

## 239. Data sharding

Large corpora should be sharded.
Recommended shard metadata:

- shard name;
- document count;
- token count;
- SHA-256;
- source mixture;
- min/max length;
- creation revision.

---

## 240. Shard validation

For every shard:

- file exists;
- hash matches;
- dtype correct;
- token count plausible;
- first/last samples decode;
- no truncated header if format has headers;
- manifest points to correct tokenizer.

---

## 241. Corpus statistics

Generate reports for:

- document length histogram;
- token length histogram;
- language distribution;
- source distribution;
- domain distribution;
- duplicate rate;
- quality-score distribution;
- code-language distribution;
- year distribution if metadata exists.

---

## 242. Data audit samples

For each quality bucket, save examples.
For each source, save examples.
For each rejection reason, save examples.

This makes filter behavior inspectable.

---

## 243. Sensitive-data handling

If corpora may contain personal data, define filtering and governance appropriate to the source and intended use.
Record what was done.
Do not assume public availability means every use is appropriate.

---

## 244. Licensing record

For each dataset record:

- dataset license;
- source terms;
- redistribution limits;
- attribution requirements;
- commercial-use restrictions if any.

Consult qualified legal guidance for release decisions when necessary.

---

## 245. Data version IDs

Example:

```text
project_xyz_corpus_v0.1
project_xyz_corpus_v0.2
project_xyz_corpus_v1.0
```

A version bump should correspond to documented changes.

---

## 246. Corpus changelog

Record:

- added sources;
- removed sources;
- filter changes;
- dedupe changes;
- mixture changes;
- token-count changes;
- benchmark decontamination changes.

---

## 247. Token accounting

Track:

- raw tokens available;
- tokens after filtering;
- unique tokens after dedupe;
- tokens sampled by source;
- tokens actually consumed by run.

Do not confuse corpus size with consumed token budget.

---

## 248. Epoch interpretation

For web-scale mixtures, “epoch” can be less useful than total tokens.
Still calculate approximate source reuse.
Repeated exposure can matter.

---

## 249. Source oversampling

If a small high-value source is oversampled, record the effective repetition count.
Oversampling can improve a domain and increase memorization risk.

---

## 250. Sampling temperature for data mixtures

Some multi-domain pipelines use temperature-based sampling.
If used, record formula and temperature.
Do not describe mixture only with nominal dataset sizes.

---

## 251. Curriculum by quality

Possible experiment:

- begin with broader corpus;
- finish with higher-quality corpus;

Or the reverse.
Evaluate rather than assume.

---

## 252. Annealed data mixtures

Mixture weights can change over training.
If used, make schedule explicit.

Example:

```text
steps 0–30%: broad web heavy
steps 30–80%: balanced
steps 80–100%: high-quality/code/math emphasis
```

This is an experiment, not a default truth.

---

## 253. Document boundaries

EOS between documents is important when packing.
Test whether the model learns to stop or transition appropriately.

If attention crosses packed-document boundaries, understand the implications.

---

## 254. BOS policy

Decide whether each sequence/document begins with BOS.
Keep training and inference consistent.

---

## 255. Padding policy

The local packed stream minimizes explicit padding.
For batched variable-length tasks later, define pad token and loss masking.

---

## 256. Loss masking

Use loss masks when only part of a sequence should contribute.
Examples:

- assistant-only SFT;
- tool output excluded;
- padded tokens excluded.

Unit-test mask alignment.

---

## 257. Label shift

Causal LM target should predict the next token.
Verify with a tiny known sequence.

Example:

```text
input:  [A, B, C]
target: [B, C, D]
```

An off-by-one error can still produce a decreasing loss while training the wrong task.

---

## 258. Attention mask test

For causal attention, future tokens must not affect earlier logits.
Test by changing a future token and comparing earlier outputs.

---

## 259. Context-boundary test

Test sequence at maximum configured length.
Test one token beyond maximum.
Expected behavior should be explicit.

---

## 260. Special-token test

Round-trip encode/decode:

- BOS;
- EOS;
- PAD;
- unknown/fallback behavior.

Confirm IDs match model config.

---

## 261. Initialization reproducibility test

Instantiate model twice with same seed.
Compare selected parameter tensors.
They should match when deterministic initialization path is used.

---

## 262. Optimizer-state test

After one step:

- optimizer state should exist;
- moving averages should change;
- parameter values should change;
- frozen parameters should not change.

---

## 263. Scheduler test

Print LR for first 20 steps and selected later steps.
Confirm warmup and decay visually before a long run.

---

## 264. Gradient-zeroing test

Ensure gradients do not unintentionally accumulate across optimizer steps beyond configured accumulation.

---

## 265. Accumulation equivalence test

Compare:

- larger micro batch, accumulation 1;
- smaller micro batch, greater accumulation;

Use same effective batch where possible.
Expect close, not necessarily bitwise-identical, updates.

---

## 266. Clipping test

Construct a batch with large gradients.
Verify norm is clipped to configured threshold.

---

## 267. AMP scaler test

On FP16 path verify:

- scaler enabled;
- scale changes if overflows occur;
- optimizer steps are skipped appropriately on overflow.

---

## 268. BF16 support test

Check:

```python
torch.cuda.is_bf16_supported()
```

If false, Project XYZ falls back to FP16 in the local trainer.

---

## 269. Data dtype test

Ensure manifest dtype matches binary file.
A uint16 file interpreted as uint32 corrupts token IDs.

---

## 270. Token-range test

Scan prepared data samples.
Verify:

```text
0 <= token_id < vocab_size
```

Any violation is a hard failure.

---

## 271. Dataset-length test

Verify computed block count is plausible from:

```text
(total_tokens - 1) // sequence_length
```

---

## 272. Shuffle test

When shuffle is enabled, inspect sample block indices across epochs.
When disabled, confirm deterministic order.

---

## 273. Validation-order test

Validation should not shuffle unless a specific protocol requires it.
Stable order makes debugging easier.

---

## 274. Evaluation batch count

Record number of validation batches.
A tiny evaluation can be noisy.
Increase for important comparisons.

---

## 275. Evaluation token count

Report evaluated token count along with loss.
This makes comparisons more interpretable.

---

## 276. Per-category benchmark counts

Always report denominator.

Bad:

```text
math 80%
```

Better:

```text
math 80% (80/100)
```

---

## 277. Exact-match normalization

If exact-match is used, define:

- case handling;
- whitespace;
- punctuation;
- numeric formatting;
- units;
- articles.

Version normalization logic.

---

## 278. Partial-credit policy

If partial credit exists, define it before evaluating.
Do not adjust rules after seeing outputs.

---

## 279. Generation truncation

A wrong answer caused by too-small `max_new_tokens` is an evaluation configuration bug.
Record truncation rate.

---

## 280. Stop sequences

Stop rules affect outputs.
Keep them fixed across checkpoint comparisons.

---

## 281. Prompt template version

A benchmark prompt template is part of the benchmark.
Version it.

---

## 282. Few-shot evaluation

If using few-shot examples:

- keep examples fixed;
- keep order fixed or controlled;
- ensure examples are not test items;
- report shot count.

---

## 283. Zero-shot evaluation

Zero-shot and few-shot are different protocols.
Do not mix scores in one chart without labels.

---

## 284. Chat versus base-model evaluation

A base model and an instruction model need different prompt formats.
Evaluate each according to its intended interface.

---

## 285. Base-model completion tests

Use natural prefixes.
Measure:

- syntax continuation;
- factual completion;
- coherence;
- repetition;
- EOS behavior.

---

## 286. Instruction-model tests

Measure:

- follows requested format;
- answers requested task;
- avoids irrelevant continuation;
- obeys multi-step constraints;
- uses tools correctly if applicable.

---

## 287. Model-card draft

Start model-card notes before release.
Include:

- architecture;
- tokenizer;
- training data summary;
- training compute summary;
- intended use;
- known limitations;
- benchmark table;
- lineage;
- license.

---

## 288. Release reproducibility package

For an open release, aim to publish:

- weights;
- model config;
- tokenizer;
- training config;
- data recipe/manifests where legally possible;
- evaluation code;
- evaluation outputs;
- checkpoint lineage;
- environment information;
- training notes.

---

## 289. Intermediate checkpoint release

Intermediate checkpoints are valuable for research.
If storage permits, publish meaningful milestones rather than only final weights.

They enable:

- capability-emergence studies;
- weight-delta studies;
- optimization research;
- reproducibility checks.

---

## 290. Training-log release

Useful logs include:

- step;
- token count;
- loss;
- LR;
- throughput;
- eval metrics;
- checkpoint timestamps.

Remove secrets or private infrastructure information before public release.

---

## 291. Data-recipe release

When raw data cannot be redistributed, release:

- source IDs;
- revisions;
- filtering logic;
- dedupe logic;
- mixture weights;
- tokenization logic;
- manifests without restricted content where allowed.

---

## 292. Security of credentials

Never commit:

- API keys;
- access tokens;
- cloud credentials;
- private SSH keys;
- service-account secrets.

Use environment variables or secret managers.

---

## 293. Secret scan

Before pushing major changes, scan git diff for credentials.
If a secret is committed, rotate it.
Deleting it from the latest file is not enough if it remains valid in history.

---

## 294. Dataset credentials

Some datasets require authentication.
Do not bake tokens into data scripts.
Use supported credential stores or environment variables.

---

## 295. Checkpoint privacy

A checkpoint can memorize training data.
If data contains sensitive information, releasing weights can create risks.
Data governance must happen before training, not only before release.

---

## 296. Reproducibility levels

Level 0:

- anecdotal run.

Level 1:

- config + checkpoint.

Level 2:

- config + code commit + tokenizer + data manifest + metrics.

Level 3:

- above + environment + logs + intermediate checkpoints + evaluation artifacts.

Project XYZ should target Level 3 for important experiments.

---

## 297. Research confidence labels

Use labels:

```text
OBSERVATION
HYPOTHESIS
REPLICATED
CAUSAL_EVIDENCE
UNRESOLVED
```

Do not describe a correlation as causal evidence.

---

## 298. Negative results

Record failed experiments.
They prevent repeating the same mistake.

Include:

- hypothesis;
- change;
- result;
- failure mode;
- artifacts;
- whether retry is worthwhile.

---

## 299. Stop rules

Define when to stop a run.
Possible rules:

- NaN/Inf persists;
- validation loss worsens for N evaluations;
- target metric peaks and regresses;
- hardware instability;
- compute budget reached;
- throughput unexpectedly collapses;
- data bug discovered.

Stopping a bad run early saves compute.

---

## 300. Run review

After every major run hold a review.
Questions:

1. Did the hypothesis survive?
2. Was the run technically valid?
3. Was evaluation valid?
4. What changed by category?
5. What is the best checkpoint?
6. What failed?
7. What should be repeated?
8. What should be abandoned?
9. What new question appeared?
10. What is the next cheapest decisive experiment?

---

## 301. Project XYZ master checklist — environment

- [ ] Git repository is current.
- [ ] Correct branch is checked out.
- [ ] Working tree status is understood.
- [ ] Virtual environment is active.
- [ ] Python version is supported.
- [ ] CUDA-enabled PyTorch is installed.
- [ ] `torch.cuda.is_available()` is true.
- [ ] Correct GPU name is detected.
- [ ] CUDA runtime is recorded.
- [ ] Driver is current enough.
- [ ] Disk space is checked.
- [ ] Host RAM is checked.
- [ ] GPU thermals are normal.
- [ ] Fan behavior is normal.
- [ ] Charger/power setup is stable.
- [ ] Background GPU apps are minimized.
- [ ] Requirements are installed.
- [ ] Package snapshot is saved for serious run.
- [ ] Project XYZ validator passes.
- [ ] Smoke-test command is ready.

---

## 302. Project XYZ master checklist — model

- [ ] Model ID assigned.
- [ ] Architecture spec written.
- [ ] Parameter target written.
- [ ] Actual parameter count measured.
- [ ] Trainable parameter count measured.
- [ ] Vocabulary size verified.
- [ ] Context length verified.
- [ ] Embedding dimension verified.
- [ ] Layer count verified.
- [ ] Head count verified.
- [ ] KV-head count verified if used.
- [ ] FFN dimension verified.
- [ ] Normalization type verified.
- [ ] Positional method verified.
- [ ] Activation function verified.
- [ ] Weight tying verified.
- [ ] Initialization verified.
- [ ] Forward shape test passes.
- [ ] Backward test passes.
- [ ] Tiny overfit test passes.

---

## 303. Project XYZ master checklist — tokenizer

- [ ] Tokenizer file selected.
- [ ] Tokenizer hash recorded.
- [ ] Vocabulary size matches model.
- [ ] BOS token ID verified.
- [ ] EOS token ID verified.
- [ ] PAD token ID verified if used.
- [ ] Unknown/fallback behavior understood.
- [ ] English samples tested.
- [ ] Numbers tested.
- [ ] Code tested.
- [ ] Unicode tested.
- [ ] Long words tested.
- [ ] Whitespace tested.
- [ ] Newlines tested.
- [ ] JSON tested.
- [ ] Markdown tested.
- [ ] Average token efficiency measured.
- [ ] Training/inference tokenizer path identical.
- [ ] Tokenizer revision frozen for run.
- [ ] Tokenizer metadata documented.

---

## 304. Project XYZ master checklist — data

- [ ] Data sources registered.
- [ ] Terms/license notes recorded.
- [ ] Source revisions recorded.
- [ ] Raw samples inspected.
- [ ] Language distribution measured.
- [ ] Document lengths measured.
- [ ] Quality filtering defined.
- [ ] Rejection samples inspected.
- [ ] Deduplication defined.
- [ ] Duplicate rate measured.
- [ ] Benchmark overlap checked.
- [ ] Train split frozen.
- [ ] Validation split frozen.
- [ ] Mixture weights recorded.
- [ ] Tokenizer revision recorded.
- [ ] Prepared data manifest exists.
- [ ] Token count plausible.
- [ ] Token range valid.
- [ ] Random decoded samples look correct.
- [ ] Storage has sufficient headroom.

---

## 305. Project XYZ master checklist — optimization

- [ ] Optimizer selected.
- [ ] Learning rate selected from evidence.
- [ ] Betas recorded.
- [ ] Epsilon recorded.
- [ ] Weight decay recorded.
- [ ] Parameter decay groups defined if used.
- [ ] Warmup defined.
- [ ] Scheduler defined.
- [ ] Minimum LR ratio defined.
- [ ] Micro batch defined.
- [ ] Accumulation defined.
- [ ] Effective batch calculated.
- [ ] Tokens per step calculated.
- [ ] Gradient clipping defined.
- [ ] Precision selected.
- [ ] FP16 scaler behavior tested if needed.
- [ ] Compile mode tested if enabled.
- [ ] Initial gradient norm inspected.
- [ ] LR schedule inspected.
- [ ] Short sweep results documented.

---

## 306. Project XYZ master checklist — checkpointing

- [ ] Run directory unique.
- [ ] Checkpoint cadence defined.
- [ ] Retention policy defined.
- [ ] Model state saved.
- [ ] Optimizer state saved.
- [ ] Scheduler state saved.
- [ ] Scaler state saved when needed.
- [ ] RNG state saved.
- [ ] Config saved.
- [ ] Step saved.
- [ ] Latest checkpoint path understood.
- [ ] Best checkpoint path separate.
- [ ] Checkpoint reload tested.
- [ ] Resume step verified.
- [ ] Resume LR verified.
- [ ] Resume loss continuity inspected.
- [ ] Checkpoint hash generated for releases.
- [ ] Milestone checkpoints archived.
- [ ] Disk usage monitored.
- [ ] Pruning cannot delete official baseline.

---

## 307. Project XYZ master checklist — evaluation

- [ ] Validation dataset frozen.
- [ ] Evaluator revision recorded.
- [ ] Deterministic decoding defined.
- [ ] Sample count recorded.
- [ ] Token count recorded where relevant.
- [ ] Perplexity evaluated.
- [ ] Semantic benchmark evaluated.
- [ ] Factual benchmark evaluated.
- [ ] Instruction benchmark evaluated.
- [ ] Reasoning benchmark evaluated.
- [ ] Arithmetic evaluated if relevant.
- [ ] Code evaluated if relevant.
- [ ] Long context evaluated if relevant.
- [ ] Free-generation prompts evaluated.
- [ ] Per-category denominators reported.
- [ ] Regression gates checked.
- [ ] Multiple checkpoints compared.
- [ ] Best checkpoint selected by declared rule.
- [ ] Raw outputs saved.
- [ ] Evaluation report written.

---

## 308. Project XYZ master checklist — distributed scale

- [ ] Local baseline valid.
- [ ] Distributed backend revision pinned.
- [ ] Architecture registered exactly.
- [ ] Parameter count matches.
- [ ] Forward shapes match.
- [ ] Logit parity sampled.
- [ ] Data semantics match.
- [ ] Tokenizer matches.
- [ ] Effective batch recalculated.
- [ ] LR reconsidered for batch change.
- [ ] Precision validated.
- [ ] Communication backend works.
- [ ] Short multi-GPU run succeeds.
- [ ] Distributed checkpoint saves.
- [ ] Distributed checkpoint reloads.
- [ ] Resume works.
- [ ] Evaluation can read checkpoint.
- [ ] Throughput measured.
- [ ] Scaling efficiency measured.
- [ ] Failure recovery rehearsed.

---

## 309. Project XYZ master checklist — interpretability

- [ ] Fixed prompt set versioned.
- [ ] Checkpoint pair selected.
- [ ] Architecture identical across pair.
- [ ] Weight deltas computed.
- [ ] Layer norms compared.
- [ ] Candidate components ranked.
- [ ] Activations collected.
- [ ] Activation storage bounded.
- [ ] Per-layer statistics computed.
- [ ] Attention analysis run if relevant.
- [ ] MLP analysis run if relevant.
- [ ] Gradient analysis run if relevant.
- [ ] Correlation claims labeled as correlation.
- [ ] Ablation experiment designed.
- [ ] Ablation control created.
- [ ] Selective capability impact measured.
- [ ] Nonspecific damage measured.
- [ ] Result replicated if important.
- [ ] Negative result saved.
- [ ] Interpretation written cautiously.

---

## 310. Project XYZ master checklist — post-training

- [ ] Base checkpoint frozen.
- [ ] Parent hash recorded.
- [ ] SFT dataset versioned.
- [ ] SFT provenance recorded.
- [ ] Chat template versioned.
- [ ] Loss mask tested.
- [ ] SFT short run validated.
- [ ] SFT checkpoint sweep evaluated.
- [ ] Base regressions measured.
- [ ] Preference data quality measured if used.
- [ ] Reward signal validated if used.
- [ ] Reward hacking tests defined if used.
- [ ] Tool environments versioned if used.
- [ ] RL parent retained if used.
- [ ] Post-training lineage recorded.
- [ ] Export tested.
- [ ] Inference prompt format tested.
- [ ] Release metrics generated.
- [ ] Known limitations documented.
- [ ] Final checkpoint hash recorded.

---

## 311. Project XYZ master checklist — release

- [ ] Model name final.
- [ ] Architecture config included.
- [ ] Tokenizer included.
- [ ] Weights included.
- [ ] Shard index included if needed.
- [ ] Chat template included if needed.
- [ ] Training summary written.
- [ ] Data summary written.
- [ ] Compute summary written.
- [ ] Benchmark table written.
- [ ] Benchmark protocols linked.
- [ ] Known limitations written.
- [ ] Intended-use notes written.
- [ ] License reviewed.
- [ ] Attribution reviewed.
- [ ] Export reload tested.
- [ ] Generation smoke test passed.
- [ ] Hashes generated.
- [ ] Intermediate checkpoints considered for release.
- [ ] Reproducibility files included.

---

## 312. Experiment queue — environment and correctness

- [ ] E001: validate current 14.66M bridge on CPU forward only.
- [ ] E002: validate current 14.66M bridge on CUDA.
- [ ] E003: confirm BF16 support on RTX 4050.
- [ ] E004: benchmark BF16 versus FP16 memory.
- [ ] E005: benchmark BF16 versus FP16 throughput.
- [ ] E006: run fixed-batch overfit.
- [ ] E007: test checkpoint save.
- [ ] E008: test checkpoint reload.
- [ ] E009: test optimizer-state restore.
- [ ] E010: test scheduler-state restore.
- [ ] E011: test RNG restore.
- [ ] E012: test interrupted-run resume.
- [ ] E013: test final-step duplicate checkpoint behavior.
- [ ] E014: test pruning with keep=3.
- [ ] E015: verify latest checkpoint remains reloadable.
- [ ] E016: measure peak VRAM.
- [ ] E017: measure reserved VRAM.
- [ ] E018: measure dataloader overhead.
- [ ] E019: measure checkpoint-write time.
- [ ] E020: measure evaluation overhead.

---

## 313. Experiment queue — tokenizer

- [ ] T001: measure English tokens/word.
- [ ] T002: measure Python tokens/character.
- [ ] T003: measure JavaScript tokens/character.
- [ ] T004: inspect numeric tokenization 0–10000.
- [ ] T005: inspect decimal tokenization.
- [ ] T006: inspect date tokenization.
- [ ] T007: inspect URL tokenization.
- [ ] T008: inspect JSON tokenization.
- [ ] T009: inspect markdown tokenization.
- [ ] T010: inspect Unicode fallback.
- [ ] T011: inspect Nepali samples.
- [ ] T012: inspect emoji.
- [ ] T013: verify BOS behavior.
- [ ] T014: verify EOS behavior.
- [ ] T015: verify PAD behavior.
- [ ] T016: decode random encoded samples.
- [ ] T017: calculate tokenizer hash.
- [ ] T018: compare 16k versus 32k candidate tokenizer on same corpus.
- [ ] T019: compare code efficiency across vocab sizes.
- [ ] T020: decide tokenizer freeze for next lineage.

---

## 314. Experiment queue — data quality

- [ ] D001: inspect 100 random FineWeb documents.
- [ ] D002: inspect 100 shortest documents.
- [ ] D003: inspect 100 longest documents.
- [ ] D004: measure duplicate hashes.
- [ ] D005: measure repeated-line ratio.
- [ ] D006: measure language distribution.
- [ ] D007: estimate code fraction.
- [ ] D008: estimate math fraction.
- [ ] D009: identify boilerplate patterns.
- [ ] D010: create first quality filter.
- [ ] D011: inspect rejected examples.
- [ ] D012: inspect retained examples.
- [ ] D013: tokenize filtered subset.
- [ ] D014: compare token length distributions.
- [ ] D015: run fixed-token training on unfiltered subset.
- [ ] D016: run fixed-token training on filtered subset.
- [ ] D017: compare validation loss.
- [ ] D018: compare semantic benchmarks.
- [ ] D019: document filter decision.
- [ ] D020: version corpus recipe.

---

## 315. Experiment queue — architecture

- [ ] A001: current learned-position baseline.
- [ ] A002: rotary-position variant.
- [ ] A003: LayerNorm baseline repeat.
- [ ] A004: RMSNorm variant.
- [ ] A005: GELU baseline repeat.
- [ ] A006: SwiGLU variant.
- [ ] A007: tied embedding/output head.
- [ ] A008: untied control.
- [ ] A009: bias-free linear variant.
- [ ] A010: current bias control.
- [ ] A011: deeper/narrower matched-param model.
- [ ] A012: shallower/wider matched-param model.
- [ ] A013: fewer attention heads.
- [ ] A014: more attention heads.
- [ ] A015: GQA variant.
- [ ] A016: FFN ratio sweep.
- [ ] A017: initialization std sweep.
- [ ] A018: residual scaling variant.
- [ ] A019: compare quality per token.
- [ ] A020: select next architecture baseline.

---

## 316. Experiment queue — optimization

- [ ] O001: LR 1e-4 short run.
- [ ] O002: LR 2e-4 short run.
- [ ] O003: LR 3e-4 short run.
- [ ] O004: LR 5e-4 short run.
- [ ] O005: LR 8e-4 short run.
- [ ] O006: compare early loss slope.
- [ ] O007: compare validation after fixed tokens.
- [ ] O008: warmup 1%.
- [ ] O009: warmup 3%.
- [ ] O010: warmup 5%.
- [ ] O011: minimum LR ratio 0.0.
- [ ] O012: minimum LR ratio 0.1.
- [ ] O013: minimum LR ratio 0.2.
- [ ] O014: beta2 0.95.
- [ ] O015: beta2 0.98.
- [ ] O016: weight decay 0.05.
- [ ] O017: weight decay 0.1.
- [ ] O018: clip 0.5 versus 1.0.
- [ ] O019: compile mode benchmark.
- [ ] O020: select stable local recipe.

---

## 317. Experiment queue — scaling

- [ ] S001: 15M reference fixed-token run.
- [ ] S002: ~30M model fixed-token run.
- [ ] S003: ~60M model fixed-token run.
- [ ] S004: ~120M model smoke run.
- [ ] S005: compare memory curves.
- [ ] S006: compare throughput curves.
- [ ] S007: compare validation loss per token.
- [ ] S008: compare benchmark score per token.
- [ ] S009: estimate compute per model.
- [ ] S010: fit preliminary scaling curve.
- [ ] S011: choose next size.
- [ ] S012: test distributed backend with small model.
- [ ] S013: verify parameter parity distributed.
- [ ] S014: verify checkpoint conversion.
- [ ] S015: benchmark 2-GPU DDP if hardware available.
- [ ] S016: benchmark sharded backend if needed.
- [ ] S017: measure scaling efficiency.
- [ ] S018: calculate larger-run storage budget.
- [ ] S019: calculate larger-run token budget.
- [ ] S020: approve or reject scale-up.

---

## 318. Experiment queue — interpretability

- [ ] I001: compare random-init and first checkpoint weights.
- [ ] I002: compare early and mid checkpoints.
- [ ] I003: compare mid and best checkpoints.
- [ ] I004: rank tensors by relative change.
- [ ] I005: plot per-layer weight delta.
- [ ] I006: collect residual norms on factual prompts.
- [ ] I007: collect residual norms on code prompts.
- [ ] I008: collect residual norms on reasoning prompts.
- [ ] I009: compare prompt-class activation similarity.
- [ ] I010: identify candidate layers.
- [ ] I011: ablate one candidate attention head.
- [ ] I012: ablate one candidate MLP block.
- [ ] I013: measure targeted capability change.
- [ ] I014: measure unrelated capability change.
- [ ] I015: run activation patching pilot.
- [ ] I016: run gradient-similarity pilot.
- [ ] I017: repeat on second seed.
- [ ] I018: reject non-replicating localization claims.
- [ ] I019: document causal evidence separately from correlations.
- [ ] I020: publish first Project XYZ interpretability report.

---

## 319. Experiment queue — post-training

- [ ] P001: freeze selected base checkpoint.
- [ ] P002: create SFT schema.
- [ ] P003: collect small human-quality SFT set.
- [ ] P004: generate synthetic candidates.
- [ ] P005: filter synthetic data.
- [ ] P006: dedupe SFT data.
- [ ] P007: split SFT validation.
- [ ] P008: define chat template.
- [ ] P009: test assistant-only mask.
- [ ] P010: 100-step SFT smoke test.
- [ ] P011: SFT LR sweep.
- [ ] P012: checkpoint sweep.
- [ ] P013: compare base validation loss.
- [ ] P014: compare instruction score.
- [ ] P015: compare factual score.
- [ ] P016: compare free generation.
- [ ] P017: prepare preference pilot.
- [ ] P018: validate preference labels.
- [ ] P019: run small preference experiment.
- [ ] P020: decide whether advanced post-training is justified.

---

## 320. Run-note template

```markdown
# Run <ID>

## Purpose

## Hypothesis

## Parent checkpoint

## Git commit

## Model

## Tokenizer

## Data

## Training config

## Hardware

## Start time

## End time

## Tokens consumed

## Checkpoints

## Validation

## Benchmarks

## Free-generation notes

## Failures

## Interpretation

## Next action
```

---

## 321. Model-spec template

```markdown
# Project XYZ model specification

Model ID:
Version:
Architecture family:
Target parameters:
Actual parameters:
Vocabulary:
Context:
Model dimension:
Layers:
Attention heads:
KV heads:
Head dimension:
FFN dimension:
Activation:
Normalization:
Norm epsilon:
Position encoding:
Position settings:
Bias policy:
Weight tying:
Dropout:
Initialization:
Attention kernel:
Expert configuration:
Expected training precision:
Expected inference precision:
Minimum hardware:
Scale hardware:
Notes:
```

---

## 322. Data-manifest extension template

```json
{
  "project": "Project XYZ",
  "corpus_version": "",
  "created_at": "",
  "tokenizer_sha256": "",
  "sources": [],
  "filters": [],
  "dedupe": {},
  "decontamination": {},
  "mixture": [],
  "documents": 0,
  "tokens": 0,
  "shards": [],
  "notes": ""
}
```

---

## 323. Evaluation-report template

```markdown
# Project XYZ evaluation report

Checkpoint:
Checkpoint hash:
Parent:
Step:
Tokens consumed:
Model config:
Tokenizer hash:
Evaluator commit:
Benchmark revision:
Decoding protocol:
Validation loss:
Perplexity:
Aggregate score:
Per-category scores:
Regression gates:
Free-generation findings:
Known limitations:
Decision:
```

---

## 324. Interpretability-report template

```markdown
# Project XYZ interpretability experiment

Question:
Capability:
Checkpoint A:
Checkpoint B:
Metric delta:
Weight-delta method:
Activation method:
Candidate components:
Ablation method:
Control:
Targeted effect:
Nontarget effect:
Replicated:
Confidence label:
Limitations:
Next experiment:
```

---

## 325. Distributed-run template

```markdown
# Project XYZ distributed run

Run ID:
Backend:
Backend revision:
Nodes:
GPUs/node:
GPU model:
Interconnect:
Model ID:
Parameters:
Sequence length:
Micro batch/rank:
Accumulation:
Data-parallel size:
Model-parallel size:
Context-parallel size:
Effective global batch:
Tokens/step:
Precision:
Optimizer:
Learning rate:
Warmup:
Total steps:
Total token target:
Checkpoint cadence:
Evaluation cadence:
Storage path:
Resume test passed:
```

---

## 326. SFT-run template

```markdown
# Project XYZ SFT run

Parent checkpoint:
SFT data version:
Chat template version:
Loss mask:
Sequence length:
Batch:
LR:
Scheduler:
Steps:
Validation set:
Base regressions:
Instruction gains:
Best checkpoint:
Decision:
```

---

## 327. Release checklist detail — files

Expected release files may include:

```text
config.json
model*.safetensors
model.safetensors.index.json
tokenizer.json
tokenizer_config.json
special_tokens_map.json
generation_config.json
README.md
LICENSE
TRAINING.md
EVALUATION.md
```

Exact files depend on export format.

---

## 328. Release checklist detail — verification

Before release:

- load from clean environment;
- run one forward pass;
- run one deterministic generation;
- compare known benchmark sample;
- verify tokenizer special tokens;
- verify model parameter count;
- verify shard checksums;
- verify README commands.

---

## 329. README command testing

Every command in this document should eventually be exercised in CI or a documented environment.
Documentation that does not run becomes technical debt.

---

## 330. Project XYZ principles summary

Project XYZ should be:

- inspectable;
- reproducible;
- architecture-independent at the trainer layer;
- evidence-driven;
- checkpoint-rich;
- evaluation-heavy;
- honest about compute limits;
- careful about data provenance;
- willing to keep negative results;
- scalable without throwing away local research tooling.

---

# Appendix A — Detailed execution gates

The following gates turn the broad workflow into binary decisions.
Do not move to the next gate merely because you are impatient.

## Gate A001 — repository

- [ ] Repository path is correct.
- [ ] Remote points to `sajancodes/project-xyz`.
- [ ] `main` is current.
- [ ] Intended branch is known.
- [ ] No unexplained local modifications.
- [ ] Latest Project XYZ stack exists.
- [ ] Old stack naming is absent from active paths.
- [ ] Root documentation opens correctly.
- [ ] Stack documentation opens correctly.
- [ ] Git can create commits.

## Gate A002 — Python

- [ ] Python executable is expected one.
- [ ] Virtual environment path is expected one.
- [ ] `pip` belongs to same environment.
- [ ] Python version recorded.
- [ ] No accidental global-package dependency.
- [ ] Import `numpy` succeeds.
- [ ] Import `tokenizers` succeeds.
- [ ] Import `datasets` succeeds.
- [ ] Import `torch` succeeds.
- [ ] Environment snapshot saved.

## Gate A003 — CUDA

- [ ] NVIDIA driver visible.
- [ ] PyTorch CUDA build present.
- [ ] CUDA availability true.
- [ ] GPU device count plausible.
- [ ] GPU name correct.
- [ ] VRAM amount plausible.
- [ ] BF16 support checked.
- [ ] Simple CUDA tensor allocation succeeds.
- [ ] Simple CUDA matmul succeeds.
- [ ] CUDA memory releases after process exit.

## Gate A004 — model import

- [ ] Config file exists.
- [ ] Model file exists.
- [ ] Config class exists.
- [ ] Model class exists.
- [ ] Dynamic import succeeds.
- [ ] Config object constructs.
- [ ] Model object constructs.
- [ ] Object is `torch.nn.Module`.
- [ ] Parameters are present.
- [ ] Parameter count logged.

## Gate A005 — forward

- [ ] Random token IDs within vocab range.
- [ ] Batch dimension correct.
- [ ] Sequence dimension correct.
- [ ] Forward returns supported structure.
- [ ] Logits tensor exists.
- [ ] Vocabulary dimension correct.
- [ ] Loss exists or adapter computes it.
- [ ] Loss finite.
- [ ] No device mismatch.
- [ ] No dtype mismatch.

## Gate A006 — backward

- [ ] `loss.backward()` succeeds.
- [ ] Gradients exist.
- [ ] Gradient norms finite.
- [ ] No unexpected frozen layers.
- [ ] No NaN gradients.
- [ ] Optimizer accepts parameters.
- [ ] Optimizer step succeeds.
- [ ] Parameters change.
- [ ] Zero-grad clears expected buffers.
- [ ] Second step succeeds.

## Gate A007 — tokenizer file

- [ ] File exists.
- [ ] File parses.
- [ ] Vocab size matches model config.
- [ ] BOS ID known.
- [ ] EOS ID known.
- [ ] PAD ID known if used.
- [ ] Hash computed.
- [ ] Encode works.
- [ ] Decode works.
- [ ] Round trip inspected.

## Gate A008 — data source

- [ ] Source resolves.
- [ ] Split resolves.
- [ ] Text field exists.
- [ ] Text field contains strings.
- [ ] Empty documents handled.
- [ ] Source terms noted.
- [ ] Source revision noted if available.
- [ ] Sample quality inspected.
- [ ] Sample language inspected.
- [ ] Source approved for experiment.

## Gate A009 — data preparation

- [ ] Output directory exists.
- [ ] Tokenizer path correct.
- [ ] EOS policy correct.
- [ ] Max-doc setting intended.
- [ ] Progress output plausible.
- [ ] Binary file nonempty.
- [ ] Meta file exists.
- [ ] Meta tokenizer hash matches.
- [ ] Meta token count plausible.
- [ ] Meta document count plausible.

## Gate A010 — prepared-data validation

- [ ] Dtype matches manifest.
- [ ] First 1k token IDs valid.
- [ ] Random token windows valid.
- [ ] EOS appears at plausible frequency.
- [ ] Decoded windows look like source data.
- [ ] No systematic corruption.
- [ ] Training block count > 0.
- [ ] Validation block count > 0.
- [ ] Train and validation files differ.
- [ ] File hashes optionally recorded.

## Gate A011 — config

- [ ] Run name unique.
- [ ] Seed intentional.
- [ ] Model paths correct.
- [ ] Data paths correct.
- [ ] Sequence length supported.
- [ ] Micro batch fits.
- [ ] Accumulation intentional.
- [ ] Precision supported.
- [ ] LR intentional.
- [ ] Output directory unique.

## Gate A012 — scheduler

- [ ] Total steps > 0.
- [ ] Warmup >= 0.
- [ ] Warmup < total steps.
- [ ] Min LR ratio valid.
- [ ] Initial LR expected.
- [ ] Warmup slope expected.
- [ ] Peak LR expected.
- [ ] Mid-run LR expected.
- [ ] Final LR expected.
- [ ] Resume restores scheduler.

## Gate A013 — smoke run

- [ ] CUDA selected.
- [ ] Model parameters printed.
- [ ] World size correct.
- [ ] Precision printed.
- [ ] Step 1 completes.
- [ ] Loss finite.
- [ ] Throughput nonzero.
- [ ] VRAM plausible.
- [ ] No thermal issue.
- [ ] Planned short run completes.

## Gate A014 — smoke checkpoint

- [ ] Checkpoint file exists.
- [ ] Latest file exists.
- [ ] Global step correct.
- [ ] Model state present.
- [ ] Optimizer state present.
- [ ] Scheduler state present.
- [ ] Config present.
- [ ] RNG present.
- [ ] Reload succeeds.
- [ ] Evaluation can load it.

## Gate A015 — resume

- [ ] Process stopped cleanly.
- [ ] Restart sees latest checkpoint.
- [ ] Reported start step correct.
- [ ] LR matches pre-stop trajectory.
- [ ] Loss remains finite.
- [ ] Optimizer statistics preserved.
- [ ] New checkpoints use higher steps.
- [ ] Latest pointer advances.
- [ ] No overwrite of unrelated run.
- [ ] Resume notes saved.

## Gate A016 — validation

- [ ] Validation loader constructs.
- [ ] Validation does not shuffle.
- [ ] Eval mode entered.
- [ ] Gradients disabled.
- [ ] Requested batch count honored.
- [ ] Loss finite.
- [ ] Perplexity finite.
- [ ] Eval mode exits back to train where required.
- [ ] Distributed reduction works if used.
- [ ] Result logged.

## Gate A017 — benchmark

- [ ] Checkpoint identity fixed.
- [ ] Evaluator version fixed.
- [ ] Prompt format fixed.
- [ ] Decoding fixed.
- [ ] Test set frozen.
- [ ] Denominators known.
- [ ] Scoring verified.
- [ ] Raw outputs saved.
- [ ] Category metrics generated.
- [ ] Aggregate generated.

## Gate A018 — baseline decision

- [ ] Candidate compared to official baseline.
- [ ] Validation compared.
- [ ] Primary benchmark compared.
- [ ] Regression metrics compared.
- [ ] Free generation reviewed.
- [ ] Best step identified.
- [ ] Result replicated if delta is small.
- [ ] Checkpoint hash generated.
- [ ] Lineage documented.
- [ ] Decision recorded.

## Gate A019 — architecture experiment

- [ ] Hypothesis written.
- [ ] One main architecture change identified.
- [ ] Parameter count recalculated.
- [ ] Memory estimate recalculated.
- [ ] Forward test passes.
- [ ] Backward test passes.
- [ ] Overfit test passes.
- [ ] Fixed-token run completes.
- [ ] Metrics compared.
- [ ] Keep/reject decision recorded.

## Gate A020 — scale decision

- [ ] Current architecture stable.
- [ ] Data recipe stable.
- [ ] Optimizer stable.
- [ ] Evaluation stable.
- [ ] Compute estimate made.
- [ ] Storage estimate made.
- [ ] Expected value of scale-up stated.
- [ ] Distributed backend validated.
- [ ] Recovery plan tested.
- [ ] Budget approved.

---

# Appendix B — 100-day Project XYZ execution plan

Each day has one primary objective.
Move faster if tasks are already complete.
Do not skip validation gates merely to match the day number.

## Day 001

Objective: clean environment.
Action: update repository.
Action: create fresh virtual environment.
Action: install CUDA-enabled PyTorch.
Action: install Project XYZ requirements.
Validation: CUDA is true.
Validation: GPU name correct.
Artifact: environment snapshot.
Decision: proceed only if validator dependencies import.

## Day 002

Objective: validate current model bridge.
Action: run validator.
Action: record parameter count.
Action: inspect logits shape.
Action: inspect loss.
Validation: finite loss.
Artifact: validation output.
Decision: fix bridge before data work if it fails.

## Day 003

Objective: tokenizer audit.
Action: test English.
Action: test code.
Action: test numbers.
Action: test Unicode.
Validation: IDs in range.
Artifact: tokenizer audit notes.
Decision: freeze or redesign tokenizer.

## Day 004

Objective: prepare tiny dataset.
Action: stream 1k–10k documents.
Action: create `train.bin`.
Action: inspect manifest.
Validation: token count plausible.
Artifact: tiny prepared corpus.
Decision: proceed to decode inspection.

## Day 005

Objective: inspect prepared tokens.
Action: decode random windows.
Action: inspect EOS boundaries.
Action: inspect long windows.
Validation: no corruption.
Artifact: sample report.
Decision: fix preprocessing if corruption appears.

## Day 006

Objective: overfit one batch.
Action: freeze batch.
Action: run repeated updates.
Validation: loss falls strongly.
Artifact: overfit curve.
Decision: trainer core valid if successful.

## Day 007

Objective: first real smoke run.
Action: 100–1000 real-data steps.
Validation: loss decreases plausibly.
Artifact: checkpoint.
Decision: proceed to resume test.

## Day 008

Objective: resume test.
Action: stop run.
Action: restart from latest.
Validation: step and LR continue.
Artifact: resume log.
Decision: fix resume before longer runs.

## Day 009

Objective: validation baseline.
Action: prepare fixed validation data.
Action: run evaluator.
Validation: stable repeated result.
Artifact: baseline loss/perplexity.
Decision: freeze validation split.

## Day 010

Objective: benchmark baseline.
Action: run deterministic Project XYZ benchmark.
Validation: raw outputs saved.
Artifact: benchmark report.
Decision: define primary and regression metrics.

## Day 011

Objective: LR sweep setup.
Action: create candidate configs.
Validation: only LR differs.
Artifact: sweep table.
Decision: launch short runs.

## Day 012

Objective: LR sweep low range.
Action: run first candidates.
Validation: no instability.
Artifact: loss curves.
Decision: eliminate poor rates.

## Day 013

Objective: LR sweep high range.
Action: run remaining candidates.
Validation: compare fixed-token validation.
Artifact: completed LR sweep.
Decision: choose provisional LR.

## Day 014

Objective: warmup sweep.
Action: compare warmup fractions.
Validation: stable initial gradients.
Artifact: warmup comparison.
Decision: select warmup.

## Day 015

Objective: batch/accumulation study.
Action: compare micro-batch settings.
Validation: effective batch recorded.
Artifact: throughput/memory table.
Decision: choose local batch recipe.

## Day 016

Objective: tokenizer efficiency comparison.
Action: train or load candidate tokenizer if needed.
Validation: quality of segmentation inspected.
Artifact: tokenizer comparison.
Decision: keep frozen tokenizer or branch new lineage.

## Day 017

Objective: data-quality baseline.
Action: inspect web samples systematically.
Validation: rejection categories defined.
Artifact: quality taxonomy.
Decision: create filter prototype.

## Day 018

Objective: filter prototype.
Action: implement basic quality filters.
Validation: inspect removed samples.
Artifact: filter revision.
Decision: tune thresholds.

## Day 019

Objective: dedupe pilot.
Action: exact-hash dedupe.
Action: estimate near duplicates.
Validation: duplicate rate measured.
Artifact: dedupe report.
Decision: choose scalable method.

## Day 020

Objective: filtered-data experiment.
Action: train fixed token budget.
Validation: compare to unfiltered baseline.
Artifact: data-quality experiment report.
Decision: accept/reject filters.

## Day 021

Objective: learned-position architecture baseline repeat.
Action: run controlled baseline.
Validation: reproducible metrics.
Artifact: architecture control.
Decision: prepare positional alternative.

## Day 022

Objective: rotary-position prototype.
Action: implement.
Validation: attention tests pass.
Artifact: model variant.
Decision: overfit test.

## Day 023

Objective: rotary fixed-token comparison.
Action: train matched token budget.
Validation: compare loss and metrics.
Artifact: architecture report.
Decision: keep/reject.

## Day 024

Objective: normalization prototype.
Action: implement RMSNorm variant.
Validation: forward/backward pass.
Artifact: model variant.
Decision: run matched experiment.

## Day 025

Objective: normalization comparison.
Action: matched token run.
Validation: stability and quality compared.
Artifact: norm report.
Decision: select current norm baseline.

## Day 026

Objective: activation-function prototype.
Action: implement gated FFN candidate.
Validation: parameter count correct.
Artifact: model variant.
Decision: fixed-token run.

## Day 027

Objective: activation-function comparison.
Action: matched run.
Validation: throughput and quality compared.
Artifact: FFN report.
Decision: keep/reject.

## Day 028

Objective: depth/width pair design.
Action: create matched-parameter configs.
Validation: parameter counts close.
Artifact: two model specs.
Decision: launch experiments.

## Day 029

Objective: deeper model run.
Action: fixed-token training.
Validation: stable.
Artifact: metrics.
Decision: await width comparison.

## Day 030

Objective: wider model run.
Action: fixed-token training.
Validation: stable.
Artifact: metrics.
Decision: compare depth/width.

## Day 031

Objective: depth/width analysis.
Action: compare loss.
Action: compare task metrics.
Action: compare throughput.
Artifact: decision memo.
Decision: choose topology.

## Day 032

Objective: GQA design.
Action: choose KV-head ratio.
Validation: dimensions valid.
Artifact: GQA spec.
Decision: implement prototype.

## Day 033

Objective: GQA prototype test.
Action: forward/backward.
Validation: mask semantics correct.
Artifact: variant.
Decision: fixed-token run.

## Day 034

Objective: GQA comparison.
Action: matched run.
Validation: quality/memory/inference impact.
Artifact: GQA report.
Decision: keep/reject.

## Day 035

Objective: weight tying comparison.
Action: build tied variant.
Validation: parameter count reduction correct.
Artifact: variant.
Decision: run matched test.

## Day 036

Objective: weight tying run.
Action: matched tokens.
Validation: quality and memory.
Artifact: report.
Decision: select tying policy.

## Day 037

Objective: initialization sweep.
Action: choose std candidates.
Validation: initial activation stats.
Artifact: initialization table.
Decision: launch short runs.

## Day 038

Objective: initialization short runs.
Action: train candidates.
Validation: stability.
Artifact: curves.
Decision: select initialization.

## Day 039

Objective: consolidate architecture.
Action: write new model spec.
Validation: all choices explicit.
Artifact: architecture v1 candidate.
Decision: freeze for serious pretrain pilot.

## Day 040

Objective: corpus mixture design.
Action: define web/code/math proportions.
Validation: actual source token counts known.
Artifact: mixture v1.
Decision: prepare data.

## Day 041

Objective: validation suite expansion.
Action: add domain-balanced validation.
Validation: duplicates checked.
Artifact: validation v2.
Decision: freeze for architecture-v1 runs.

## Day 042

Objective: benchmark decontamination pilot.
Action: fingerprint benchmark.
Action: search corpus subset.
Validation: overlaps reported.
Artifact: contamination report.
Decision: remove confirmed overlaps where feasible.

## Day 043

Objective: corpus manifest v1.
Action: finalize source registry.
Validation: token totals consistent.
Artifact: corpus manifest.
Decision: approve pilot pretrain.

## Day 044

Objective: serious-run recovery drill.
Action: launch short run.
Action: force stop.
Action: resume.
Validation: recovery succeeds.
Artifact: drill report.
Decision: approve longer run.

## Day 045

Objective: pretrain pilot start.
Action: launch selected architecture/data.
Validation: first checkpoints healthy.
Artifact: run directory.
Decision: continue under stop rules.

## Day 046

Objective: monitor pretrain pilot.
Action: inspect loss.
Action: inspect throughput.
Action: inspect thermals.
Validation: no instability.
Artifact: monitoring notes.
Decision: continue/stop.

## Day 047

Objective: checkpoint evaluation.
Action: evaluate early milestones.
Validation: metrics stored.
Artifact: checkpoint table.
Decision: identify trends.

## Day 048

Objective: complete pretrain pilot.
Action: finish token budget.
Validation: final checkpoint reloads.
Artifact: full run.
Decision: sweep checkpoints.

## Day 049

Objective: checkpoint sweep.
Action: evaluate all milestones.
Validation: same protocol.
Artifact: sweep report.
Decision: select best checkpoint.

## Day 050

Objective: baseline freeze.
Action: hash best checkpoint.
Action: document lineage.
Validation: clean reload.
Artifact: official architecture-v1 pilot baseline.
Decision: begin interpretability baseline.

## Day 051

Objective: weight-delta pipeline.
Action: compare early/mid/best.
Validation: tensor names align.
Artifact: delta report.
Decision: rank changing layers.

## Day 052

Objective: activation pipeline.
Action: implement bounded hooks.
Validation: outputs match no-hook run.
Artifact: activation collector.
Decision: collect fixed prompts.

## Day 053

Objective: prompt-set design.
Action: create factual/code/reasoning/control prompts.
Validation: versioned and fixed.
Artifact: prompt set.
Decision: collect across checkpoints.

## Day 054

Objective: activation checkpoint comparison.
Action: collect summary stats.
Validation: storage manageable.
Artifact: layer comparison.
Decision: identify candidates.

## Day 055

Objective: first head ablation.
Action: ablate candidate.
Validation: baseline control run.
Artifact: ablation metrics.
Decision: assess selectivity.

## Day 056

Objective: first MLP ablation.
Action: ablate candidate block/unit group.
Validation: control comparison.
Artifact: ablation metrics.
Decision: assess selectivity.

## Day 057

Objective: activation patching pilot.
Action: define clean/corrupt pair.
Validation: corruption changes output.
Artifact: patching result.
Decision: expand if causal signal appears.

## Day 058

Objective: gradient-similarity pilot.
Action: collect domain gradients.
Validation: same model state.
Artifact: per-layer similarity.
Decision: test mixture implication.

## Day 059

Objective: interpretability replication.
Action: repeat strongest finding on another checkpoint/run.
Validation: effect direction replicates.
Artifact: replication report.
Decision: upgrade/downgrade confidence.

## Day 060

Objective: interpretability report v1.
Action: separate correlation from causal tests.
Validation: claims supported.
Artifact: report.
Decision: define next questions.

## Day 061

Objective: distributed backend setup.
Action: bootstrap optional backends.
Validation: revisions pinned.
Artifact: upstream lock.
Decision: register Project XYZ architecture.

## Day 062

Objective: distributed architecture registration.
Action: map architecture exactly.
Validation: parameter count matches.
Artifact: backend config.
Decision: forward parity.

## Day 063

Objective: forward parity.
Action: align weights/input.
Validation: logits close within expected precision.
Artifact: parity report.
Decision: short distributed run.

## Day 064

Objective: distributed smoke run.
Action: few steps.
Validation: all ranks healthy.
Artifact: smoke checkpoint.
Decision: checkpoint reload.

## Day 065

Objective: distributed resume.
Action: reload.
Validation: state restored.
Artifact: resume report.
Decision: throughput benchmark.

## Day 066

Objective: distributed throughput.
Action: benchmark configurations.
Validation: global batch explicit.
Artifact: scaling table.
Decision: select launch config.

## Day 067

Objective: data-pipeline scale test.
Action: benchmark shard readers/packing.
Validation: GPUs not starved.
Artifact: pipeline profile.
Decision: tune workers/buffers.

## Day 068

Objective: checkpoint storage scale test.
Action: measure save/load time.
Validation: recovery window acceptable.
Artifact: I/O report.
Decision: set cadence.

## Day 069

Objective: async evaluation design.
Action: tie evaluation jobs to immutable checkpoints.
Validation: result identity clear.
Artifact: eval workflow.
Decision: scale-run readiness review.

## Day 070

Objective: scale-run readiness review.
Action: verify gates.
Validation: recovery and budget approved.
Artifact: readiness memo.
Decision: launch or defer.

## Day 071

Objective: scale pretrain start.
Action: launch.
Validation: initial metrics healthy.
Artifact: run.
Decision: continue under stop rules.

## Day 072

Objective: scale pretrain monitoring.
Action: inspect utilization.
Validation: no data starvation.
Artifact: monitoring snapshot.
Decision: continue/tune.

## Day 073

Objective: early scale checkpoint eval.
Action: run validation/tasks.
Validation: pipeline works.
Artifact: early metrics.
Decision: continue/stop.

## Day 074

Objective: mid-run analysis.
Action: compare expected scaling curve.
Validation: no unexplained divergence.
Artifact: mid-run memo.
Decision: continue/stop.

## Day 075

Objective: recovery rehearsal if run permits.
Action: verify a recent checkpoint on separate process.
Validation: reload works.
Artifact: recovery verification.
Decision: retain cadence.

## Day 076

Objective: late-run evaluation.
Action: evaluate milestones.
Validation: best-step trend known.
Artifact: late metrics.
Decision: extend or stop at budget.

## Day 077

Objective: scale pretrain completion.
Action: finish intentional budget.
Validation: final save verified.
Artifact: completed run.
Decision: checkpoint sweep.

## Day 078

Objective: full scale checkpoint sweep.
Action: evaluate milestones.
Validation: protocol consistent.
Artifact: sweep report.
Decision: best base checkpoint.

## Day 079

Objective: base-model release candidate.
Action: export selected checkpoint.
Validation: export parity.
Artifact: base RC.
Decision: begin post-training.

## Day 080

Objective: SFT data schema.
Action: define roles and provenance.
Validation: parser tests.
Artifact: schema.
Decision: collect data.

## Day 081

Objective: SFT pilot data.
Action: create high-quality small set.
Validation: human spot check.
Artifact: SFT v0.
Decision: dedupe/split.

## Day 082

Objective: SFT validation split.
Action: freeze held-out prompts.
Validation: no duplicate prompts.
Artifact: SFT validation.
Decision: smoke training.

## Day 083

Objective: SFT mask test.
Action: inspect token-level loss mask.
Validation: prompt tokens masked as intended.
Artifact: mask visualization/log.
Decision: run smoke SFT.

## Day 084

Objective: SFT smoke.
Action: 100–500 steps.
Validation: instruction behavior improves on pilot.
Artifact: SFT checkpoint.
Decision: LR sweep.

## Day 085

Objective: SFT LR sweep.
Action: short candidates.
Validation: base regressions monitored.
Artifact: sweep.
Decision: select recipe.

## Day 086

Objective: full SFT pilot.
Action: train selected recipe.
Validation: milestone saves.
Artifact: run.
Decision: checkpoint sweep.

## Day 087

Objective: SFT checkpoint sweep.
Action: evaluate instruction/base metrics.
Validation: best step identified.
Artifact: report.
Decision: freeze SFT baseline.

## Day 088

Objective: preference-data pilot.
Action: create/collect pairs.
Validation: labels inspected.
Artifact: preference v0.
Decision: judge reliability test.

## Day 089

Objective: judge reliability.
Action: compare labels to human sample.
Validation: acceptable agreement.
Artifact: reliability report.
Decision: preference experiment or revise data.

## Day 090

Objective: preference optimization smoke.
Action: tiny run.
Validation: no collapse.
Artifact: checkpoint.
Decision: evaluate.

## Day 091

Objective: preference evaluation.
Action: compare SFT parent.
Validation: helpfulness and regressions.
Artifact: report.
Decision: continue/reject.

## Day 092

Objective: verifiable-reward pilot.
Action: choose math/code task.
Validation: reward checker robust.
Artifact: reward harness.
Decision: trajectory generation.

## Day 093

Objective: trajectory pilot.
Action: generate attempts.
Validation: reward distribution sane.
Artifact: trajectories.
Decision: reward-hacking audit.

## Day 094

Objective: reward-hacking audit.
Action: inspect successes/failures.
Validation: no obvious exploit dominates.
Artifact: audit.
Decision: advanced training or revise environment.

## Day 095

Objective: final post-training candidate.
Action: select lineage branch.
Validation: all parent hashes known.
Artifact: candidate.
Decision: full evaluation.

## Day 096

Objective: full evaluation.
Action: run all declared suites.
Validation: raw outputs complete.
Artifact: evaluation package.
Decision: release candidate or reject.

## Day 097

Objective: inference optimization.
Action: benchmark export/quantization candidates.
Validation: quality delta measured.
Artifact: inference table.
Decision: recommended formats.

## Day 098

Objective: release documentation.
Action: model card.
Action: training summary.
Action: data summary.
Action: limitations.
Validation: commands tested.
Artifact: docs.
Decision: final verification.

## Day 099

Objective: clean-room reload.
Action: new environment.
Action: load release.
Action: deterministic test.
Validation: outputs correct.
Artifact: release verification log.
Decision: approve/reject release.

## Day 100

Objective: Project XYZ retrospective.
Action: summarize discoveries.
Action: summarize failures.
Action: summarize compute/data costs.
Action: list strongest evidence.
Action: list unresolved questions.
Validation: all major artifacts linked.
Artifact: retrospective and next roadmap.
Decision: define Project XYZ next generation.

---

# Appendix C — Per-run observation sheet

Copy this block for every meaningful run.

## C001 Identity

Run ID:
Parent run:
Parent checkpoint:
Git commit:
Branch:
Operator:
Date:
Purpose:
Hypothesis:
Status:

## C002 Model

Model ID:
Parameter count:
Trainable count:
Vocabulary:
Context:
Layers:
Model dimension:
Heads:
KV heads:
FFN:
Normalization:
Position encoding:
Activation:
Tying:
Initialization:

## C003 Data

Corpus ID:
Train manifest:
Validation manifest:
Tokenizer hash:
Train tokens available:
Validation tokens:
Mixture:
Filters:
Dedupe:
Decontamination:
EOS policy:
Packing policy:

## C004 Optimization

Optimizer:
LR:
Min LR:
Warmup:
Scheduler:
Beta1:
Beta2:
Epsilon:
Weight decay:
Micro batch:
Accumulation:
World size:
Effective batch:
Sequence length:
Tokens per step:
Grad clip:
Precision:
Compile:

## C005 Hardware

OS:
Python:
PyTorch:
CUDA runtime:
Driver:
GPU model:
GPU count:
VRAM/GPU:
CPU:
RAM:
Storage:
Interconnect:

## C006 Performance

Step 1 loss:
Step 10 loss:
Step 100 loss:
Peak throughput:
Average throughput:
Peak VRAM:
Average GPU utilization:
Peak GPU temperature:
Checkpoint-write time:
Eval time:

## C007 Validation

Eval step:
Validation loss:
Perplexity:
Tokens evaluated:
Benchmark aggregate:
Factual:
Instruction:
Reasoning:
Arithmetic:
Code:
Long context:

## C008 Outcome

Best checkpoint:
Best step:
Best hash:
Primary metric delta:
Regression metric delta:
Unexpected behavior:
Failure modes:
Interpretation:
Confidence label:
Next run:

---

# Appendix D — Debug decision trees

## D001 CUDA false

Question: does `nvidia-smi` work?
If no: fix driver/hardware visibility.
If yes: inspect PyTorch build.
Question: does PyTorch version include CUDA build?
If no: reinstall CUDA-enabled PyTorch.
If yes: inspect environment conflicts.
Question: is wrong Python executable active?
If yes: activate correct environment.
If no: inspect driver/runtime compatibility.
Stop when CUDA tensor allocation succeeds.

## D002 Immediate OOM

Question: does model alone fit?
If no: reduce model or use sharding/offload strategically.
If yes: inspect batch/context.
Reduce micro batch first.
If still OOM: reduce context for debugging.
If stable: restore effective batch with accumulation.
Measure allocated and reserved memory.
Check activation hooks.
Check duplicate model copies.
Check compile warmup overhead.

## D003 NaN at first step

Check input token range.
Check initial logits.
Check initial loss in FP32.
Check gradient norms.
Check learning rate.
Check initialization.
Check attention mask.
Check normalization epsilon.
Check mixed precision.
Disable compile.
Reduce to fixed batch.

## D004 NaN later

Find first bad step.
Save preceding checkpoint.
Replay nearby batches if possible.
Inspect LR at failure.
Inspect gradient norm.
Inspect scaler events.
Inspect data batch.
Inspect temperature/hardware errors.
Try lower LR.
Try BF16 if FP16.
Test without compile.

## D005 Loss flat

Verify parameters change.
Verify gradients nonzero.
Verify target shift.
Verify optimizer LR.
Verify correct dataset.
Overfit one batch.
If overfit fails, trainer/model bug likely.
If overfit succeeds, data/optimization issue likely.
Inspect token entropy and duplication.

## D006 Loss near zero quickly

Check repeated data.
Check tiny dataset.
Check duplicated blocks.
Check target leakage.
Check train/valid overlap.
Decode batches.
Inspect data cursor.
Compare validation.
Do not celebrate yet.

## D007 Resume diverges

Verify model state.
Verify optimizer state.
Verify scheduler state.
Verify scaler state.
Verify RNG state.
Verify global step.
Verify config unchanged.
Verify data order limitation.
Compare uninterrupted control.
Document expected nondeterminism.

## D008 Evaluation mismatch

Verify checkpoint identity.
Verify tokenizer.
Verify model config.
Verify eval mode.
Verify prompt template.
Verify decoding.
Verify benchmark revision.
Verify normalization rules.
Verify max output length.
Verify random seed.

## D009 Free generation bad, benchmark good

Inspect benchmark narrowness.
Inspect chat template.
Inspect sampling.
Inspect overfitting.
Inspect SFT diversity.
Run base completion test.
Run broader prompts.
Measure repetition.
Measure instruction omissions.
Do not replace benchmark; add complementary tests.

## D010 One domain regresses

Confirm metric significance.
Inspect data mixture change.
Inspect gradient conflict.
Inspect continued-training LR.
Inspect overtraining.
Check best earlier checkpoint.
Try mixed replay data.
Try lower LR.
Try shorter continuation.
Document tradeoff.

---

# Appendix E — Project XYZ coding standards for training code

## E001 Imports

Keep standard library first.
Keep third-party libraries second.
Keep Project XYZ imports third.
Avoid wildcard imports.
Avoid import-time GPU allocation.
Avoid hidden network calls at import time.
Keep optional heavy dependencies lazy where practical.

## E002 Paths

Use `pathlib.Path` where practical.
Avoid hard-coded user directories.
Resolve paths explicitly when crossing modules.
Keep generated data outside git-tracked source directories.
Create parent directories intentionally.

## E003 Configuration

Prefer explicit config values.
Validate config early.
Avoid magic constants inside training loop.
Record resolved config.
Use names that include units where ambiguity exists.

## E004 Logging

Rank zero owns ordinary console logs in distributed runs.
Include step in every training line.
Use stable metric names.
Avoid printing giant tensors.
Flush critical checkpoint messages.

## E005 Errors

Fail loudly on:

- missing CUDA when CUDA required;
- missing model file;
- invalid token range;
- empty dataset;
- non-finite loss;
- incompatible checkpoint format;
- impossible dimensions.

## E006 Checkpoints

Never write directly over the only valid checkpoint without temporary staging.
Version format.
Record step.
Record config.
Test reload.

## E007 Evaluation

Use `torch.no_grad()` or inference mode.
Set model eval mode.
Restore train mode when evaluator is embedded in training.
Keep decoding explicit.
Record sample counts.

## E008 Distributed code

Do not let every rank write same file.
Barrier only where needed.
Destroy process group cleanly.
Log world size.
Handle local rank explicitly.

## E009 Performance

Profile before optimizing.
Avoid premature custom CUDA code.
Use pinned memory when it helps.
Use nonblocking transfers correctly.
Measure compile benefits.
Measure attention-kernel benefits.

## E010 Tests

Every bug fix should gain a regression test when practical.
Tiny deterministic fixtures are valuable.
Avoid tests requiring huge downloads.
Separate CPU unit tests from GPU integration tests.

---

# Appendix F — Project XYZ research questions

These are open questions, not claims.

## F001 Data

What data quality metric correlates best with downstream quality at Project XYZ scale?
How much exact deduplication helps the small model?
How much near-deduplication helps?
What web/code/math mixture is best?
Does quality annealing help?
Does late-stage high-quality data help more than constant mixing?
How much synthetic data can be added before diversity falls?
Which data domains cause the strongest gradient conflict?

## F002 Tokenizer

Is 16k vocabulary optimal for the next Project XYZ scale?
Does 32k improve code/English efficiency enough to justify more embedding parameters?
How should Nepali text be handled if multilingual scope expands?
Which numeric tokenization improves arithmetic learning?
Does tokenizer choice affect entity binding measurably?

## F003 Architecture

Does rotary positioning outperform learned positions at matched compute?
Does RMSNorm improve stability?
Does SwiGLU improve quality per parameter?
What depth/width ratio is best at 60M?
What depth/width ratio is best at 120M?
Does GQA preserve quality at small scale?
Does weight tying help or hurt?
Which initialization gives the most stable early training?

## F004 Optimization

What LR scales best with effective batch?
How long should warmup be?
What minimum LR ratio is best?
Does beta2 0.95 outperform 0.98 here?
How often does grad clipping activate?
Does compile mode alter numerical behavior meaningfully?
What token budget reaches diminishing returns?

## F005 Semantics

At what checkpoint does factual QA first emerge?
At what checkpoint does paraphrase generalization emerge?
At what checkpoint does entity tracking emerge?
Does arithmetic require specialized data?
Can instruction behavior improve without catastrophic forgetting?
Which capabilities peak before validation loss stops improving?

## F006 Interpretability

Are capability changes concentrated in later layers?
Do code prompts activate distinct MLP features?
Are factual and reasoning gradients aligned or conflicting?
Can activation patching recover a corrupted entity relation?
Do candidate heads replicate across seeds?
Are large weight-delta layers actually causal?

## F007 Scaling

How does validation loss scale from 15M to 120M under fixed tokens?
How does benchmark quality scale?
Which size gives best quality per laptop-hour?
At what size does local training become impractical?
When does distributed overhead become worthwhile?

## F008 Post-training

How much SFT data is enough for stable instruction behavior?
What mixture preserves base-model breadth?
Does assistant-only loss materially reduce forgetting?
Can verifiable-reward training improve arithmetic without hurting language quality?
What reward-hacking patterns appear first?

---

# Appendix G — Start-work command sheet

## G001 Environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## G002 PyTorch check

```powershell
python -c "import torch; print('Torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
```

## G003 Stack dependencies

```powershell
python -m pip install -r project_xyz_stack/requirements.txt
```

## G004 Validate model

```powershell
python project_xyz_stack/validate_stack.py
```

## G005 Prepare train subset

```powershell
python project_xyz_stack/prepare_data.py `
  --dataset HuggingFaceFW/fineweb `
  --dataset-config sample-10BT `
  --split train `
  --tokenizer tokenizer.json `
  --output data/project_xyz/train.bin `
  --max-docs 10000
```

## G006 Prepare validation subset

Use a deliberately separate source window or frozen local JSONL split.
Do not generate validation by reading the same prepared training file.

Example local JSONL:

```powershell
python project_xyz_stack/prepare_data.py `
  --jsonl data/raw/project_xyz_valid.jsonl `
  --text-field text `
  --tokenizer tokenizer.json `
  --output data/project_xyz/valid.bin
```

## G007 Train

```powershell
python project_xyz_stack/train.py `
  --config project_xyz_stack/config/project_xyz.json
```

## G008 Evaluate checkpoint

```powershell
python project_xyz_stack/evaluate.py `
  --config project_xyz_stack/config/project_xyz.json `
  --checkpoint runs/project_xyz_pretraining/latest.pt `
  --data data/project_xyz/valid.bin `
  --batches 100
```

## G009 Bootstrap optional scale backends

```powershell
python project_xyz_stack/bootstrap_project_xyz.py
```

## G010 Git status

```powershell
git status
git log --oneline -10
```

---

# Appendix H — The first Project XYZ build sequence

Follow this sequence if you want to start immediately.

### Step 1

Run the CUDA check.
Do not proceed on CPU accidentally.

### Step 2

Run `validate_stack.py`.
Save the output.

### Step 3

Open `src/config/model_config.py`.
Decide whether the next experiment uses the existing model or a new model.

### Step 4

If using a new architecture, create it under `src/models/` or a new versioned architecture folder.
Do not delete the current reference model.

### Step 5

Point `project_xyz_stack/config/project_xyz.json` at the model and config.

### Step 6

Run validator again.

### Step 7

Run overfit-a-batch test.
Add a dedicated script if necessary.

### Step 8

Prepare a small train dataset.

### Step 9

Prepare a separate validation dataset.

### Step 10

Inspect both manifests.

### Step 11

Decode random windows to verify data.

### Step 12

Set training steps to a smoke-test value.

### Step 13

Train.

### Step 14

Verify checkpoint save.

### Step 15

Stop and resume.

### Step 16

Evaluate validation loss and perplexity.

### Step 17

Run existing Project XYZ semantic benchmarks.

### Step 18

Write experiment conclusion.

### Step 19

Change one variable.

### Step 20

Repeat.

---

# Appendix I — Definition of “done”

A Project XYZ experiment is **not done** because training reached the final step.

It is done when:

- hypothesis is recorded;
- code revision is recorded;
- configuration is recorded;
- tokenizer is identified;
- data is identified;
- checkpoint is reloadable;
- evaluation is complete;
- regressions are checked;
- artifacts are preserved;
- conclusion is written;
- next action is clear.

A Project XYZ model release is **not done** because a weight file uploads successfully.

It is done when:

- clean reload succeeds;
- architecture config is included;
- tokenizer is included;
- lineage is included;
- evaluation protocol is included;
- limitations are included;
- hashes are included;
- reproduction instructions are tested.

---

# Appendix J — Final operating rules

Rule 001: Project XYZ owns the model identity.
Rule 002: Backends are replaceable.
Rule 003: Data provenance is part of the experiment.
Rule 004: The tokenizer is part of the model.
Rule 005: Loss is not understanding.
Rule 006: Benchmarks are not free generation.
Rule 007: Free generation is not a benchmark.
Rule 008: Intermediate checkpoints are valuable.
Rule 009: A script is not a completed experiment.
Rule 010: A final checkpoint is not automatically the best checkpoint.
Rule 011: A low training loss is not automatically good.
Rule 012: A high benchmark score is not automatically generalization.
Rule 013: A correlation in activations is not causal evidence.
Rule 014: An ablation that hurts everything is not capability localization.
Rule 015: One seed is weak evidence for small differences.
Rule 016: One prompt is weak evidence for a neuron/head interpretation.
Rule 017: One successful launch is weak evidence for distributed correctness.
Rule 018: A checkpoint that has never reloaded is untrusted.
Rule 019: A dataset that has never been sampled manually is untrusted.
Rule 020: A filter whose rejected samples were never inspected is untrusted.
Rule 021: A tokenizer whose special IDs were never checked is untrusted.
Rule 022: An evaluator with untested scoring logic is untrusted.
Rule 023: A result without a code revision is incomplete.
Rule 024: A result without a config is incomplete.
Rule 025: A result without a checkpoint ID is incomplete.
Rule 026: A comparison with different tokenizers needs explicit justification.
Rule 027: A comparison with different data needs explicit justification.
Rule 028: A comparison with different token budgets needs explicit labeling.
Rule 029: A comparison with different decoding needs explicit labeling.
Rule 030: A distributed scale-up begins with parity testing.
Rule 031: Memory estimates must include optimizer and activations.
Rule 032: Compute estimates should include evaluation and checkpoint overhead.
Rule 033: Thermal stability matters on laptop experiments.
Rule 034: Abnormal fan behavior is a hardware issue, not a training hyperparameter.
Rule 035: Secrets never belong in git.
Rule 036: Negative results belong in research history.
Rule 037: Stop bad runs early when evidence says they are bad.
Rule 038: Preserve enough artifacts to explain surprising results.
Rule 039: Prefer the cheapest experiment that can falsify a hypothesis.
Rule 040: Scale only after correctness.
Rule 041: Optimize only after profiling.
Rule 042: Post-train only after base training is stable.
Rule 043: Preference-train only after preference quality is validated.
Rule 044: Use verifiable rewards when the task supports them.
Rule 045: Reward improvements require reward-hacking audits.
Rule 046: Export requires parity testing.
Rule 047: Quantization requires quality testing.
Rule 048: Release notes must state limitations.
Rule 049: Model cards must distinguish measured facts from goals.
Rule 050: Project XYZ should remain reproducible as it grows.

---

# End of Project XYZ master manual

Start with `validate_stack.py`.
Then prepare a tiny dataset.
Then run a tiny real training job.
Then test resume.
Then evaluate.
Only after those five steps should you increase model size, token budget, or infrastructure complexity.
