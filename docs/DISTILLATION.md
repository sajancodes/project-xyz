# Track B: Teacher Distillation & Synthetic Generation Architecture

## 1. Distillation Methodology: Response Distillation vs Logits Distillation

It is vital to distinguish between classical knowledge distillation and the response distillation methodology employed in this project:

- **Classical Knowledge Distillation (KD)**: Formulated by Hinton et al., classical KD requires computing the soft probability distribution (logits) across the full vocabulary $\mathcal{V}$ from both the teacher model ($z^T$) and student model ($z^S$), minimizing Kullback-Leibler (KL) divergence:
  $$\mathcal{L}_{KD} = (1 - \alpha) \mathcal{L}_{CE}(y, \sigma(z^S)) + \alpha T^2 D_{KL}(\sigma(z^S / T) \parallel \sigma(z^T / T))$$
  This approach requires white-box token-level logit access to the teacher, which is impossible with black-box API models or compute-prohibitive for large teachers.

- **Response Distillation / Teacher-Generated SFT (Used Here)**:
  Project XYZ utilizes **sequence-level response distillation / teacher-generated SFT**. High-capability teacher models (or expert-crafted domain completions) generate clean, defensively grounded explanations for curated cybersecurity prompts. The student model (`Qwen3-4B-Instruct-2507`) is fine-tuned on these sequences using standard cross-entropy loss applied exclusively to assistant tokens.
  
  **No false claims of logits-level distillation are made in this repository.**

---

## 2. Provider-Neutral Teacher Abstraction

The distillation engine (`src/distillation/teacher.py`) implements an abstract provider interface `TeacherProvider`:

```
TeacherProvider (ABC)
├── OpenAITeacherProvider        # Standard OpenAI-compatible API (Ollama, vLLM, DeepSeek, OpenAI)
├── LocalTeacherProvider         # Local model / transformers pipeline
└── ManualDatasetTeacherProvider # Seed & curated expert dataset loader
```

### Zero Secrets Hardcoding
No API keys or provider tokens are committed to the repository or embedded in notebooks.
Providers dynamically query environment variables:
- `OPENAI_API_KEY`: API token for OpenAI or compatible gateways.
- `TEACHER_API_BASE`: Target endpoint URL (e.g., `http://localhost:11434/v1` for local Ollama, `https://api.openai.com/v1`).

---

## 3. Generating and Expanding Datasets

To run the automated dataset generator locally or with a teacher provider:

```bash
# Ingest raw seeds, validate, deduplicate, and split
python scripts/prepare_qwen4b_dataset.py

# Validate processed splits
python scripts/validate_qwen4b_dataset.py
```
Outputs are written to `data/qwen4b/processed/` and `data/qwen4b/stats.json`.
