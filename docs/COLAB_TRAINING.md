# Google Colab & VS Code Training Guide (Track B — Qwen 4B QLoRA)

This guide documents the exact workflow for running the full QLoRA fine-tuning experiment on an **NVIDIA T4 GPU** in Google Colab using the official VS Code Google Colab extension.

---

## 1. Prerequisites & Manual Steps Required From You

All repository code, datasets, configurations, tests, and notebooks are prepared.
The ONLY steps you need to perform manually are:

1. **Open the Notebook in VS Code**:
   Navigate to `notebooks/qwen3_4b_cyber_qlora_colab.ipynb` inside VS Code.
2. **Select the Google Colab Kernel**:
   In the top-right kernel selector of the notebook, choose **Colab**.
3. **Authenticate your Google Account**:
   Sign in to your Google Account when prompted by the extension.
4. **Select an NVIDIA T4 GPU Runtime**:
   Configure the runtime type to **T4 GPU** (16 GB VRAM).
5. **Authorize Google Drive**:
   When prompted in Cell 5, grant authorization for Google Drive mounting.
6. **Run the Notebook Cells Sequentially**:
   Execute cells top-to-bottom through Cell 15.
7. **Execute the clearly marked FINAL TRAINING CELL**:
   Run Cell 16 to begin actual model training.

---

## 2. Notebook Execution Map (Cells 1 to 24)

| Cell | Type | Purpose |
|---|---|---|
| **Cell 1** | Markdown | Project overview & architecture explanation |
| **Cell 2** | Code | GPU check with `nvidia-smi` (verifies T4 VRAM) |
| **Cell 3** | Code | Install pinned QLoRA dependencies from `requirements-qlora.txt` |
| **Cell 4** | Code | Print dependency versions (PyTorch, Transformers, PEFT, TRL, BNB) |
| **Cell 5** | Code | Mount Google Drive & verify persistence directories |
| **Cell 6** | Code | Hyperparameter configuration variables |
| **Cell 7** | Code | Clone or access repository files |
| **Cell 8** | Code | Run automated dataset schema & quality validation |
| **Cell 9** | Code | Load official Qwen tokenizer |
| **Cell 10** | Code | Load `Qwen/Qwen3-4B-Instruct-2507` in NF4 4-bit |
| **Cell 11** | Code | Attach PEFT LoRA adapter |
| **Cell 12** | Code | Inspect and print trainable parameter counts |
| **Cell 13** | Code | Run tiny forward/backward smoke step (verifies gradient flow & OOM safety) |
| **Cell 14** | Code | Run **BASELINE** benchmark on un-tuned base model & persist scores |
| **Cell 15** | Code | Initialize Trainer with assistant-only loss masking and Drive backup callback |
| **Cell 16** | Code | **FINAL TRAINING CELL** — Executes full QLoRA fine-tuning |
| **Cell 17** | Code | Save final LoRA adapter locally and sync to Google Drive |
| **Cell 18** | Code | Clear GPU memory for clean evaluation |
| **Cell 19** | Code | Reload adapter from Google Drive |
| **Cell 20** | Code | Run single-prompt inference verification |
| **Cell 21** | Code | Run **POST-TRAINING** benchmark on fine-tuned model |
| **Cell 22** | Code | Generate Before vs After Markdown comparison table |
| **Cell 23** | Code | Interactive query test block |
| **Cell 24** | Code | Optional adapter merge and 16-bit weight export |

---

## 3. Google Drive Persistence & Auto-Resume

All training outputs are persisted directly under:
```
MyDrive/
└── ProjectXYZ/
    └── qwen3-4b-cyber/
        ├── checkpoints/        # Intermediate checkpoint-25, checkpoint-50...
        ├── adapters/           # final_cyber_adapter/
        ├── benchmarks/         # base_benchmark_results.json, qlora_benchmark_results.json
        ├── logs/               # Training logs
        └── reports/            # comparison_report.md
```

### Resume Behavior
If the Colab runtime disconnects or times out:
1. Re-open the notebook and connect to a T4 GPU.
2. Run cells up to Cell 15.
3. The trainer automatically inspects `MyDrive/ProjectXYZ/qwen3-4b-cyber/checkpoints/` and finds the highest step checkpoint (e.g. `checkpoint-100`).
4. Cell 16 automatically resumes training from that exact step without manual edits.
