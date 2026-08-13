# TROUBLESHOOTING

Real problems encountered in Project XYZ, with symptoms, causes, verified fixes, and prevention. Only fixes that were actually observed to work are listed.

## 1. FineWeb loading failure / download hang

- **Symptom**: `train_semv2e.py` (or any FineWeb-dependent script) hangs or errors while loading `HuggingFaceFW/fineweb`.
- **Cause**: first-time stream download; no cached copy; unauthenticated hub rate limits (see the warn about HF_TOKEN in `semv2e_train_log.txt`).
- **Fix**: use **smoke mode** for code checks: `python src\training\train_semv2e.py --smoke` uses a preset text buffer (instant, no network). For real runs, let the cache populate once.
- **Prevent**: set `HF_TOKEN` in the environment to avoid rate-limit warnings.

## 2. PowerShell + `python -c "..."` parse errors

- **Symptom**: `python -c` inline commands fail with bizarre parsing errors (quotes, commas, `;`).
- **Cause**: PowerShell interprets `"`, `,`, `;`, `|` inside the command string.
- **Fix**: write a small `.py` file (in a temp dir) and run `python file.py`. Works every time.
- **Prevent**: don't use `python -c` with complex code in PowerShell.

## 3. torch is CPU-only (e.g. `2.13.0+cpu`) even though CUDA GPU exists

- **Symptom**: `torch.cuda.is_available()` returns False; training runs slowly on CPU.
- **Cause**: some install scripts / checks resolved the bare `torch` from the `uv` base python (`C:\Users\Sajan\AppData\Roaming\uv\python\cpython-3.11-...`), which carries a CPU build. The project venv (`.venv`) uses torch **2.11.0+cu128**.
- **Fix**:
  ```powershell
  .venv\Scripts\activate
  python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  # expect: 2.11.0+cu128 True
  ```
  If it prints a cpu build, reinstall in venv: `pip uninstall torch; pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128`.
- **Prevent**: always activate `.venv`; never rely on the bare `python`.

## 4. Wrong checkpoint used (latest_checkpoint.pt vs checkpoint_best.pt)

- **Symptom**: evaluations/tests give nonsensical results even though "the same" checkpoint is used.
- **Cause**: `checkpoints/latest_checkpoint.pt` is a **step-20 SMOKE-TEST** checkpoint (loss 3.6832, type `smoke_test`), not the best model. The best is `checkpoints/best/checkpoint_best.pt` (step 3000, loss 3.5710, `semantic_v2_mixed`).
- **Fix**: always load `checkpoints/best/checkpoint_best.pt`, or verify via metadata first (see CHECKPOINT_GUIDE §6).
- **Prevent**: a guard that refuses to run inference on `training_type == "smoke_test"`.

## 5. Repetition / echo in generation

- **Symptom**: model repeats the prompt, repeats a phrase, or loops without reaching an answer.
- **Cause**: tiny model + stochastic decoding + prompt-continuation training traces. Echo is treated as FAIL in evaluation.
- **Mitigations (verified)**: assistant-only loss masking (SEMV2+/SEMV2E) reduced echo; lower temperature/top-k helps in interaction; greedy decoding removes randomness for benchmark decisions.
- **Prevent**: keep loss masking; test with real prompts, not just benchmark templates.

## 6. Same prompt gives very different answers across runs (13/15 vs 5/15)

- **Symptom**: interactive runs of the same checkpoint score wildly differently.
- **Cause**: sampling randomness (temp 0.3, top-k 40).
- **Fix**: use `canonical_eval.py` (greedy, temp 0) for any decision. Real-generation tests are qualitative; report the configuration.
- **Prevent**: always record decoding settings; decide promotions on deterministic metrics.

## 7. Unicode / encoding errors on Windows (cp1252)

- **Symptom**: `UnicodeEncodeError: 'charmap' codec can't encode ...` or garbled output (`�`) when printing/matplotlibing non-ASCII text.
- **Cause**: Windows console default codepage cp1252.
- **Fix**: set `PYTHONIOENCODING=utf-8`; write output to files with `encoding="utf-8"`; keep console prints ASCII-only.
- **Prevent**: avoid non-ASCII in console output (e.g. the `→` character caused an early failure; use `->`).

## 8. CUDA out-of-memory

- **Symptom**: OOM during training on the RTX 4050 (6 GB).
- **Cause**: batch size / seq length too large, or CPU/GPU mis-scheduling.
- **Verified**: batch 16 × seq 512 at fp32 fits on 6 GB; SEMV2E ran at ~42k tok/s. Do not scale batch beyond ~16 without checking.
- **Prevent**: keep batch=16, seq=512 for this model.

## 9. "Fact injection did nothing" (repetition = no learning)

- **Symptom**: repeated training on one fact yields 0% generalization.
- **Cause**: memorization, not understanding.
- **Fix**: use procedural, paraphrased, varied data (see DATA_GUIDE).
- **Prevent**: never test generalization on the exact training string.

## 10. Evaluation metadata confusion (`_verify` files, mislabeled checkpoints)

- **Symptom**: two JSON files for the "same" model disagree (e.g. `bench_semv2c_3000.json` vs `bench_best_verify.json`).
- **Cause**: some files duplicate results; file NAMES sometimes mislabel lineage (e.g. `eval_mixed2k_fixed.json` labels best as "step 2000 mixed", but the real best is step 3000 semantic_v2_mixed).
- **Fix**: always trust `checkpoint_info` inside each JSON; compare steps/loss/types; use the audited table in CHECKPOINT_GUIDE.
- **Prevent**: when saving eval outputs, name them after `checkpoint_info` content, not file names.

## 11. SEMV2E training: base checkpoint loaded wrong

- **Symptom**: training starts from a low-quality or smoke checkpoint.
- **Cause**: pointing at `checkpoints/latest_checkpoint.pt` (smoke) instead of `checkpoints/best/checkpoint_best.pt`.
- **Fix**: verify base prints `step 3000, loss 3.5710` before starting (the trainer logs it).
- **Prevent**: print-and-check base checkpoint metadata at startup.

---

*Related: [docs/TRAINING_GUIDE.md](TRAINING_GUIDE.md), [docs/CHECKPOINT_GUIDE.md](CHECKPOINT_GUIDE.md).*