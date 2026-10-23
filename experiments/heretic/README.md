# Future Experimental Track: Refusal-Direction Ablation & Heretic Integration

## 1. Strict Isolation from Canonical Model

> [!IMPORTANT]
> **CRITICAL ARCHITECTURAL ISOLATION RULE**:
> Refusal-direction ablation, weight vector surgery, or Heretic techniques MUST NEVER be applied to the primary/canonical QLoRA training pipeline.
> The primary canonical training path is strictly:
> ```
> Base Qwen 4B -> Baseline Benchmark -> Cybersecurity QLoRA -> Post-Training Benchmark -> Canonical Adapter
> ```

Any future experimentation with refusal vector modification must operate strictly on a **copy** of the saved canonical adapter and must run in this isolated experimental directory.

---

## 2. Experimental Concept

Large instruction-tuned models often contain specific directional sub-spaces in their activation layers responsible for triggering refusal responses.
In future research, after canonical benchmarking is complete:
1. Make an exact copy of the trained adapter weights:
   ```bash
   cp -r outputs/qwen3_4b_cyber/final_adapter experiments/heretic/candidate_adapter
   ```
2. Identify the refusal direction across intermediate hidden states $\mathbf{r} = \mu_{\text{refuse}} - \mu_{\text{comply}}$.
3. Orthogonalize weight projections against $\mathbf{r}$ to evaluate changes in refusal vs defensive depth.
4. Execute full post-ablation benchmarks using `scripts/benchmark_qwen4b.py` with tag `heretic_ablation` to measure any divergence or regression.
