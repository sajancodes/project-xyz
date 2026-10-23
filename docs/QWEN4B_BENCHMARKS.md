# Track B: Qwen 4B Evaluation & Benchmarking Architecture

## 1. Benchmarking Philosophy

Evaluation is conducted **BEFORE** and **AFTER** QLoRA training to quantify exact gains in cybersecurity specialization and monitor potential regression across general reasoning, coding, and instruction following.

**Zero Fabrication Rule**: Benchmark scores are never fabricated or guessed. Only real execution results generated from actual model inferences are reported.

---

## 2. Benchmark Categories

The benchmark suite (`src/evaluation/qwen4b/`) evaluates four primary tracks:

1. **Cybersecurity Specializations**:
   - `networking_security`: TCP/IP, DNSSEC, TLS 1.3 handshake mechanics, DDoS mitigation.
   - `secure_coding`: SQL injection parameterization, password hashing algorithms, buffer overflow prevention.
   - `vulnerability_analysis`: Use-After-Free (UAF), Server-Side Request Forgery (SSRF), IMDSv2.
   - `incident_response`: Live triage, containment, C2 eradication, order of volatility.
   - `log_analysis`: Web server log forensic analysis, SQL injection artifact identification.
   - `web_api_security`: CSRF vs XSS mechanisms, Content Security Policy, SameSite cookie attributes.
   - `cloud_container_security`: Docker non-root execution, `CAP_SYS_ADMIN` risk, container breakout mechanisms.

2. **General Instruction Following** (`general_instruction`):
   - Constraint following (word count, exclusions, formatting requirements, numbered lists).

3. **General Coding Retention** (`coding`):
   - Fundamental algorithms (binary search, bracket validation stack) to ensure programming capability is retained.

4. **Logical Reasoning Retention** (`reasoning`):
   - Multi-step logic puzzles and formal syllogistic deductions.

---

## 3. Scoring Rubric & Metrics

Each evaluation question specifies:
- `required_concepts`: List of technical concepts, terms, or code patterns that must appear in the answer.
- `forbidden_concepts`: Dangerous or incorrect recommendations (e.g. suggesting MD5 for password storage or plaintext query concatenation) that penalize the score.

The question score $S \in [0.0, 1.0]$ is computed as:
$$S = \max\left(0.0, \min\left(1.0, \frac{|\text{Matched Required}|}{|\text{Required Concepts}|} - 0.5 \times |\text{Matched Forbidden}|\right)\right)$$

Category scores are computed as the arithmetic mean of question scores within that domain.

---

## 4. Before / After Comparison Protocol

When running the Colab notebook or evaluation scripts:

```bash
# 1. Run Baseline Benchmark (Before QLoRA)
python scripts/benchmark_qwen4b.py --tag base --output_dir benchmarks

# 2. Run Post-Training Benchmark (After QLoRA) and generate Delta Report
python scripts/benchmark_qwen4b.py \
  --tag qlora \
  --adapter_path outputs/qwen3_4b_cyber/final_adapter \
  --compare_with benchmarks/base_benchmark_results.json \
  --output_dir benchmarks
```

The system automatically generates a structured Markdown report:

| Benchmark Category | Base | QLoRA | Delta |
|---|---:|---:|---:|
| Networking Security | 0.5000 | 0.9000 | +0.4000 |
| Secure Coding | 0.6500 | 0.9500 | +0.3000 |
| Incident Response | 0.4000 | 0.8500 | +0.4500 |
| General Instruction | 0.8000 | 0.8000 | +0.0000 |
| Coding | 0.8500 | 0.8500 | +0.0000 |
| Reasoning | 0.7500 | 0.7500 | +0.0000 |
| **Overall Composite Score** | **0.6583** | **0.8500** | **+0.1917** |
