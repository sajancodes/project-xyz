# Track B: Cybersecurity Dataset Architecture, Validation & Hygiene

## 1. Scope & Covered Domains

The cybersecurity dataset is structured to provide comprehensive defensive grounding across modern enterprise and cloud computing environments:

- **Networking**: TCP/IP mechanics, SYN flood mitigation, UDP amplification, DNSSEC validation, HTTP/HTTPS headers, TLS 1.3 key exchange, stateful firewall inspection, routing protocols.
- **Authentication & Identity**: OAuth2 flows, OpenID Connect, JWT signing and validation, multi-factor authentication (TOTP/FIDO2), password hashing algorithms (Argon2id vs bcrypt vs SHA-256).
- **Operating System Security**:
  - *Linux*: Granular Capabilities (`getcap`/`setcap`), namespace isolation, cgroups, file permissions, PAM configuration, SSH hardening.
  - *Windows*: Active Directory Kerberos authentication, Ticket Granting Service (TGS), PAC validation, Golden Ticket / Silver Ticket defense, KRBTGT double-reset protocols, Sysmon event logging.
- **Secure Coding**:
  - *Python*: SQL injection prevention via parameterized statements, safe deserialization, SSRF mitigation via allowlisting and egress filtering, path traversal prevention.
  - *JavaScript / Node.js*: Prototype pollution mitigation, Content Security Policy (CSP) headers, CORS configurations, non-root Docker deployment.
- **Vulnerability Analysis & CVE Remediation**: CVE mechanics, CVSS scoring, Log4Shell (CVE-2021-44228) JNDI lookup dissection, memory corruption mechanisms (Use-After-Free, buffer overflow defenses: ASLR, DEP/NX, Stack Canaries).
- **Incident Response & SOC Operations**: Order of volatility (RFC 3227), live memory capture vs rebooting, Volatility 3 forensics (`pslist`, `netscan`, `malfind`), SIEM detection engineering (Sigma rules, Splunk SPL), MITRE ATT&CK mapping.
- **Cryptography**: Symmetric ciphers (AES-256-GCM authenticated encryption), asymmetric cryptography (RSA-3072 vs Ed25519), PKI certificate authority chains, OCSP stapling vs CRL.
- **Cloud & Container Security**: AWS IAM least-privilege policy scoping, S3 secure transport enforcement, Docker multi-stage builds, non-root execution, Kubernetes NetworkPolicies, Pod Security Standards.

### Ethical & Defensive Grounding
All potentially offensive examples (exploits, payloads, injection strings) are strictly contextualized within **authorized lab simulations, CTFs, toy environments, detection engineering, or patch remediation**. Any request referencing unauthorized third-party infrastructure is rejected.

---

## 2. Dataset Schema (JSONL)

Every record is stored as a single-line JSON object adhering to this schema:

```json
{
  "id": "cyber_seed_001",
  "category": "networking",
  "difficulty": "intermediate",
  "source": "curated-expert-seed",
  "tags": ["networking", "tcp", "handshake", "dos"],
  "messages": [
    {
      "role": "system",
      "content": "You are a specialized cybersecurity AI assistant providing rigorous, defensively grounded technical explanations."
    },
    {
      "role": "user",
      "content": "How does the TCP three-way handshake establish a connection..."
    },
    {
      "role": "assistant",
      "content": "The transmission control protocol (TCP) initiates a reliable..."
    }
  ]
}
```

---

## 3. Strict Cleaning & Validation Pipeline

Before admission into the dataset splits, all raw records are filtered through `src/distillation/clean.py` and `src/distillation/validate.py`:

1. **Whitespace & Unicode Normalization**: Formats newlines, converts non-breaking spaces, strips null bytes.
2. **Secret & Credential Scrubbing**: Scans for real/fake API keys (`sk-*`, AWS `AKIA*`, GitHub tokens), private keys (`-----BEGIN PRIVATE KEY-----`), and JWT tokens.
3. **Placeholder & Refusal Elimination**: Strips canned assistant refusals ("As an AI language model...", "I cannot assist with..."), TODO tags, and boilerplate filler.
4. **Repetition & Loop Detection**: Computes word n-gram repetition ratios to catch model generation loops.
5. **Truncation Detection**: Rejects samples ending mid-sentence or with unclosed code fences.
6. **Defensive Grounding Audit**: Requires offensive keywords (exploit, payload, bypass) to be accompanied by defensive or lab context terms.

---

## 4. Exact & Near Deduplication

- **Exact Deduplication**: Computes SHA-256 hashes of whitespace-normalized user prompts and enforces unique record IDs.
- **Near Deduplication**: Computes word 3-shingle sets and calculates Jaccard similarity between all records:
  $$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
  Records with $J(A, B) \ge 0.80$ are flagged as near-duplicates and rejected.

---

## 5. Deterministic Splitting & Contamination Prevention

- **Split Ratio**: 90% Train / 5% Validation / 5% Test.
- **Random Seed**: Seed `42` ensures bit-for-bit reproducible splits.
- **Contamination Guard**: Evaluates cross-split Jaccard similarity. The independent benchmark set (`data/qwen4b/benchmarks/cyber_benchmark.jsonl`) is strictly verified to have **zero** exact or near-duplicate overlap with the training set.
