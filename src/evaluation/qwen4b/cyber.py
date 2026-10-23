"""Cybersecurity Benchmark Suite for Qwen 4B Evaluation.

Covers:
- Networking security (TCP/IP, DNS, TLS, firewalls)
- Secure coding (SQL injection, XSS, input sanitization)
- Vulnerability analysis (CVE remediation, CVSS)
- Incident response & SOC operations (triage, containment)
- Log analysis (web server, auth logs, Windows Event IDs)
- Web & API security (OAuth2, CORS, JWT, OWASP)
- Cryptography (hashing, salting, PKI, AES/RSA)
- Cloud & container security (Docker, Kubernetes RBAC)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class EvalQuestion:
    id: str
    category: str
    prompt: str
    required_concepts: List[str]
    forbidden_concepts: List[str] = field(default_factory=list)
    weight: float = 1.0


CYBER_BENCHMARK_QUESTIONS: List[EvalQuestion] = [
    EvalQuestion(
        id="cyber_net_01",
        category="networking_security",
        prompt="Explain the difference between TCP SYN flood attacks and UDP amplification attacks at the transport and network layers, and describe two distinct mitigation strategies for each.",
        required_concepts=["syn flood", "syn cookies", "three-way handshake", "amplification", "spoofed", "dns", "ntp", "rate limit"],
        forbidden_concepts=["turn off tcp", "disable ip"],
    ),
    EvalQuestion(
        id="cyber_net_02",
        category="networking_security",
        prompt="In the context of TLS 1.3, how does the handshake achieve forward secrecy (PFS), and why was static RSA key exchange removed?",
        required_concepts=["forward secrecy", "diffie-hellman", "ephemeral", "ecdhe", "rsa", "compromise", "private key"],
    ),
    EvalQuestion(
        id="cyber_sec_code_01",
        category="secure_coding",
        prompt="Provide a Python function using SQLite that securely queries user accounts by username without SQL injection vulnerability. Explain why parameterized queries prevent injection.",
        required_concepts=["parameterized", "placeholder", "?", "ast", "separation", "data", "code", "execute"],
        forbidden_concepts=["f\"", "format(", "% s", "concatenate"],
    ),
    EvalQuestion(
        id="cyber_sec_code_02",
        category="secure_coding",
        prompt="How should a modern web application securely store user passwords in a database? Specify the cryptographic algorithm, work factor / salt requirements, and why simple SHA-256 is insufficient.",
        required_concepts=["argon2", "bcrypt", "salt", "work factor", "slow", "gpu", "rainbow table", "sha-256"],
        forbidden_concepts=["md5", "des", "plain text"],
    ),
    EvalQuestion(
        id="cyber_vuln_01",
        category="vulnerability_analysis",
        prompt="What is a Use-After-Free (UAF) memory corruption vulnerability? Explain the mechanism, potential consequence, and how developers can eliminate UAF in C/C++ code.",
        required_concepts=["heap", "pointer", "free", "dangling pointer", "null", "reallocate", "smart pointer", "raii"],
    ),
    EvalQuestion(
        id="cyber_vuln_02",
        category="vulnerability_analysis",
        prompt="Explain Server-Side Request Forgery (SSRF). In a cloud environment like AWS, how can an SSRF vulnerability lead to credential compromise, and what defensive architecture mitigates it?",
        required_concepts=["ssrf", "metadata", "169.254.169.254", "imdsv2", "token", "allowlist", "egress"],
    ),
    EvalQuestion(
        id="cyber_ir_01",
        category="incident_response",
        prompt="Outline the containment and eradication steps an incident responder should take upon discovering an active Cobalt Strike Beacon beaconing out from an internal enterprise workstation.",
        required_concepts=["isolate", "network", "containment", "dump memory", "edr", "c2", "ip/domain", "firewall block", "revoke credentials", "re-image"],
    ),
    EvalQuestion(
        id="cyber_log_01",
        category="log_analysis",
        prompt="Analyze the following web server log snippet and identify the attack technique: `GET /catalog.php?id=1%20UNION%20SELECT%20null,username,password%20FROM%20users-- HTTP/1.1 200 4521`. What is the attacker attempting, and what log evidence indicates success or failure?",
        required_concepts=["sql injection", "union", "exfiltrat", "status 200", "payload", "response size"],
    ),
    EvalQuestion(
        id="cyber_web_01",
        category="web_api_security",
        prompt="Compare Cross-Site Request Forgery (CSRF) and Cross-Site Scripting (XSS). Why does implementing SameSite cookie attributes protect against CSRF but not XSS?",
        required_concepts=["csrf", "xss", "samesite", "cross-origin", "javascript", "browser", "context", "token"],
    ),
    EvalQuestion(
        id="cyber_cloud_01",
        category="cloud_container_security",
        prompt="In Docker and Kubernetes environments, why is running containers as non-root essential? What is the risk of granting `CAP_SYS_ADMIN` or mounting `/var/run/docker.sock`?",
        required_concepts=["non-root", "container breakout", "escape", "privilege", "docker.sock", "host", "root", "cap_sys_admin"],
    ),
]
