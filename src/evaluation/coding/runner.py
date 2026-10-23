"""Sandboxed Subprocess Execution Runner for Coding Benchmark.

Executes candidate Python completions against unit tests in isolated subprocesses with timeouts.
Calculates pass@1 accuracy.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.coding.tasks import CODING_BENCHMARK_TASKS, CodingTask
from src.models.qwen4b.inference import generate_response

logger = logging.getLogger(__name__)


def clean_code_completion(raw_text: str, entry_point: str) -> str:
    """Extract clean Python code from model completion."""
    # Look for python code blocks
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, raw_text, re.DOTALL | re.IGNORECASE)
    if matches:
        # Choose the block that defines the entry_point or the longest block
        for block in matches:
            if f"def {entry_point}" in block:
                return block
        return matches[0]

    # If no code fence, take lines that look like Python code
    lines = raw_text.split("\n")
    code_lines = []
    found_entry = False
    for line in lines:
        if f"def {entry_point}" in line:
            found_entry = True
        if found_entry:
            # Stop if markdown header or prose section starts
            if line.startswith("# ") or line.startswith("## ") or line.startswith("Explanation:"):
                break
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines)

    return raw_text.strip()


def execute_code_sandbox(code: str, test_cases: str, timeout: float = 3.0) -> Tuple[bool, str]:
    """Execute Python code with test assertions inside an isolated subprocess."""
    full_script = (
        "import sys\n"
        "import math\n"
        "import collections\n"
        "import itertools\n"
        "import functools\n"
        "import heapq\n\n"
        f"{code}\n\n"
        "# --- Unit Tests ---\n"
        f"{test_cases}\n\n"
        "sys.exit(0)\n"
    )

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(full_script)
        temp_path = f.name

    try:
        proc = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode == 0:
            return True, "PASSED"
        else:
            err_msg = proc.stderr.strip() or proc.stdout.strip()
            return False, f"Failed with exit code {proc.returncode}: {err_msg[-300:]}"
    except subprocess.TimeoutExpired:
        return False, f"TimeoutExpired after {timeout} seconds"
    except Exception as e:
        return False, f"Execution exception: {e}"
    finally:
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass


def evaluate_coding_benchmark(
    model: Any,
    tokenizer: Any,
    tasks: Optional[List[CodingTask]] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Run model on coding benchmark tasks, execute code in sandbox, and compute pass@1."""
    if tasks is None:
        tasks = CODING_BENCHMARK_TASKS

    logger.info("Executing Coding Benchmark on %d tasks...", len(tasks))
    task_results = []
    passed_count = 0

    for idx, t in enumerate(tasks, start=1):
        prompt_text = (
            f"Complete the following Python function. Provide only clean, working Python code.\n\n"
            f"```python\n{t.prompt}```"
        )
        completion = generate_response(
            model=model,
            tokenizer=tokenizer,
            messages_or_prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
        )

        extracted_code = clean_code_completion(completion, t.entry_point)
        # Prepend prompt if the model only gave body without signature
        if f"def {t.entry_point}" not in extracted_code:
            extracted_code = t.prompt + "\n" + extracted_code

        passed, msg = execute_code_sandbox(extracted_code, t.test_cases)
        if passed:
            passed_count += 1

        logger.info(
            "[%d/%d] Task %s (%s): %s",
            idx,
            len(tasks),
            t.task_id,
            t.entry_point,
            "PASS" if passed else "FAIL",
        )

        task_results.append({
            "task_id": t.task_id,
            "entry_point": t.entry_point,
            "passed": passed,
            "error_message": msg if not passed else None,
            "generated_code": extracted_code,
        })

    pass_at_1 = (passed_count / len(tasks)) * 100.0 if tasks else 0.0

    summary = {
        "num_tasks": len(tasks),
        "passed_count": passed_count,
        "failed_count": len(tasks) - passed_count,
        "pass_at_1": round(pass_at_1, 2),
        "tasks": task_results,
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        logger.info("Saved coding benchmark results to '%s'", out_p)

    return summary
