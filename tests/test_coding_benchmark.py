"""Unit Tests for Execution-Based Coding Benchmark and Sandbox Runner."""

from pathlib import Path
import pytest

from src.evaluation.coding.runner import clean_code_completion, execute_code_sandbox
from src.evaluation.coding.tasks import CODING_BENCHMARK_TASKS


def test_coding_tasks_inventory():
    assert len(CODING_BENCHMARK_TASKS) >= 50, f"Expected at least 50 coding tasks, found {len(CODING_BENCHMARK_TASKS)}"
    for task in CODING_BENCHMARK_TASKS:
        assert task.task_id.startswith("coding_task_")
        assert task.entry_point.strip()
        assert f"def {task.entry_point}" in task.prompt
        assert "assert" in task.test_cases


def test_clean_code_completion():
    raw_markdown = (
        "Here is the function you requested:\n\n"
        "```python\n"
        "def two_sum(nums, target):\n"
        "    lookup = {}\n"
        "    for i, num in enumerate(nums):\n"
        "        if target - num in lookup:\n"
        "            return [lookup[target - num], i]\n"
        "        lookup[num] = i\n"
        "```\n\n"
        "This function runs in O(n) time."
    )
    cleaned = clean_code_completion(raw_markdown, "two_sum")
    assert "def two_sum" in cleaned
    assert "Here is the function" not in cleaned
    assert "This function runs" not in cleaned


def test_execute_code_sandbox_success():
    code = (
        "def has_close_elements(numbers, threshold):\n"
        "    for i in range(len(numbers)):\n"
        "        for j in range(i + 1, len(numbers)):\n"
        "            if abs(numbers[i] - numbers[j]) < threshold:\n"
        "                return True\n"
        "    return False\n"
    )
    test_cases = (
        "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0], 0.2) == True\n"
        "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0], 0.05) == False\n"
    )
    passed, msg = execute_code_sandbox(code, test_cases, timeout=3.0)
    assert passed is True
    assert msg == "PASSED"


def test_execute_code_sandbox_failure():
    bad_code = (
        "def has_close_elements(numbers, threshold):\n"
        "    return False\n"  # Wrong answer
    )
    test_cases = (
        "assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0], 0.2) == True\n"
    )
    passed, msg = execute_code_sandbox(bad_code, test_cases, timeout=3.0)
    assert passed is False
    assert "AssertionError" in msg or "Failed with exit code" in msg
