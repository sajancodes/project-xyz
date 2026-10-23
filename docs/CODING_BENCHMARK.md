# Execution-Based Coding Benchmark (60 Tasks)

## 1. Motivation: Why Execution Over Keyword Matching

A 2-question keyword benchmark cannot measure programming ability. A model can parrot keywords like `def binary_search` or `while low <= high` while generating syntactically broken code or off-by-one errors.

In **Experiment V2**, the primary judge of coding quality is **isolated subprocess execution against unit tests**.

```
Model Prompt (Function Signature + Docstring)
                      ↓
Model Generates Python Implementation
                      ↓
Extraction & Sanitization (`clean_code_completion`)
                      ↓
Subprocess Sandbox Execution with Hidden Unit Tests (3.0s timeout)
                      ↓
Assertion Verifications (Pass / Fail)
                      ↓
Metric: Pass@1 Accuracy (% of tasks completely passing all tests)
```

---

## 2. Benchmark Suite Composition (60 Tasks)

The benchmark suite (`src/evaluation/coding/tasks.py`) contains **60 independent algorithmic problems** spanning:

| Category | Tasks | Example Topics |
|---|---|---|
| **Arrays & Two Pointers** | Tasks 1–10 | Two sum, max subarray (Kadane), close elements, longest common prefix, valid parentheses, merge intervals, product except self, reverse words |
| **Dynamic Programming & Search** | Tasks 11–20 | Climbing stairs, group anagrams, coin change, rotated array minimum, longest substring without repeats, prime count sieve, spiral order, binary search |
| **Math & Number Theory** | Tasks 21–30 | Fibonacci, GCD, LCM, Roman numeral conversions, valid Sudoku, edit distance (Levenshtein), longest palindromic substring, trapping rain water |
| **Strings & Compression** | Tasks 31–40 | Run-length encoding/decoding, Pascal triangle, Boyer-Moore majority element, single number XOR, word break, number of islands (BFS/DFS), top K frequent elements |
| **Advanced Data Structures** | Tasks 41–50 | Jump game, longest increasing subsequence, median of two sorted arrays, best time to buy/sell stock, daily temperatures, min window substring, K-th largest, prefix trie, longest consecutive sequence |
| **Parsing & Stack Algorithms** | Tasks 51–60 | Generate parentheses, string decoding (`k[str]`), partition labels, find all anagrams, max product subarray, search 2D matrix II, next permutation, simplify path, merge K sorted lists, valid IP address, Reverse Polish Notation evaluator |

---

## 3. Sandboxed Subprocess Runner

Code execution (`src/evaluation/coding/runner.py`) runs in a temporary isolated environment:
- **Subprocess Isolation**: Executed using `sys.executable -c` in a separate OS process.
- **Strict Timeout**: Enforces a 3.0-second timeout per task to catch infinite loops.
- **Fail Safety**: Captures `SyntaxError`, `AssertionError`, `IndexError`, and runtime exceptions with exact tracebacks.
- **Zero Contamination**: Benchmark tasks are strictly held-out and decontaminated from the training stream using `BenchmarkDecontaminator`.
