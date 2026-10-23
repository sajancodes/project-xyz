"""Standardized 60-Task Execution-Based Python Coding Benchmark.

Covers algorithmic problem solving, data structures, parsing, math, and string processing.
Evaluated via isolated execution against hidden unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class CodingTask:
    task_id: str
    entry_point: str
    prompt: str
    test_cases: str
    difficulty: str = "medium"


CODING_BENCHMARK_TASKS: List[CodingTask] = [
    # 1. Array & Two Pointers
    CodingTask(
        task_id="coding_task_001",
        entry_point="two_sum",
        prompt="def two_sum(nums: list[int], target: int) -> list[int]:\n    \"\"\"Given a list of integers and a target, return indices of the two numbers such that they add up to target.\n    Assume exactly one solution exists. Indices may be returned in any order.\n    >>> two_sum([2, 7, 11, 15], 9)\n    [0, 1]\n    \"\"\"\n",
        test_cases="assert sorted(two_sum([2, 7, 11, 15], 9)) == [0, 1]\nassert sorted(two_sum([3, 2, 4], 6)) == [1, 2]\nassert sorted(two_sum([3, 3], 6)) == [0, 1]\n",
    ),
    CodingTask(
        task_id="coding_task_002",
        entry_point="max_subarray_sum",
        prompt="def max_subarray_sum(nums: list[int]) -> int:\n    \"\"\"Find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.\n    >>> max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4])\n    6\n    \"\"\"\n",
        test_cases="assert max_subarray_sum([-2, 1, -3, 4, -1, 2, 1, -5, 4]) == 6\nassert max_subarray_sum([1]) == 1\nassert max_subarray_sum([5, 4, -1, 7, 8]) == 23\nassert max_subarray_sum([-1, -2, -3]) == -1\n",
    ),
    CodingTask(
        task_id="coding_task_003",
        entry_point="has_close_elements",
        prompt="def has_close_elements(numbers: list[float], threshold: float) -> bool:\n    \"\"\"Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0], 0.2)\n    True\n    >>> has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0], 0.05)\n    False\n    \"\"\"\n",
        test_cases="assert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0], 0.2) == True\nassert has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0], 0.05) == False\nassert has_close_elements([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\nassert has_close_elements([1.0, 2.0, 3.0, 4.0], 0.5) == False\n",
    ),
    CodingTask(
        task_id="coding_task_004",
        entry_point="longest_common_prefix",
        prompt="def longest_common_prefix(strs: list[str]) -> str:\n    \"\"\"Find the longest common prefix string amongst an array of strings. Return empty string if no common prefix.\n    >>> longest_common_prefix(['flower', 'flow', 'flight'])\n    'fl'\n    \"\"\"\n",
        test_cases="assert longest_common_prefix(['flower', 'flow', 'flight']) == 'fl'\nassert longest_common_prefix(['dog', 'racecar', 'car']) == ''\nassert longest_common_prefix(['interspecies', 'interstellar', 'interstate']) == 'inters'\nassert longest_common_prefix(['throne']) == 'throne'\nassert longest_common_prefix([]) == ''\n",
    ),
    CodingTask(
        task_id="coding_task_005",
        entry_point="is_valid_parentheses",
        prompt="def is_valid_parentheses(s: str) -> bool:\n    \"\"\"Determine if the input string containing '()', '[]', '{}' has valid and balanced bracket ordering.\n    >>> is_valid_parentheses('()[]{}')\n    True\n    >>> is_valid_parentheses('(]')\n    False\n    \"\"\"\n",
        test_cases="assert is_valid_parentheses('()') == True\nassert is_valid_parentheses('()[]{}') == True\nassert is_valid_parentheses('(]') == False\nassert is_valid_parentheses('([)]') == False\nassert is_valid_parentheses('{[]}') == True\nassert is_valid_parentheses('') == True\nassert is_valid_parentheses('((') == False\n",
    ),
    CodingTask(
        task_id="coding_task_006",
        entry_point="merge_intervals",
        prompt="def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    \"\"\"Given an array of intervals where intervals[i] = [start_i, end_i], merge all overlapping intervals.\n    Return an array of the non-overlapping intervals in ascending order.\n    >>> merge_intervals([[1,3],[2,6],[8,10],[15,18]])\n    [[1,6],[8,10],[15,18]]\n    \"\"\"\n",
        test_cases="assert merge_intervals([[1,3],[2,6],[8,10],[15,18]]) == [[1,6],[8,10],[15,18]]\nassert merge_intervals([[1,4],[4,5]]) == [[1,5]]\nassert merge_intervals([[1,4],[0,4]]) == [[0,4]]\nassert merge_intervals([]) == []\n",
    ),
    CodingTask(
        task_id="coding_task_007",
        entry_point="is_palindrome",
        prompt="def is_palindrome(s: str) -> bool:\n    \"\"\"Given a string s, return True if it is a palindrome considering only alphanumeric characters and ignoring cases.\n    >>> is_palindrome('A man, a plan, a canal: Panama')\n    True\n    \"\"\"\n",
        test_cases="assert is_palindrome('A man, a plan, a canal: Panama') == True\nassert is_palindrome('race a car') == False\nassert is_palindrome(' ') == True\nassert is_palindrome('0P') == False\n",
    ),
    CodingTask(
        task_id="coding_task_008",
        entry_point="product_except_self",
        prompt="def product_except_self(nums: list[int]) -> list[int]:\n    \"\"\"Return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].\n    Do not use division, solve in O(n) time.\n    >>> product_except_self([1, 2, 3, 4])\n    [24, 12, 8, 6]\n    \"\"\"\n",
        test_cases="assert product_except_self([1, 2, 3, 4]) == [24, 12, 8, 6]\nassert product_except_self([-1, 1, 0, -3, 3]) == [0, 0, 9, 0, 0]\n",
    ),
    CodingTask(
        task_id="coding_task_009",
        entry_point="reverse_words",
        prompt="def reverse_words(s: str) -> str:\n    \"\"\"Given an input string s, reverse the order of the words. Return a string of words joined by a single space.\n    >>> reverse_words('the sky is blue')\n    'blue is sky the'\n    \"\"\"\n",
        test_cases="assert reverse_words('the sky is blue') == 'blue is sky the'\nassert reverse_words('  hello world  ') == 'world hello'\nassert reverse_words('a good   example') == 'example good a'\n",
    ),
    CodingTask(
        task_id="coding_task_010",
        entry_point="climb_stairs",
        prompt="def climb_stairs(n: int) -> int:\n    \"\"\"You are climbing a staircase. It takes n steps to reach the top. Each time you can climb 1 or 2 steps.\n    In how many distinct ways can you climb to the top?\n    >>> climb_stairs(3)\n    3\n    \"\"\"\n",
        test_cases="assert climb_stairs(1) == 1\nassert climb_stairs(2) == 2\nassert climb_stairs(3) == 3\nassert climb_stairs(5) == 8\nassert climb_stairs(10) == 89\n",
    ),
    CodingTask(
        task_id="coding_task_011",
        entry_point="group_anagrams",
        prompt="def group_anagrams(strs: list[str]) -> list[list[str]]:\n    \"\"\"Given an array of strings, group the anagrams together. You can return the answer in any order.\n    >>> group_anagrams(['eat','tea','tan','ate','nat','bat'])\n    \"\"\"\n",
        test_cases="res = [sorted(g) for g in group_anagrams(['eat','tea','tan','ate','nat','bat'])]\nexpected = [sorted(g) for g in [['bat'],['nat','tan'],['ate','eat','tea']]]\nassert sorted(res) == sorted(expected)\n",
    ),
    CodingTask(
        task_id="coding_task_012",
        entry_point="coin_change",
        prompt="def coin_change(coins: list[int], amount: int) -> int:\n    \"\"\"Return the fewest number of coins needed to make up given amount. If cannot be made up, return -1.\n    >>> coin_change([1, 2, 5], 11)\n    3\n    \"\"\"\n",
        test_cases="assert coin_change([1, 2, 5], 11) == 3\nassert coin_change([2], 3) == -1\nassert coin_change([1], 0) == 0\nassert coin_change([186, 419, 83, 408], 6249) == 20\n",
    ),
    CodingTask(
        task_id="coding_task_013",
        entry_point="find_min_rotated_sorted_array",
        prompt="def find_min_rotated_sorted_array(nums: list[int]) -> int:\n    \"\"\"Find minimum element in sorted rotated array of unique elements in O(log n) time.\n    >>> find_min_rotated_sorted_array([3, 4, 5, 1, 2])\n    1\n    \"\"\"\n",
        test_cases="assert find_min_rotated_sorted_array([3, 4, 5, 1, 2]) == 1\nassert find_min_rotated_sorted_array([4, 5, 6, 7, 0, 1, 2]) == 0\nassert find_min_rotated_sorted_array([11, 13, 15, 17]) == 11\nassert find_min_rotated_sorted_array([2, 1]) == 1\n",
    ),
    CodingTask(
        task_id="coding_task_014",
        entry_point="length_of_longest_substring",
        prompt="def length_of_longest_substring(s: str) -> int:\n    \"\"\"Given a string s, find the length of the longest substring without repeating characters.\n    >>> length_of_longest_substring('abcabcbb')\n    3\n    \"\"\"\n",
        test_cases="assert length_of_longest_substring('abcabcbb') == 3\nassert length_of_longest_substring('bbbbb') == 1\nassert length_of_longest_substring('pwwkew') == 3\nassert length_of_longest_substring('') == 0\nassert length_of_longest_substring('dvdf') == 3\n",
    ),
    CodingTask(
        task_id="coding_task_015",
        entry_point="is_prime",
        prompt="def is_prime(n: int) -> bool:\n    \"\"\"Return True if n is a prime number, False otherwise.\n    >>> is_prime(17)\n    True\n    >>> is_prime(1)\n    False\n    \"\"\"\n",
        test_cases="assert is_prime(1) == False\nassert is_prime(2) == True\nassert is_prime(3) == True\nassert is_prime(4) == False\nassert is_prime(17) == True\nassert is_prime(97) == True\nassert is_prime(100) == False\n",
    ),
    CodingTask(
        task_id="coding_task_016",
        entry_point="count_primes",
        prompt="def count_primes(n: int) -> int:\n    \"\"\"Given an integer n, return the number of prime numbers strictly less than n (using Sieve of Eratosthenes).\n    >>> count_primes(10)\n    4\n    \"\"\"\n",
        test_cases="assert count_primes(10) == 4\nassert count_primes(0) == 0\nassert count_primes(1) == 0\nassert count_primes(2) == 0\nassert count_primes(20) == 8\n",
    ),
    CodingTask(
        task_id="coding_task_017",
        entry_point="rotate_matrix",
        prompt="def rotate_matrix(matrix: list[list[int]]) -> list[list[int]]:\n    \"\"\"Rotate an n x n 2D matrix representing an image by 90 degrees clockwise in-place or returning new matrix.\n    >>> rotate_matrix([[1, 2], [3, 4]])\n    [[3, 1], [4, 2]]\n    \"\"\"\n",
        test_cases="assert rotate_matrix([[1, 2], [3, 4]]) == [[3, 1], [4, 2]]\nassert rotate_matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [[7, 4, 1], [8, 5, 2], [9, 6, 3]]\n",
    ),
    CodingTask(
        task_id="coding_task_018",
        entry_point="spiral_order",
        prompt="def spiral_order(matrix: list[list[int]]) -> list[int]:\n    \"\"\"Given an m x n matrix, return all elements of the matrix in spiral order.\n    >>> spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]])\n    [1, 2, 3, 6, 9, 8, 7, 4, 5]\n    \"\"\"\n",
        test_cases="assert spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) == [1, 2, 3, 6, 9, 8, 7, 4, 5]\nassert spiral_order([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]) == [1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]\nassert spiral_order([]) == []\n",
    ),
    CodingTask(
        task_id="coding_task_019",
        entry_point="binary_search",
        prompt="def binary_search(nums: list[int], target: int) -> int:\n    \"\"\"Search target in ascending sorted array nums. Return index if found, else -1.\n    >>> binary_search([-1, 0, 3, 5, 9, 12], 9)\n    4\n    \"\"\"\n",
        test_cases="assert binary_search([-1, 0, 3, 5, 9, 12], 9) == 4\nassert binary_search([-1, 0, 3, 5, 9, 12], 2) == -1\nassert binary_search([5], 5) == 0\nassert binary_search([], 5) == -1\n",
    ),
    CodingTask(
        task_id="coding_task_020",
        entry_point="find_peak_element",
        prompt="def find_peak_element(nums: list[int]) -> int:\n    \"\"\"Find a peak element (strictly greater than neighbors) and return its index in O(log n) time.\n    >>> find_peak_element([1, 2, 3, 1])\n    2\n    \"\"\"\n",
        test_cases="idx = find_peak_element([1, 2, 3, 1])\nassert idx == 2\nidx2 = find_peak_element([1, 2, 1, 3, 5, 6, 4])\nassert idx2 in [1, 5]\n",
    ),
    CodingTask(
        task_id="coding_task_021",
        entry_point="flatten_nested_list",
        prompt="def flatten_nested_list(nested: list) -> list:\n    \"\"\"Flatten arbitrarily nested list into a single 1D list.\n    >>> flatten_nested_list([1, [2, [3, 4], 5], 6])\n    [1, 2, 3, 4, 5, 6]\n    \"\"\"\n",
        test_cases="assert flatten_nested_list([1, [2, [3, 4], 5], 6]) == [1, 2, 3, 4, 5, 6]\nassert flatten_nested_list([]) == []\nassert flatten_nested_list([[[]]]) == []\nassert flatten_nested_list([1, 2, 3]) == [1, 2, 3]\n",
    ),
    CodingTask(
        task_id="coding_task_022",
        entry_point="fibonacci",
        prompt="def fibonacci(n: int) -> int:\n    \"\"\"Return n-th Fibonacci number where fib(0)=0, fib(1)=1.\n    >>> fibonacci(10)\n    55\n    \"\"\"\n",
        test_cases="assert fibonacci(0) == 0\nassert fibonacci(1) == 1\nassert fibonacci(2) == 1\nassert fibonacci(7) == 13\nassert fibonacci(10) == 55\nassert fibonacci(20) == 6765\n",
    ),
    CodingTask(
        task_id="coding_task_023",
        entry_point="gcd",
        prompt="def gcd(a: int, b: int) -> int:\n    \"\"\"Compute greatest common divisor of a and b using Euclidean algorithm.\n    >>> gcd(48, 18)\n    6\n    \"\"\"\n",
        test_cases="assert gcd(48, 18) == 6\nassert gcd(101, 103) == 1\nassert gcd(54, 24) == 6\nassert gcd(0, 5) == 5\n",
    ),
    CodingTask(
        task_id="coding_task_024",
        entry_point="lcm",
        prompt="def lcm(a: int, b: int) -> int:\n    \"\"\"Compute least common multiple of positive integers a and b.\n    >>> lcm(4, 6)\n    12\n    \"\"\"\n",
        test_cases="assert lcm(4, 6) == 12\nassert lcm(21, 6) == 42\nassert lcm(1, 99) == 99\n",
    ),
    CodingTask(
        task_id="coding_task_025",
        entry_point="roman_to_int",
        prompt="def roman_to_int(s: str) -> int:\n    \"\"\"Convert Roman numeral string to integer (I=1, V=5, X=10, L=50, C=100, D=500, M=1000).\n    >>> roman_to_int('MCMXCIV')\n    1994\n    \"\"\"\n",
        test_cases="assert roman_to_int('III') == 3\nassert roman_to_int('LVIII') == 58\nassert roman_to_int('MCMXCIV') == 1994\n",
    ),
    CodingTask(
        task_id="coding_task_026",
        entry_point="int_to_roman",
        prompt="def int_to_roman(num: int) -> str:\n    \"\"\"Convert integer (1 <= num <= 3999) to Roman numeral string.\n    >>> int_to_roman(1994)\n    'MCMXCIV'\n    \"\"\"\n",
        test_cases="assert int_to_roman(3) == 'III'\nassert int_to_roman(58) == 'LVIII'\nassert int_to_roman(1994) == 'MCMXCIV'\n",
    ),
    CodingTask(
        task_id="coding_task_027",
        entry_point="valid_sudoku",
        prompt="def valid_sudoku(board: list[list[str]]) -> bool:\n    \"\"\"Determine if a 9 x 9 Sudoku board is valid according to rules ('.' denotes empty cell).\n    \"\"\"\n",
        test_cases="board = [['5','3','.','.','7','.','.','.','.'],['6','.','.','1','9','5','.','.','.'],['.','9','8','.','.','.','.','6','.'],['8','.','.','.','6','.','.','.','3'],['4','.','.','8','.','3','.','.','1'],['7','.','.','.','2','.','.','.','6'],['.','6','.','.','.','.','2','8','.'],['.','.','.','4','1','9','.','.','5'],['.','.','.','.','8','.','.','7','9']]\nassert valid_sudoku(board) == True\nboard[0][0] = '8'\nassert valid_sudoku(board) == False\n",
    ),
    CodingTask(
        task_id="coding_task_028",
        entry_point="edit_distance",
        prompt="def edit_distance(word1: str, word2: str) -> int:\n    \"\"\"Given two strings word1 and word2, return the minimum number of operations (insert, delete, replace) required to convert word1 to word2.\n    >>> edit_distance('horse', 'ros')\n    3\n    \"\"\"\n",
        test_cases="assert edit_distance('horse', 'ros') == 3\nassert edit_distance('intention', 'execution') == 5\nassert edit_distance('', '') == 0\nassert edit_distance('a', '') == 1\n",
    ),
    CodingTask(
        task_id="coding_task_029",
        entry_point="longest_palindromic_substring",
        prompt="def longest_palindromic_substring(s: str) -> str:\n    \"\"\"Given string s, return the longest palindromic substring in s.\n    >>> longest_palindromic_substring('babad') in ['bab', 'aba']\n    True\n    \"\"\"\n",
        test_cases="assert longest_palindromic_substring('babad') in ['bab', 'aba']\nassert longest_palindromic_substring('cbbd') == 'bb'\nassert longest_palindromic_substring('a') == 'a'\n",
    ),
    CodingTask(
        task_id="coding_task_030",
        entry_point="trapping_rain_water",
        prompt="def trapping_rain_water(height: list[int]) -> int:\n    \"\"\"Compute how much water it can trap after raining given elevation map height.\n    >>> trapping_rain_water([0,1,0,2,1,0,1,3,2,1,2,1])\n    6\n    \"\"\"\n",
        test_cases="assert trapping_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]) == 6\nassert trapping_rain_water([4,2,0,3,2,5]) == 9\nassert trapping_rain_water([]) == 0\n",
    ),
    # 31-60: Further algorithmic & data structure problems
    CodingTask(
        task_id="coding_task_031",
        entry_point="run_length_encode",
        prompt="def run_length_encode(s: str) -> str:\n    \"\"\"Perform run-length encoding on string (e.g., 'aaabbc' -> 'a3b2c1').\n    >>> run_length_encode('aaabbc')\n    'a3b2c1'\n    \"\"\"\n",
        test_cases="assert run_length_encode('aaabbc') == 'a3b2c1'\nassert run_length_encode('') == ''\nassert run_length_encode('a') == 'a1'\nassert run_length_encode('abcd') == 'a1b1c1d1'\n",
    ),
    CodingTask(
        task_id="coding_task_032",
        entry_point="run_length_decode",
        prompt="def run_length_decode(s: str) -> str:\n    \"\"\"Decode run-length encoded string (e.g., 'a3b2c1' -> 'aaabbc').\n    >>> run_length_decode('a3b2c1')\n    'aaabbc'\n    \"\"\"\n",
        test_cases="assert run_length_decode('a3b2c1') == 'aaabbc'\nassert run_length_decode('') == ''\nassert run_length_decode('a10') == 'aaaaaaaaaa'\n",
    ),
    CodingTask(
        task_id="coding_task_033",
        entry_point="pascal_triangle",
        prompt="def pascal_triangle(num_rows: int) -> list[list[int]]:\n    \"\"\"Generate the first num_rows of Pascal's triangle.\n    >>> pascal_triangle(5)\n    [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]\n    \"\"\"\n",
        test_cases="assert pascal_triangle(1) == [[1]]\nassert pascal_triangle(5) == [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]\n",
    ),
    CodingTask(
        task_id="coding_task_034",
        entry_point="majority_element",
        prompt="def majority_element(nums: list[int]) -> int:\n    \"\"\"Return the majority element that appears more than floor(n/2) times using Boyer-Moore Voting algorithm.\n    >>> majority_element([3, 2, 3])\n    3\n    \"\"\"\n",
        test_cases="assert majority_element([3, 2, 3]) == 3\nassert majority_element([2, 2, 1, 1, 1, 2, 2]) == 2\n",
    ),
    CodingTask(
        task_id="coding_task_035",
        entry_point="single_number",
        prompt="def single_number(nums: list[int]) -> int:\n    \"\"\"Every element appears twice except for one. Find that single one in O(n) time and O(1) space.\n    >>> single_number([4, 1, 2, 1, 2])\n    4\n    \"\"\"\n",
        test_cases="assert single_number([2, 2, 1]) == 1\nassert single_number([4, 1, 2, 1, 2]) == 4\nassert single_number([1]) == 1\n",
    ),
    CodingTask(
        task_id="coding_task_036",
        entry_point="word_break",
        prompt="def word_break(s: str, word_dict: list[str]) -> bool:\n    \"\"\"Return True if s can be segmented into a space-separated sequence of one or more dictionary words.\n    >>> word_break('leetcode', ['leet', 'code'])\n    True\n    \"\"\"\n",
        test_cases="assert word_break('leetcode', ['leet', 'code']) == True\nassert word_break('applepenapple', ['apple', 'pen']) == True\nassert word_break('catsandog', ['cats', 'dog', 'sand', 'and', 'cat']) == False\n",
    ),
    CodingTask(
        task_id="coding_task_037",
        entry_point="num_islands",
        prompt="def num_islands(grid: list[list[str]]) -> int:\n    \"\"\"Given an m x n 2D binary grid representing map of '1's (land) and '0's (water), count number of islands.\n    \"\"\"\n",
        test_cases="g1 = [['1','1','1','1','0'],['1','1','0','1','0'],['1','1','0','0','0'],['0','0','0','0','0']]\nassert num_islands(g1) == 1\ng2 = [['1','1','0','0','0'],['1','1','0','0','0'],['0','0','1','0','0'],['0','0','0','1','1']]\nassert num_islands(g2) == 3\n",
    ),
    CodingTask(
        task_id="coding_task_038",
        entry_point="top_k_frequent",
        prompt="def top_k_frequent(nums: list[int], k: int) -> list[int]:\n    \"\"\"Given an integer array nums and an integer k, return the k most frequent elements in any order.\n    >>> sorted(top_k_frequent([1,1,1,2,2,3], 2))\n    [1, 2]\n    \"\"\"\n",
        test_cases="assert sorted(top_k_frequent([1,1,1,2,2,3], 2)) == [1, 2]\nassert top_k_frequent([1], 1) == [1]\n",
    ),
    CodingTask(
        task_id="coding_task_039",
        entry_point="subsets",
        prompt="def subsets(nums: list[int]) -> list[list[int]]:\n    \"\"\"Given an integer array nums of unique elements, return all possible subsets (the power set).\n    \"\"\"\n",
        test_cases="res = [sorted(s) for s in subsets([1, 2, 3])]\nassert len(res) == 8\nassert [] in res\nassert [1, 2, 3] in res\n",
    ),
    CodingTask(
        task_id="coding_task_040",
        entry_point="permute",
        prompt="def permute(nums: list[int]) -> list[list[int]]:\n    \"\"\"Given an array nums of distinct integers, return all the possible permutations in any order.\n    \"\"\"\n",
        test_cases="res = permute([1, 2, 3])\nassert len(res) == 6\nassert [1, 2, 3] in res\nassert [3, 2, 1] in res\n",
    ),
    CodingTask(
        task_id="coding_task_041",
        entry_point="can_jump",
        prompt="def can_jump(nums: list[int]) -> bool:\n    \"\"\"Given integer array nums where nums[i] is max jump length at position i, determine if you can reach the last index.\n    >>> can_jump([2,3,1,1,4])\n    True\n    >>> can_jump([3,2,1,0,4])\n    False\n    \"\"\"\n",
        test_cases="assert can_jump([2,3,1,1,4]) == True\nassert can_jump([3,2,1,0,4]) == False\nassert can_jump([0]) == True\n",
    ),
    CodingTask(
        task_id="coding_task_042",
        entry_point="longest_increasing_subsequence",
        prompt="def longest_increasing_subsequence(nums: list[int]) -> int:\n    \"\"\"Given integer array nums, return the length of the longest strictly increasing subsequence.\n    >>> longest_increasing_subsequence([10,9,2,5,3,7,101,18])\n    4\n    \"\"\"\n",
        test_cases="assert longest_increasing_subsequence([10,9,2,5,3,7,101,18]) == 4\nassert longest_increasing_subsequence([0,1,0,3,2,3]) == 4\nassert longest_increasing_subsequence([7,7,7,7]) == 1\n",
    ),
    CodingTask(
        task_id="coding_task_043",
        entry_point="find_median_sorted_arrays",
        prompt="def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:\n    \"\"\"Given two sorted arrays nums1 and nums2, return the median of the two sorted arrays.\n    >>> find_median_sorted_arrays([1, 3], [2])\n    2.0\n    \"\"\"\n",
        test_cases="assert find_median_sorted_arrays([1, 3], [2]) == 2.0\nassert find_median_sorted_arrays([1, 2], [3, 4]) == 2.5\nassert find_median_sorted_arrays([], [1]) == 1.0\n",
    ),
    CodingTask(
        task_id="coding_task_044",
        entry_point="max_profit",
        prompt="def max_profit(prices: list[int]) -> int:\n    \"\"\"Maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.\n    >>> max_profit([7,1,5,3,6,4])\n    5\n    \"\"\"\n",
        test_cases="assert max_profit([7,1,5,3,6,4]) == 5\nassert max_profit([7,6,4,3,1]) == 0\nassert max_profit([]) == 0\n",
    ),
    CodingTask(
        task_id="coding_task_045",
        entry_point="daily_temperatures",
        prompt="def daily_temperatures(temperatures: list[int]) -> list[int]:\n    \"\"\"Return an array answer such that answer[i] is the number of days you have to wait after the i-th day to get a warmer temperature.\n    If there is no future day for which this is possible, keep answer[i] == 0 instead.\n    >>> daily_temperatures([73,74,75,71,69,72,76,73])\n    [1, 1, 4, 2, 1, 1, 0, 0]\n    \"\"\"\n",
        test_cases="assert daily_temperatures([73,74,75,71,69,72,76,73]) == [1, 1, 4, 2, 1, 1, 0, 0]\nassert daily_temperatures([30,40,50,60]) == [1, 1, 1, 0]\nassert daily_temperatures([30,60,90]) == [1, 1, 0]\n",
    ),
    CodingTask(
        task_id="coding_task_046",
        entry_point="min_window_substring",
        prompt="def min_window_substring(s: str, t: str) -> str:\n    \"\"\"Given two strings s and t, return minimum window substring of s such that every character in t (including duplicates) is included.\n    If no such substring, return empty string.\n    >>> min_window_substring('ADOBECODEBANC', 'ABC')\n    'BANC'\n    \"\"\"\n",
        test_cases="assert min_window_substring('ADOBECODEBANC', 'ABC') == 'BANC'\nassert min_window_substring('a', 'a') == 'a'\nassert min_window_substring('a', 'aa') == ''\n",
    ),
    CodingTask(
        task_id="coding_task_047",
        entry_point="kth_largest_element",
        prompt="def kth_largest_element(nums: list[int], k: int) -> int:\n    \"\"\"Given integer array nums and integer k, return the k-th largest element in the array.\n    >>> kth_largest_element([3,2,1,5,6,4], 2)\n    5\n    \"\"\"\n",
        test_cases="assert kth_largest_element([3,2,1,5,6,4], 2) == 5\nassert kth_largest_element([3,2,3,1,2,4,5,5,6], 4) == 4\n",
    ),
    CodingTask(
        task_id="coding_task_048",
        entry_point="trie_prefix_match",
        prompt="def trie_prefix_match(words: list[str], prefix: str) -> list[str]:\n    \"\"\"Return all words from words list that start with given prefix, sorted alphabetically.\n    >>> trie_prefix_match(['apple', 'app', 'application', 'banana'], 'app')\n    ['app', 'apple', 'application']\n    \"\"\"\n",
        test_cases="assert trie_prefix_match(['apple', 'app', 'application', 'banana'], 'app') == ['app', 'apple', 'application']\nassert trie_prefix_match(['cat', 'dog'], 'z') == []\nassert trie_prefix_match([], 'abc') == []\n",
    ),
    CodingTask(
        task_id="coding_task_049",
        entry_point="longest_consecutive",
        prompt="def longest_consecutive(nums: list[int]) -> int:\n    \"\"\"Given an unsorted array of integers nums, return length of longest consecutive elements sequence in O(n) time.\n    >>> longest_consecutive([100, 4, 200, 1, 3, 2])\n    4\n    \"\"\"\n",
        test_cases="assert longest_consecutive([100, 4, 200, 1, 3, 2]) == 4\nassert longest_consecutive([0,3,7,2,5,8,4,6,0,1]) == 9\nassert longest_consecutive([]) == 0\n",
    ),
    CodingTask(
        task_id="coding_task_050",
        entry_point="generate_parentheses",
        prompt="def generate_parentheses(n: int) -> list[str]:\n    \"\"\"Given n pairs of parentheses, generate all combinations of well-formed parentheses.\n    >>> sorted(generate_parentheses(3))\n    ['((()))', '(()())', '(())()', '()(())', '()()()']\n    \"\"\"\n",
        test_cases="assert sorted(generate_parentheses(3)) == ['((()))', '(()())', '(())()', '()(())', '()()()']\nassert generate_parentheses(1) == ['()']\n",
    ),
    CodingTask(
        task_id="coding_task_051",
        entry_point="decode_string",
        prompt="def decode_string(s: str) -> str:\n    \"\"\"Given an encoded string k[encoded_string], return its decoded string.\n    >>> decode_string('3[a]2[bc]')\n    'aaabcbc'\n    >>> decode_string('3[a2[c]]')\n    'accaccacc'\n    \"\"\"\n",
        test_cases="assert decode_string('3[a]2[bc]') == 'aaabcbc'\nassert decode_string('3[a2[c]]') == 'accaccacc'\nassert decode_string('2[abc]3[cd]ef') == 'abcabccdcdcdef'\n",
    ),
    CodingTask(
        task_id="coding_task_052",
        entry_point="partition_labels",
        prompt="def partition_labels(s: str) -> list[int]:\n    \"\"\"Partition string s into as many parts as possible so that each letter appears in at most one part.\n    Return list of integers representing size of these parts.\n    >>> partition_labels('ababcbacadefegdehijhklij')\n    [9, 7, 8]\n    \"\"\"\n",
        test_cases="assert partition_labels('ababcbacadefegdehijhklij') == [9, 7, 8]\nassert partition_labels('eccbbbbdec') == [10]\n",
    ),
    CodingTask(
        task_id="coding_task_053",
        entry_point="find_all_anagrams",
        prompt="def find_all_anagrams(s: str, p: str) -> list[int]:\n    \"\"\"Given two strings s and p, return array of all start indices of p's anagrams in s.\n    >>> find_all_anagrams('cbaebabacd', 'abc')\n    [0, 6]\n    \"\"\"\n",
        test_cases="assert find_all_anagrams('cbaebabacd', 'abc') == [0, 6]\nassert find_all_anagrams('abab', 'ab') == [0, 1, 2]\nassert find_all_anagrams('a', 'aa') == []\n",
    ),
    CodingTask(
        task_id="coding_task_054",
        entry_point="max_product_subarray",
        prompt="def max_product_subarray(nums: list[int]) -> int:\n    \"\"\"Given integer array nums, find a contiguous non-empty subarray within the array that has largest product.\n    >>> max_product_subarray([2,3,-2,4])\n    6\n    \"\"\"\n",
        test_cases="assert max_product_subarray([2,3,-2,4]) == 6\nassert max_product_subarray([-2,0,-1]) == 0\nassert max_product_subarray([-2, 3, -4]) == 24\n",
    ),
    CodingTask(
        task_id="coding_task_055",
        entry_point="search_matrix",
        prompt="def search_matrix(matrix: list[list[int]], target: int) -> bool:\n    \"\"\"Search for target integer in an m x n integer matrix where rows and columns are sorted ascending.\n    \"\"\"\n",
        test_cases="mat = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]\nassert search_matrix(mat, 5) == True\nassert search_matrix(mat, 20) == False\nassert search_matrix([], 1) == False\n",
    ),
    CodingTask(
        task_id="coding_task_056",
        entry_point="next_permutation",
        prompt="def next_permutation(nums: list[int]) -> list[int]:\n    \"\"\"Rearrange numbers into lexicographically next greater permutation of numbers. If not possible, rearrange to lowest possible order.\n    >>> next_permutation([1, 2, 3])\n    [1, 3, 2]\n    \"\"\"\n",
        test_cases="assert next_permutation([1, 2, 3]) == [1, 3, 2]\nassert next_permutation([3, 2, 1]) == [1, 2, 3]\nassert next_permutation([1, 1, 5]) == [1, 5, 1]\n",
    ),
    CodingTask(
        task_id="coding_task_057",
        entry_point="simplify_path",
        prompt="def simplify_path(path: str) -> str:\n    \"\"\"Given string path which is absolute path to a file or directory in Unix-style file system, convert it to simplified canonical path.\n    >>> simplify_path('/home//foo/')\n    '/home/foo'\n    \"\"\"\n",
        test_cases="assert simplify_path('/home/') == '/home'\nassert simplify_path('/../') == '/'\nassert simplify_path('/home//foo/') == '/home/foo'\nassert simplify_path('/a/./b/../../c/') == '/c'\n",
    ),
    CodingTask(
        task_id="coding_task_058",
        entry_point="merge_k_sorted_lists",
        prompt="def merge_k_sorted_lists(lists: list[list[int]]) -> list[int]:\n    \"\"\"Merge k sorted lists of integers into one sorted list.\n    >>> merge_k_sorted_lists([[1,4,5],[1,3,4],[2,6]])\n    [1,1,2,3,4,4,5,6]\n    \"\"\"\n",
        test_cases="assert merge_k_sorted_lists([[1,4,5],[1,3,4],[2,6]]) == [1,1,2,3,4,4,5,6]\nassert merge_k_sorted_lists([]) == []\nassert merge_k_sorted_lists([[]]) == []\n",
    ),
    CodingTask(
        task_id="coding_task_059",
        entry_point="valid_ip_address",
        prompt="def valid_ip_address(query_ip: str) -> str:\n    \"\"\"Given string query_ip, return 'IPv4' if valid IPv4, 'IPv6' if valid IPv6, or 'Neither'.\n    >>> valid_ip_address('172.16.254.1')\n    'IPv4'\n    \"\"\"\n",
        test_cases="assert valid_ip_address('172.16.254.1') == 'IPv4'\nassert valid_ip_address('2001:0db8:85a3:0:0:8A2E:0370:7334') == 'IPv6'\nassert valid_ip_address('256.256.256.256') == 'Neither'\nassert valid_ip_address('2001:0db8:85a3:0:0:8A2E:0370:7334:') == 'Neither'\nassert valid_ip_address('1e1.4.5.6') == 'Neither'\n",
    ),
    CodingTask(
        task_id="coding_task_060",
        entry_point="eval_rpn",
        prompt="def eval_rpn(tokens: list[str]) -> int:\n    \"\"\"Evaluate value of arithmetic expression in Reverse Polish Notation ('+', '-', '*', '/'). Truncate toward zero for division.\n    >>> eval_rpn(['2', '1', '+', '3', '*'])\n    9\n    \"\"\"\n",
        test_cases="assert eval_rpn(['2', '1', '+', '3', '*']) == 9\nassert eval_rpn(['4', '13', '5', '/', '+']) == 6\nassert eval_rpn(['10', '6', '9', '3', '+', '-11', '*', '/', '*', '17', '+', '5', '+']) == 22\n",
    ),
]
