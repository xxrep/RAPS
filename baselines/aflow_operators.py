"""The operator set AFlow searches over, bound to one backbone: Custom plus the
dataset-kind operators (AnswerGenerate for QA, Programmer for math,
CustomCodeGenerate for code) and the ScEnsemble consistency vote. The Test operator is
omitted: it scores against public test cases this harness does not ship."""
import re

from RAPS.tools.coding.executor_utils import function_with_timeout

from baselines.common import extract_code, get_llm

PROGRAMMER_TRIES = 3
EXEC_TIMEOUT = 30

ANSWER_GENERATE_PROMPT = """\
Think step by step and solve the problem.
1. In the "thought" field, explain your thinking process in detail.
2. In the "answer" field, provide the final answer concisely and clearly. The answer should be a direct response to the question, without including explanations or reasoning.
Your task: """

SC_ENSEMBLE_PROMPT = """\
Given the question described as follows: {problem}
Several solutions have been generated to address the given problem. They are as follows:
{solutions}

Carefully evaluate these solutions and identify the answer that appears most frequently across them. This consistency in answers is crucial for determining the most reliable solution.

In the "thought" field, provide a detailed explanation of your thought process. In the "solution_letter" field, output only the single letter ID (A, B, C, etc.) corresponding to the most consistent solution. Do not include any additional text or explanation in the "solution_letter" field."""

PROGRAMMER_PROMPT = """\
You are a professional Python programmer. Your task is to write complete, self-contained code based on a given mathematical problem and output the answer. The code should include all necessary imports and dependencies, and be ready to run without additional setup or environment configuration.

Problem description: {problem}
Other analysis: {analysis}
{feedback}

Your code should:
1. Implement the calculation steps described in the problem.
2. Define a function named `solve` that performs the calculation and returns the result. The `solve` function should not require any input parameters; instead, it should obtain all necessary inputs from within the function or from globally defined variables.
3. `solve` function return the final calculation result.

Please ensure your code is efficient, well-commented, and follows Python best practices. The output should be limited to basic data types such as strings, integers, and floats. It is prohibited to transmit images or other file formats. The code output is intended for a text-based language model."""

CODE_FILL_SUFFIX = """\

Please write your code solution in Python. Return ONLY the complete, runnable code. \
Do not include any explanations, comments, or example usage in your response. \
Make sure to include a function named '{function_name}' in your solution."""


def _xml_request(*fields: str) -> str:
    """The response-format suffix asking for the given XML fields."""
    spec = "\n".join(f"<{f}></{f}>" for f in fields)
    return ("\n# Response format (must be strictly followed) (do not include any other "
            f"formats except for the given XML format):\n{spec}")


def _xml_field(text: str, name: str) -> str:
    m = re.findall(rf"<{name}>(.*?)</{name}>", text or "", re.DOTALL)
    return m[-1].strip() if m else ""


def _run_solve(code: str) -> str:
    """str() of the generated solve() in an isolated namespace; raises on failure."""
    def _call():
        ns = {}
        exec(code, ns)
        return ns["solve"]()
    return str(function_with_timeout(_call, (), EXEC_TIMEOUT))


class Operators:
    """The operators a workflow may call, bound to a backbone and (for code tasks) to
    the record's entry point. A workflow receives one instance as `ops`."""

    def __init__(self, llm_name: str, entry_point: str = ""):
        self.llm = get_llm(llm_name)
        self.entry_point = entry_point

    def custom(self, input: str, instruction: str = "") -> str:
        return self.llm.gen(instruction + input)

    def answer_generate(self, input: str) -> str:
        out = self.llm.gen(ANSWER_GENERATE_PROMPT + input
                           + _xml_request("thought", "answer"))
        return _xml_field(out, "answer") or out

    def custom_code_generate(self, problem: str, instruction: str = "") -> str:
        out = self.llm.gen(instruction + problem
                           + CODE_FILL_SUFFIX.format(function_name=self.entry_point))
        return extract_code(out)

    def sc_ensemble(self, solutions, problem: str) -> str:
        """One consistency vote over the candidates; returns the chosen original."""
        if not solutions:
            return ""
        body = "\n\n".join(f"Solution {chr(ord('A') + i)}:\n{s}"
                           for i, s in enumerate(solutions))
        out = self.llm.gen(SC_ENSEMBLE_PROMPT.format(problem=problem, solutions=body)
                           + _xml_request("thought", "solution_letter"))
        letter = _xml_field(out, "solution_letter")
        idx = ord(letter.upper()[:1]) - ord('A') if letter else -1
        return solutions[idx] if 0 <= idx < len(solutions) else solutions[0]

    def programmer(self, problem: str, analysis: str = "None") -> str:
        """Write-and-run a solve() function, retrying with the error as feedback."""
        feedback = ""
        for _ in range(PROGRAMMER_TRIES):
            out = self.llm.gen(PROGRAMMER_PROMPT.format(
                problem=problem, analysis=analysis, feedback=feedback))
            code = extract_code(out)
            try:
                return _run_solve(code)
            except Exception as e:
                feedback = (f"\nThe result of the error from the code you wrote in the "
                            f"previous round:\nCode: {code}\n\nStatus: Failed, {e}")
        return ""
