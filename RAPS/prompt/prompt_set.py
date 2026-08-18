from typing import Dict, Any
from abc import ABC, abstractmethod


class PromptSet(ABC):
    """
    Abstract base class for a set of prompts.
    """
    @staticmethod
    @abstractmethod
    def get_role() -> str:
        """Return the next role name of the set."""

    @staticmethod
    @abstractmethod
    def get_constraint() -> str:
        """Return the prompt constraint of a role."""

    @staticmethod
    @abstractmethod
    def get_format() -> str:
        """Return the answer-format tag of the set."""

    @staticmethod
    @abstractmethod
    def get_answer_prompt(question) -> str:
        """Build the answering prompt for a question."""

    @staticmethod
    @abstractmethod
    def get_adversarial_answer_prompt(question) -> str:
        """Build the adversarial answering prompt for a question."""

    @staticmethod
    @abstractmethod
    def get_query_prompt(question) -> str:
        """Build the information-gathering query prompt."""

    @staticmethod
    @abstractmethod
    def get_file_analysis_prompt(query, file) -> str:
        """Build the file-analysis prompt for a query over a file."""

    @staticmethod
    @abstractmethod
    def get_websearch_prompt(query) -> str:
        """Build the web-search prompt for a query."""

    @staticmethod
    @abstractmethod
    def get_distill_websearch_prompt(query, results) -> str:
        """Build the prompt that distills web-search results."""

    @staticmethod
    @abstractmethod
    def get_reflect_prompt(question, answer) -> str:
        """Build the reflection prompt over a question and an answer."""

    @staticmethod
    def get_react_prompt(question, solutions, feedback) -> str:
        """Build the feedback-driven rewrite prompt."""

    @staticmethod
    @abstractmethod
    def get_decision_constraint() ->str:
        """Return the output constraint of the final-decision role."""

    @staticmethod
    @abstractmethod
    def get_decision_role() ->str:
        """Return the description of the final-decision role."""

    @staticmethod
    @abstractmethod
    def get_decision_few_shot() ->str:
        """Return the few-shot example of the final-decision role."""
