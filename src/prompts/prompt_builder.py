from abc import ABC, abstractmethod
from typing import List, Sequence, Union, Dict, Any

from src.prompts.prompts_enum import Prompts
from src.texts.docs import Article


class PromptBuilder(ABC):
    def __init__(self, prompt_name):

        self.prompt = Prompts[prompt_name].value

    @abstractmethod
    def build_prompt(self, text_inputs: Sequence[Any]) -> str:
        """
        Main method of the PromptBuilder class.
        :param text_inputs: A list of texts inputs to transform into a prompt. Can be documents, questions, RAG nodes...
        :return: The final prompt to provide the LLM.
        """
        ...

class NewtralizePromptBuilder(PromptBuilder):
    _prompt_name = "NEWTRALIZE"
    def __init__(self):
        super().__init__(self._prompt_name)

    def build_prompt(self, articles: List[Article]) -> str:
        """
        For now the prompt building is a simple concatenation of the prompt and the documents.
        :param documents: The articles to summarize.
        """
        return self.prompt + " ".join(articles)