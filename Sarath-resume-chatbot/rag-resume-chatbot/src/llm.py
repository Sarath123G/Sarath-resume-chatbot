"""Open-source LLM served locally by Ollama."""
from typing import Iterator

import ollama

from . import config


class LLM:
    def __init__(self, model: str = config.LLM_MODEL, host: str = config.OLLAMA_HOST):
        self.model = model
        self.client = ollama.Client(host=host)

    def stream(self, messages: list[dict]) -> Iterator[str]:
        try:
            for part in self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"temperature": 0.2},
            ):
                yield part["message"]["content"]
        except Exception as e:  # connection refused, model missing, etc.
            raise RuntimeError(
                f"Could not reach Ollama model '{self.model}' at {config.OLLAMA_HOST}. "
                f"Is Ollama running and the model pulled? (ollama pull {self.model})\n{e}"
            ) from e

    def complete(self, messages: list[dict]) -> str:
        return "".join(self.stream(messages))
