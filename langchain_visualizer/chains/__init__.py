from langchain.chains.llm import LLMChain

from .base import Chain

Chain._should_trace = True  # type: ignore
LLMChain._should_trace = False  # type: ignore
