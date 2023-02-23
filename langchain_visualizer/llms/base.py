import asyncio
from typing import List, Optional

from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult

from langchain_visualizer.hijacking import ice_hijack


def overridden_generate(
    self, prompts: List[str], stop: Optional[List[str]] = None
) -> LLMResult:
    """Preserve langchain's sync generate method"""
    ice_agent = self.get_ice_agent()
    generations = []
    llm_output = {}
    for prompt in prompts:
        llm_result = asyncio.get_event_loop().run_until_complete(
            ice_agent.generate(prompt=prompt, stop=stop)
        )
        generations.extend(llm_result.generations)
        llm_output.update(llm_result.llm_output)
    return LLMResult(generations, llm_output)


ice_hijack(BaseLLM, "generate")
