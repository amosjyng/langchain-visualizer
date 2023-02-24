from typing import List, Optional

from ice.trace import add_fields
from langchain.llms.base import BaseLLM
from langchain.llms.openai import OpenAI
from langchain.schema import LLMResult

from langchain_visualizer.hijacking import VisualizationWrapper, ice_hijack


class LlmVisualizer(VisualizationWrapper):
    async def run(
        self, prompts: List[str], stop: Optional[List[str]] = None
    ) -> LLMResult:
        """Run the LLM on the given prompt and input."""
        result = self.og_fn(self.og_obj, prompts=prompts, stop=stop)
        if (
            isinstance(self.og_obj, OpenAI)
            and self.og_obj.model_name == "text-davinci-003"
        ):
            total_tokens = result.llm_output.get("token_usage", {}).get(
                "total_tokens", 0
            )
            add_fields(davinci_equivalent_tokens=total_tokens)
        return result


ice_hijack(BaseLLM, "generate", LlmVisualizer)
