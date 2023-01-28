import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio
from typing import Dict, List

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.base import Chain
from langchain.llms import OpenAI

from tests import vcr

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/chains/getting_started.html


class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys).union(
            set(self.chain_2.input_keys)
        )
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ["concat_output"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(**inputs)
        output_2 = self.chain_2.run(**inputs)
        return {"concat_output": output_1 + output_2}


llm = OpenAI()

prompt_1 = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = PromptTemplate(
    input_variables=["product"],
    template="What is a good slogan for a company that makes {product}?",
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)
chain = concat_chain


# ================================== Execute example ==================================


@vcr.use_cassette()
async def custom_chain_demo():
    return chain.run("colorful socks")


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(custom_chain_demo())
    assert (
        result.strip()
        == 'Rainbow Toes Co.\n\n"Brighten Up Your Day with Colorful Socks!"'
    )


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(custom_chain_demo)
