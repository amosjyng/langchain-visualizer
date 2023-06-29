import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# ========================== Start of langchain example code ==========================
# https://python.langchain.com/docs/modules/chains/foundational/router


@vcr.use_cassette()
async def router_demo():
    physics_template = (
        "You are a very smart physics professor. "
        "You are great at answering questions about physics in a concise and easy to "
        "understand manner. When you don't know the answer to a question you admit "
        "that you don't know."
        "\n\n"
        "Here is a question:\n"
        "{input}"
    )

    math_template = (
        "You are a very good mathematician. "
        "You are great at answering math questions. You are so good because you are "
        "able to break down hard problems into their component parts, answer the "
        "component parts, and then put them together to answer the broader question."
        "\n\n"
        "Here is a question:\n"
        "{input}"
    )

    prompt_infos = [
        {
            "name": "physics",
            "description": "Good for answering questions about physics",
            "prompt_template": physics_template,
        },
        {
            "name": "math",
            "description": "Good for answering math questions",
            "prompt_template": math_template,
        },
    ]

    llm = OpenAI()
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = PromptTemplate(template=prompt_template, input_variables=["input"])
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain
    default_chain = ConversationChain(llm=llm, output_key="text")

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=["input"],
        # note: output parsers should now be attached to the LLMChain and not to the
        # PromptTemplate. But because of this bug:
        # https://github.com/hwchase17/langchain/issues/6819
        # we are just going to ignore the error instead
        output_parser=RouterOutputParser(),
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    chain = MultiPromptChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
    )
    return [
        chain.run("What is black body radiation?"),
        chain.run(
            "What is the first prime number greater than 40 such that one plus the "
            "prime number is divisible by 3"
        ),
        chain.run("What is the name of the type of cloud that rins"),
    ]


# ================================== Execute example ==================================


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    results = asyncio.get_event_loop().run_until_complete(router_demo())
    assert len(results) == 3
    assert results[-1].endswith("heavy rain.")


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(router_demo)
