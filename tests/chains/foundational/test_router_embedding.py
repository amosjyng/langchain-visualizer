import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.embedding_router import EmbeddingRouterChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from tiktoken_ext.openai_public import cl100k_base

# ========================== Start of langchain example code ==========================
# https://python.langchain.com/docs/modules/chains/foundational/router


@vcr.use_cassette()
async def router_embedding_demo():
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

    names_and_descriptions = [
        ("physics", ["for questions about physics"]),
        ("math", ["for questions about math"]),
    ]
    router_chain = EmbeddingRouterChain.from_names_and_descriptions(
        names_and_descriptions, FAISS, OpenAIEmbeddings(), routing_keys=["input"]
    )

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

# run this before cassette to download blob first
# avoids errors in CI such as:
# No match for the request (<Request (GET) https://.../cl100k_base.tiktoken>) was found
cl100k_base()


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    results = asyncio.get_event_loop().run_until_complete(router_embedding_demo())
    assert len(results) == 3
    assert "Cumulonimbus" in results[-1]


if __name__ == "__main__":
    from langchain_visualizer import visualize

    visualize(router_embedding_demo)
