import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

from tests.sotu import load_sotu

# ========================== Start of langchain example code ==========================
# https://python.langchain.com/en/latest/modules/chains/index_examples/qa_with_sources.html


docsearch = load_sotu()


@vcr.use_cassette()
async def quickstart_demo():
    query = "What did the president say about Justice Breyer"
    docs = docsearch.similarity_search(query)
    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0, max_tokens=500), chain_type="stuff"
    )
    return chain({"input_documents": docs, "question": query}, return_only_outputs=True)


# ================================== Execute example ==================================


def test_quickstart_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(quickstart_demo())
    assert (
        "The president thanked" in result["output_text"]
        or "The president honored" in result["output_text"]
    )


if __name__ == "__main__":
    from langchain_visualizer import visualize, visualize_embeddings

    visualize_embeddings()
    visualize(quickstart_demo)
