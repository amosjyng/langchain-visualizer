import langchain_visualizer  # isort:skip  # noqa: F401
import asyncio

import vcr_langchain as vcr
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI

from tests.sotu import load_sotu

# ========================== Start of langchain example code ==========================
# https://langchain.readthedocs.io/en/latest/modules/chains/combine_docs_examples/qa_with_sources.html


@vcr.use_cassette()
async def map_rerank_demo():
    docsearch = load_sotu()
    query = "What did the president say about Justice Breyer"
    docs = docsearch.similarity_search(query)
    chain = load_qa_with_sources_chain(
        OpenAI(temperature=0, max_tokens=500), chain_type="map_rerank"
    )
    return chain({"input_documents": docs, "question": query}, return_only_outputs=True)


# ================================== Execute example ==================================


def test_llm_usage_succeeds():
    """Check that the chain can run normally"""
    result = asyncio.get_event_loop().run_until_complete(map_rerank_demo())
    assert "The president thanked" in result["output_text"]


if __name__ == "__main__":
    from langchain_visualizer import visualize, visualize_embeddings

    visualize_embeddings()
    visualize(map_rerank_demo)
