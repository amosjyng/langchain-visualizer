from langchain.embeddings.openai import OpenAIEmbeddings

from langchain_visualizer.hijacking import ice_hijack


def visualize_embeddings():
    ice_hijack(OpenAIEmbeddings, "_embedding_func")
    ice_hijack(OpenAIEmbeddings, "embed_documents")
