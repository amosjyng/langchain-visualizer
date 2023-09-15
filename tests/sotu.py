import os
import pickle
from io import BufferedWriter

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

FAISS_PATH = "tests/resources/sotu_faiss.pkl"


def load_sotu() -> FAISS:
    if os.path.isfile(FAISS_PATH):
        with open(FAISS_PATH, "rb") as f:
            return pickle.load(f)

    loader = TextLoader("tests/resources/state_of_the_union.txt")
    state_of_the_union = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(state_of_the_union)
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(docs, embeddings)
    with open(FAISS_PATH, "wb") as f:  # type: ignore
        assert isinstance(f, BufferedWriter)  # mypy complains otherwise
        pickle.dump(docsearch, f)
    return docsearch
