import os
import pickle
from io import BufferedWriter

import vcr_langchain as vcr
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

FAISS_PATH = "tests/resources/sotu_faiss.pkl"


@vcr.use_cassette
def load_sotu() -> FAISS:
    if os.path.isfile(FAISS_PATH):
        with open(FAISS_PATH, "rb") as f:
            return pickle.load(f)

    with open("tests/resources/state_of_the_union.txt") as f:
        state_of_the_union = f.read()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_text(state_of_the_union)
        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(
            texts, embeddings, metadatas=[{"source": i} for i in range(len(texts))]
        )
        with open(FAISS_PATH, "wb") as f:  # type: ignore
            assert isinstance(f, BufferedWriter)  # mypy complains otherwise
            pickle.dump(docsearch, f)
        return docsearch
