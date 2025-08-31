from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings



def load_pdf(data: str):
    loader = DirectoryLoader(
        path= data,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )

    document = loader.load()
    return document


def filter_to_minimal_document(docs: List[Document]) ->List[Document]:
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    
    return minimal_docs

def text_split(minimal_docs: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )

    texts_chunks = text_splitter.split_documents(minimal_docs)
    return texts_chunks


def download_embeddings():
    model_name = 'ibm-granite/granite-embedding-english-r2'
    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=model_name,
    )
    return embedding_model

embedding_model = download_embeddings()
