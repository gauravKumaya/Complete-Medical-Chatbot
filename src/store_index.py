from dotenv import load_dotenv
from src.helper import load_pdf, filter_to_minimal_document, text_split, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
load_dotenv()

extracted_data = load_pdf("data")
minimal_docs = filter_to_minimal_document(extracted_data)
text_chunks = text_split(minimal_docs)

embedding_model = download_embeddings()

pc = Pinecone()
index_name = 'medical-chatbot'

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

index = pc.Index(index_name)


vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_model,
    index_name=index_name
)

for i in range(0, len(text_chunks), 500):
    batch = text_chunks[i:i+100]
    vector_store.add_documents(batch)
