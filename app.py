from flask import Flask, render_template, jsonify, request
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

load_dotenv()


app = Flask(__name__)

embedding_model = download_embeddings()

index_name = 'medical-chatbot'

vector_store = PineconeVectorStore.from_existing_index(
    embedding=embedding_model,
    index_name=index_name
)

retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k':3})

chat_model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(chat_model, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/get", methods=['GET', 'POST'])
def chat():
    msg = request.form['msg']
    print(msg)
    response = rag_chain.invoke({'input': msg})
    print("Response: ", response['answer'])
    return str(response['answer'])


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True) 