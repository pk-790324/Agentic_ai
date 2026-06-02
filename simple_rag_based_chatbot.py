import warnings
warnings.filterwarnings("ignore")

import gradio as gr

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_ollama import OllamaLLM, OllamaEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


# =========================
# LLM (TinyLlama via Ollama)
# =========================
def get_llm():
    return OllamaLLM(
        model="tinyllama:1.1b",
        temperature=0.2
    )


# =========================
# Embeddings (Ollama)
# =========================
def get_embeddings():
    return OllamaEmbeddings(
        model="nomic-embed-text"
    )


# =========================
# Load PDF
# =========================
def document_loader(file):
    loader = PyPDFLoader(file.name)
    return loader.load()


# =========================
# Split Text
# =========================
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    return splitter.split_documents(data)


# =========================
# Vector DB (Chroma)
# =========================
def vector_database(chunks):
    embedding_model = get_embeddings()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    vectordb.persist()
    return vectordb


# =========================
# Retriever
# =========================
def retriever(file):
    docs = document_loader(file)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()


# =========================
# RAG QA Chain (MODERN LCEL)
# =========================
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant.
        Answer ONLY using the given context.

        Context:
        {context}

        Question:
        {input}

        If the answer is not in the context, say "I don't know based on the document."
        """
    )

    # LCEL chain: retriever -> format docs -> prompt -> llm
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    rag_chain = (
        {"context": retriever_obj | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
    )

    response = rag_chain.invoke(query)

    return response


# =========================
# Gradio UI
# =========================
rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=[".pdf"],
            type="filepath"
        ),
        gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Ask a question from the document..."
        )
    ],
    outputs=gr.Textbox(label="Answer"),
    title="RAG Chatbot (Ollama + Chroma + TinyLlama)",
    description="Upload a PDF and ask questions. The system retrieves relevant chunks and answers using TinyLlama."
)


# =========================
# Launch App
# =========================
if __name__ == "__main__":
    rag_application.launch()