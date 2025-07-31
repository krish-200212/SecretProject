import os
import tempfile
import requests
from dotenv import load_dotenv
import re
from typing import List
import time
import json
import warnings
import hashlib
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore", category=FutureWarning)

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio

# LangChain & related
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredEmailLoader

# Google GenAI for direct embedding
from google.generativeai import configure, embed_content
from langchain.embeddings.base import Embeddings

load_dotenv()

# --------------------- Embedding Fix ---------------------
class GoogleGenAIEmbedder(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [
            embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )["embedding"]
            for text in texts
        ]

    def embed_query(self, text: str) -> List[float]:
        return embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )["embedding"]

# --------------------- FastAPI Setup ---------------------
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    configure(api_key=GOOGLE_API_KEY)

    ml_models["embeddings"] = GoogleGenAIEmbedder()

    if "SSL_CERT_FILE" in os.environ:
        del os.environ["SSL_CERT_FILE"]

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    ml_models["llm"] = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="meta-llama/llama-4-scout-17b-16e-instruct"
    )

    yield

app = FastAPI(title="HackRx RAG API", lifespan=lifespan)

# --------------------- Request/Response Models ---------------------
class QARequest(BaseModel):
    documents: str
    questions: List[str]

class QAResponse(BaseModel):
    answers: List[str]

# --------------------- Helper Functions ---------------------
def download_file(url):
    resp = requests.get(url)
    resp.raise_for_status()
    path = requests.utils.urlparse(url).path
    suffix = os.path.splitext(path)[1][1:].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}")
    tmp.write(resp.content)
    tmp.close()
    return tmp.name, suffix

def load_document_by_ext(file_path, ext):
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    elif ext == "eml":
        loader = UnstructuredEmailLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Only PDF, DOCX, and EML are supported.")
    return loader.load()

def trim_answer(answer, max_sentences=3, max_chars=350):
    sentences = re.split(r'(?<=[.!?]) +', answer)
    trimmed = ' '.join(sentences[:max_sentences])
    if len(trimmed) > max_chars:
        trimmed = trimmed[:max_chars].rsplit(' ', 1)[0] + '...'
    return trimmed

# --------------------- RAG Endpoint ---------------------
@app.post("/hackrx/run", response_model=QAResponse)
async def hackrx_run(request: QARequest):
    file_path, ext = download_file(request.documents)
    documents = load_document_by_ext(file_path, ext)
    os.remove(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    vectorstore = await asyncio.to_thread(
        FAISS.from_documents, splits, ml_models["embeddings"]
    )
    retriever = vectorstore.as_retriever()

    parser = StrOutputParser()
    system_prompt = (
        "You are a helpful AI assistant specialized in question answering related to insurance policies and related documents. "
        "Use the provided context to answer the question as clearly and precisely as possible. "
        "If the answer is not known from the context, then give the answer which is related to the contest. "
        "Keep answers concise, within two to three sentences.\n\n"
        "Make it strictly in three lines only not more than that. "
        "and if there are any answers reflecting the numbers also give the number in numerical format. "
        "and if there are any questions that have answer in the table so extract the content in the table related to the query. "
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    rag_chain = create_stuff_documents_chain(
        llm=ml_models["llm"],
        prompt=qa_prompt,
        output_parser=parser
    )
    retrieval_chain = create_retrieval_chain(retriever, rag_chain)

    async def get_rag_answer(q):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, retrieval_chain.invoke, {"input": q})
        return trim_answer(response['answer'])

    batch_size = 30
    all_answers = []
    for i in range(0, len(request.questions), batch_size):
        batch = request.questions[i:i+batch_size]
        batch_answers = await asyncio.gather(*(get_rag_answer(q) for q in batch))
        all_answers.extend(batch_answers)
        if i + batch_size < len(request.questions):
            await asyncio.sleep(60)
    return QAResponse(answers=all_answers)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
