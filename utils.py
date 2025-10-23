import os
import torch
import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms.base import LLM
from transformers import pipeline
from PyPDF2 import PdfReader
import docx

# ---------------- Embeddings & Vector Store ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_documents(folder="data_files"):
    if not os.path.exists(folder):
        os.makedirs(folder)
    docs = []
    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            loader = TextLoader(os.path.join(folder, file_name), encoding="utf-8")
            docs.extend(loader.load())
    return docs

documents = load_documents()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': device}
    
)
vector_store = FAISS.from_documents(chunks, embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# ---------------- Local LLM ----------------
llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)

class LocalLLM(LLM):
    def _call(self, prompt, stop=None):
        result = llm_pipeline(prompt, max_length=512, truncation=True)
        return result[0]['generated_text']

    @property
    def _identifying_params(self):
        return {"model": "local-flan-t5"}

    @property
    def _llm_type(self):
        return "local"

llm = LocalLLM()
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
    memory=memory
)

# ---------------- File Upload Helper ----------------
def process_uploaded_files(uploaded_files):
    combined_data = pd.DataFrame()
    for file in uploaded_files:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        elif file.name.endswith(".txt"):
            df = pd.DataFrame([{"Text": file.read().decode("utf-8")}])
        elif file.name.endswith(".pdf"):
            pdf = PdfReader(file)
            text = "\n".join([p.extract_text() for p in pdf.pages if p.extract_text()])
            df = pd.DataFrame([{"Text": text}])
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            text = "\n".join([p.text for p in doc.paragraphs])
            df = pd.DataFrame([{"Text": text}])
        combined_data = pd.concat([combined_data, df], ignore_index=True)
    return combined_data

def create_qa_chain_from_texts(texts):
    """
    Create a new ConversationalRetrievalChain from uploaded document texts.
    """
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import ConversationalRetrievalChain

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([t for t in texts if t.strip()])

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device}
    )
    vector_store = FAISS.from_documents(docs, embeddings)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain_new = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        memory=memory
    )

    return qa_chain_new
