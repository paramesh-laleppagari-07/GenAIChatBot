import streamlit as st
from transformers import pipeline
from utils import process_uploaded_files, create_qa_chain_from_texts
import os
import io

# -------------------- PAGE SETUP --------------------
st.set_page_config(page_title="My AI Chatbot", layout="wide", page_icon="🤖")
st.title("🤖 Welcome to My Smart AI Chatbot")

st.markdown("""
This chatbot can answer your questions from documents (if available) or chat naturally like an assistant.  
**Developed by [Laleppagari Paramesh]**
""")

# -------------------- LOAD DOCUMENTS FROM FOLDER --------------------
DATA_FOLDER = r"C:\Users\HP\Desktop\AIChatBot\GenAIChatBot\data_files"

def load_all_files_from_folder(folder_path):
    files = []
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        if os.path.isfile(fpath) and fname.lower().endswith((".txt", ".csv", ".xlsx", ".pdf", ".docx")):
            with open(fpath, "rb") as f:
                file_bytes = f.read()
                file_obj = io.BytesIO(file_bytes)
                file_obj.name = fname
                files.append(file_obj)
    return files

files = load_all_files_from_folder(DATA_FOLDER)

if files:
    from utils import process_uploaded_files, create_qa_chain_from_texts
    st.session_state.file_data = process_uploaded_files(files)

    if "Text" in st.session_state.file_data.columns:
        texts = st.session_state.file_data["Text"].astype(str).tolist()
        st.session_state.qa_chain = create_qa_chain_from_texts(texts)
else:
    st.warning("⚠️ No documents found in your data_files folder. The chatbot will use general responses.")

# -------------------- CHATBOT --------------------
st.subheader("💬 Chat with Your AI Assistant")

# Create model for general conversation
chat_model = pipeline(
    "text2text-generation",
    model="google/flan-t5-large",
    max_length=200,
    truncation=True
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def on_enter():
    user_q = st.session_state.user_question.strip()
    if user_q:
        try:
            # Try to get document-based answer first
            response = ""
            if "qa_chain" in st.session_state:
                response = st.session_state.qa_chain.run(user_q)

            # If no relevant answer found → fallback to general chatbot
            if not response or len(response.split()) < 3 or "sorry" in response.lower():
                response = chat_model(
                    f"Answer this like a friendly helpful AI assistant: {user_q}",
                    max_length=150,
                    truncation=True
                )[0]["generated_text"]

            # Save chat
            st.session_state.chat_history.append({
                "question": user_q,
                "answer": response
            })

        except Exception as e:
            st.session_state.chat_history.append({
                "question": user_q,
                "answer": f"⚠️ Error: {str(e)}"
            })

        st.session_state.user_question = ""

# Input box
st.text_input(
    "Type your question here...",
    key="user_question",
    on_change=on_enter,
    placeholder="Ask something... (Press Enter to send)"
)

# -------------------- CHAT HISTORY --------------------
if st.session_state.chat_history:
    for chat in reversed(st.session_state.chat_history):
        st.markdown(f"**🧑 You:** {chat['question']}**")
        st.markdown(f"**🤖 Bot:** {chat['answer']}**")
        st.markdown("---")

# -------------------- CLEAR HISTORY BUTTON --------------------
if st.button("🗑️ Clear Chat History"):
    st.session_state.chat_history = []
    st.success("Chat history cleared!")
    st.rerun()

else:   
    st.info("💡 Type a question and press Enter to chat with the AI assistant." )