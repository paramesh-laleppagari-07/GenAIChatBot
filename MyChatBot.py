import streamlit as st

st.set_page_config(
    page_title="My Local AI ChatBot",
    layout="wide",
    page_icon="🤖"
)

st.title("🤖 Welcome to My Local AI ChatBot")

st.markdown("""
### 📖 About This Bot
This chatbot can:
- 💬 Answer questions using AI or from your uploaded documents  
- 📂 Support multiple file types — PDF, DOCX, TXT, CSV, XLSX  
- 📊 Display quick charts and text analysis  
- 🧠 Use a local FLAN-T5 model with HuggingFace Embeddings (FAISS vector store)  

Use the **sidebar** to open the **Upload & Chat** page to begin chatting with your data.
""")

st.info("👉 Go to **Upload & Chat** page to upload a document and start asking questions.")
