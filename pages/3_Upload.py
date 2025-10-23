import streamlit as st
import matplotlib.pyplot as plt
from utils import process_uploaded_files, create_qa_chain_from_texts
import re

st.title("📂 Upload, Analyze & Chat with Your Documents")

# -------------------- FILE UPLOAD --------------------
uploaded_files = st.file_uploader(
    "Upload files (txt, csv, xlsx, pdf, docx)",
    type=["txt", "csv", "xlsx", "pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    # Process files once
    st.session_state.file_data = process_uploaded_files(uploaded_files)
    st.success("✅ Files uploaded and processed successfully!")

    # Build QA chain for uploaded documents
    if "Text" in st.session_state.file_data.columns:
        all_texts = st.session_state.file_data["Text"].astype(str).tolist()
        st.session_state.qa_chain = create_qa_chain_from_texts(all_texts)

    st.markdown("---")

    # -------------------- DOCUMENT SUMMARY BUTTON --------------------
    st.subheader("📘 Document Summary (Optional)")
    if st.button("Show Document Summary"):
        st.info("📄 Document Summary Details:")
        for f in uploaded_files:
            file_name = f.name
            file_data = st.session_state.file_data
            if "Text" in file_data.columns:
                text_content = " ".join(file_data["Text"].tolist())
                word_count = len(re.findall(r'\w+', text_content))
                st.markdown(f"**📁 File:** {file_name}")
                st.markdown(f"- Total Words: `{word_count}`")
                if file_name.lower().endswith(".pdf"):
                    st.markdown(f"- Estimated Pages: `{text_content.count('') + 1}`")
                st.markdown("---")

    st.markdown("---")
    

    # -------------------- CHAT SECTION --------------------
    st.subheader("💬 Chat About Your Uploaded Documents")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    def on_enter():
        user_q = st.session_state.user_question.strip()
        if user_q:
            try:
                response = st.session_state.qa_chain.run(user_q)
                if not response or "sorry" in response.lower() or len(response.split()) < 3:
                    response = "❌ No relevant information found in your uploaded documents."
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

    st.text_input(
        "Ask a question about your uploaded document:",
        key="user_question",
        on_change=on_enter,
        placeholder="Type your question and press Enter..."
    )

    if st.session_state.chat_history:
        for chat in reversed(st.session_state.chat_history):
            st.markdown(f"**🧑 You:** {chat['question']}")
            st.markdown(f"**🤖 Bot:** {chat['answer']}") 
            st.markdown("---")
    else:
        st.info("Start chatting with your uploaded document...")

else:
    st.info("📥 Please upload at least one document to start chatting or view charts.")
