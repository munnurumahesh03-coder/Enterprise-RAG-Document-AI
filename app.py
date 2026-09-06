import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- UI Configuration ---
st.set_page_config(page_title="Enterprise Document AI", page_icon="📄", layout="wide")
st.title("📄 Enterprise Document AI & RAG Engine")
st.markdown("Upload any PDF document and ask questions. The AI will extract the exact context and provide hallucination-free answers.")

# Fetch API key securely from Streamlit Secrets
# (This ensures the user never has to type it!)
# Check if we are on Streamlit Cloud (secrets) or Docker/Render (Environment Variables)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


# --- Sidebar: Setup & Upload ---
with st.sidebar:
    st.header("📂 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    
    if st.button("Process Document"):
        if not uploaded_file:
            st.error("⚠️ Please upload a PDF document.")
        else:
            with st.spinner("Chunking and Embedding Document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = text_splitter.split_documents(pages)
                
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_db = Chroma.from_documents(documents=chunks, embedding=embeddings)
                
                st.session_state.vector_db = vector_db
                st.success("✅ Document processed successfully! You can now chat.")

# --- Helper Function ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if "vector_db" not in st.session_state:
            st.warning("⚠️ Please upload and process a PDF document first.")
        else:
            with st.spinner("Searching document..."):
                llm = ChatGroq(temperature=0, model_name="qwen/qwen3.8-27b", api_key=GROQ_API_KEY, max_tokens=500)
                retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
                
                template = """
                You are an elite AI Research Assistant. Answer the question based ONLY on the following context.
                If the answer is not in the context, say "I cannot answer this based on the provided document."
                Do not hallucinate.

                Context:
                {context}

                Question: {question}

                Answer:
                """
                prompt_template = ChatPromptTemplate.from_template(template)
                
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
