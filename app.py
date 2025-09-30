# app.py

import streamlit as st
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import cohere
from groq import Groq

# --- 1. एनवायरनमेंट और क्लाइंट्स लोड करना ---

# .env फ़ाइल से वेरिएबल्स लोड करें
load_dotenv()

st.set_page_config(page_title="Mini RAG Final", layout="wide")
st.title("Mini RAG: Ask Your Document 💬")

# क्रेडेंशियल्स
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
COLLECTION_NAME = "mini_rag_collection"

# सभी क्लाइंट्स को इनिशियलाइज़ और कैश करें
@st.cache_resource
def load_clients_and_models():
    if not all([QDRANT_URL, QDRANT_API_KEY, COHERE_API_KEY, GROQ_API_KEY]):
        st.error("⚠️ कृपया सभी API keys और URL को .env फ़ाइल में सेट करें।")
        return None, None, None, None
    
    embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    cohere_client = cohere.Client(COHERE_API_KEY)
    groq_client = Groq(api_key=GROQ_API_KEY)
    
    return embedding_model, qdrant_client, cohere_client, groq_client

model, qdrant_client, cohere_client, groq_client = load_clients_and_models()

# --- 2. यूजर इंटरफ़ेस (UI) ---

# UI को दो कॉलम में बांटें
col1, col2 = st.columns(2)

with col1:
    st.header("1. Add Document to Knowledge Base")
    input_text = st.text_area("यहाँ अपना टेक्स्ट पेस्ट करें:", height=250, key="text_input")
    
    if st.button("Process & Store"):
        if input_text.strip() and all([model, qdrant_client]):
            with st.spinner("⏳ Processing and storing text..."):
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, length_function=len)
                chunks = text_splitter.split_text(input_text)
                
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=[
                        models.PointStruct(
                            id=str(uuid.uuid4()),
                            vector=model.encode(chunk).tolist(),
                            payload={"text": chunk}
                        ) for chunk in chunks
                    ],
                    wait=True
                )
                st.success("✅ Document processed and stored successfully!")
                st.session_state.document_processed = True

with col2:
    st.header("2. Ask a Question")
    query = st.text_input("अपने डॉक्यूमेंट के बारे में एक सवाल पूछें:", key="query_input")

    if st.button("Get Answer"):
        if not query.strip():
            st.warning("⚠️ कृपया एक सवाल पूछें।")
        elif not all([model, qdrant_client, cohere_client, groq_client]):
             st.error("Clients are not initialized. Please check your credentials.")
        else:
            with st.spinner("⏳ Finding the best answer..."):
                # --- 3. RAG पाइपलाइन का लॉजिक ---
                
                # a. Retrieve: Qdrant से मिलते-जुलते चंक्स खोजें
                query_vector = model.encode(query).tolist()
                retrieved_hits = qdrant_client.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=10  # 10 सबसे मिलते-जुलते रिजल्ट्स प्राप्त करें
                )
                retrieved_docs = [hit.payload['text'] for hit in retrieved_hits]

                # b. Rerank: Cohere से रिजल्ट्स को और बेहतर बनाएँ
                reranked_docs = cohere_client.rerank(
                    model='rerank-english-v3.0',
                    query=query,
                    documents=retrieved_docs,
                    top_n=3  # 3 सबसे सटीक रिजल्ट्स चुनें
                )

                context_docs = [retrieved_docs[doc.index] for doc in reranked_docs.results]
                
                # c. Answer: Groq LLM से जवाब बनाएँ
                context_for_prompt = "\n\n".join([f"[source_{i+1}]: {doc}" for i, doc in enumerate(context_docs)])
                
                prompt = f"""
                You are an expert Q&A system. Your answer must be grounded in the provided context.
                Answer the user's question based ONLY on the following context.
                For each statement, you MUST cite the source using the format [source_N].
                If the context does not contain the answer, reply with "I cannot answer this question based on the provided text."

                Context:
                {context_for_prompt}

                Question: {query}
                Answer:
                """

                chat_completion = groq_client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.1-8b-instant",
                )
                final_answer = chat_completion.choices[0].message.content

                # --- 4. रिजल्ट्स दिखाएँ ---
                st.subheader("Answer:")
                st.markdown(final_answer)
                
                with st.expander("📚 See Sources"):
                    for i, doc in enumerate(context_docs):
                        st.markdown(f"**Source {i+1}:**")
                        st.info(doc)