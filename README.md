---
title: Mini RAG Project
emoji: 💬
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
---
# Mini RAG Q&A Application 💬

This is a Retrieval-Augmented Generation application built for an AI Engineer assessment. It allows users to upload a document and ask questions about its content, receiving answers grounded in the provided text with inline citations.

**Live Application URL:** [https://huggingface.co/spaces/Akshatjangid/mini_rag](https://huggingface.co/spaces/Akshatjangid/mini_rag)

---
## Architecture

The application follows a standard RAG pipeline:
1.  **User Interface (Streamlit):** A web-based frontend for user interaction.
2.  **Vector DB (Qdrant):** Stores document chunks and their vector embeddings in a cloud-hosted database.
3.  **Retriever:** Fetches the top 10 most relevant chunks from Qdrant based on the user's query.
4.  **Reranker (Cohere):** Re-orders the retrieved chunks to identify the top 3 for maximum relevance.
5.  **LLM (Groq):** Generates a final, cited answer using the top reranked chunks as context.

---
## Technical Specifications

- **Cloud Services:** Qdrant Cloud, Cohere API, Groq Cloud
- **Frontend:** Streamlit
- **Hosting:** Hugging Face Spaces
- **Embedding Model:** `BAAI/bge-small-en-v1.5` (384 dimensions)
- **Reranker Model:** `rerank-english-v3.0`
- **LLM:** `llama-3.1-70b-versatile`
- **Chunking Strategy:** Size: 1000 tokens, Overlap: 150 tokens

---
## Quick Start (Local Setup)

To run this project locally, follow these steps:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Akshatjangid/mini-rag-project.git](https://github.com/Akshatjangid/mini-rag-project.git)
    ```
2.  Navigate to the project directory:
    ```bash
    cd mini-rag-project
    ```
3.  Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create a `.env` file and add your API keys (use `.env.example` as a template).
5.  Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

---
## Evaluation

The application was tested on the Wikipedia article for the "ICC World Test Championship".

**Success Rate: 5/5**

1.  **Q: Who is the current champion?**
    - **A:** The current ICC World Test Champion is South Africa. (Correct)
2.  **Q: Who has scored the most runs in the tournament's history?**
    - **A:** Joe Root has scored the most runs with 6,080. (Correct)
3.  **Q: (*Add your third question here*)**
    - **A:** (*Add the answer your app gave*)
4.  **Q: (*Add your fourth question here*)**
    - **A:** (*Add the answer your app gave*)
5.  **Q: (*Add your fifth question here*)**
    - **A:** (*Add the answer your app gave*)

---
## Remarks

A key challenge during this project was navigating the frequent decommissioning of LLM models on the Groq API, which required updating the model ID multiple times. I also learned the importance of using repository secrets for secure deployment on Hugging Face Spaces instead of relying on a local `.env` file, which led to initial `ModuleNotFound` and authentication errors that were resolved during the deployment process.

---
## Contact

- **Resume:** [https://drive.google.com/file/d/14QwlwtTELdf7CAeCNMhl6zAp5nbEK7M9/view?usp=sharing]
- **GitHub:** [https://github.com/Akshatjangid](https://github.com/Akshatjangid
