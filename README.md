# 💬 PDF Chat App with Ollama and RAG

Chat with your PDF files using powerful local language models (LLMs) and Retrieval-Augmented Generation (RAG).  
Built with **Streamlit**, **LangChain**, **ChromaDB**, and **Ollama** — all running inside a single Docker container.

## Image Demo
![pdf chat](pdfchat.png)


## 🚀 Features

- Upload a PDF and chat with it using natural language
- Uses **Ollama** to run local models like `qwen2:0.5b`
- Fast document chunking and vector search with **LangChain + Chroma**
- Clean and interactive **Streamlit UI**
- Fully containerized with **Docker**
  
## Installation 🛠️
1. Clone the repository:
    ```bash
    git clone https://github.com/YOUR_USERNAME/pdf-chat-app.git
    cd pdf-chat-app
    
2. Install dependencies:
    ```bash
   pip install -r requirements.txt
    
3. Run the app locally

## Running with Docker
1. Build the Docker image:
   ```bash
   docker build -t pdf-chat .
2. Run the Docker container:
   ```bash
   docker run -p 8501:8501 -p 11434:11434 pdf-chat
## ⚙️ How It Works
- Upload a PDF file
- It's split into chunks and embedded using nomic-embed-text
- The chunks are stored in ChromaDB
- A local LLM (here i used qwen2:0.5b) answers your questions using RAG

## 📌 Notes
- Vector data is stored in db/ (excluded from Git)
- Only qwen2:0.5b is used by default to keep memory usage low
- You can clear chat/cache from the sidebar at any time
   

