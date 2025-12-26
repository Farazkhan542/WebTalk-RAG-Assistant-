# Chat with Website – AI RAG Application

This project is an AI-powered application that allows users to chat with the content of any website. It fetches and cleans website text, converts it into embeddings using Google Gemini, stores them in ChromaDB, and answers user questions using a retrieval-based approach.

---

## Features
- Load and process content from any public website URL  
- Generate embeddings using Gemini Embedding Model  
- Store and retrieve data using ChromaDB  
- Ask questions and get answers based only on website content  
- Simple and interactive Streamlit interface  

---

## Technologies Used
- Python  
- Streamlit  
- LangChain  
- Google Gemini (Embeddings & LLM)  
- ChromaDB  
- BeautifulSoup  

---

## How to Run
1. Install required packages  
2. Add your Google API key in a `.env` file  
3. Run the app using:
   ```bash
   streamlit run app.py
