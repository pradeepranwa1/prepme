from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Optional, List
import PyPDF2
from pathlib import Path
import chromadb
from chromadb.config import Settings
import shutil
import re
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

os.environ["OPENAI_API_KEY"] = "test"

app = FastAPI(title="Book Q&A API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="db",
    is_persistent=True
))

# Initialize the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize OpenAI LLM
llm = ChatOpenAI(
    temperature=0.7,
    model_name="gpt-3.5-turbo"
)

# Create prompt template for answer generation
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Based on the following context from a book, please answer the question.
    If the answer cannot be found in the context, say "I cannot find the answer in the provided context."
    Keep the answer concise but informative.

    Context:
    {context}

    Question: {question}

    Answer:"""
)

# Create LLM chain
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex pattern."""
    # This pattern handles common sentence endings including Mr., Mrs., Dr., etc.
    sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def create_chunks(sentences: List[str], max_chunk_size: int = 1000) -> List[str]:
    """Create chunks of text while preserving sentence boundaries."""
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If adding this sentence would exceed max_chunk_size and we already have content,
        # start a new chunk
        if current_size + sentence_size > max_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using sentence-transformers."""
    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()

@app.post("/upload-book/")
async def upload_book(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save the uploaded file
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process the PDF
    try:
        # Extract text from PDF
        with open(file_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        
        # Create a collection for this book
        collection_name = f"book_{file.filename.replace('.pdf', '')}"
        collection = chroma_client.create_collection(name=collection_name)
        
        # Split text into sentences and then create chunks
        sentences = split_into_sentences(text)
        chunks = create_chunks(sentences)
        
        # Generate embeddings for chunks
        embeddings = get_embeddings(chunks)
        
        # Add chunks and their embeddings to ChromaDB
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{
                    "source": file.filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }],
                ids=[f"chunk_{i}"]
            )
        
        return {
            "message": "Book processed successfully",
            "filename": file.filename,
            "total_chunks": len(chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask-question/")
async def ask_question(book_name: str, question: str):
    try:
        # Get the collection for the specified book
        collection_name = f"book_{book_name.replace('.pdf', '')}"
        collection = chroma_client.get_collection(name=collection_name)
        
        # Generate embedding for the question
        question_embedding = get_embeddings([question])[0]
        
        # Query the collection using the question embedding
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )
        
        # Combine relevant chunks for context
        context = "\n".join(results['documents'][0])
        
        # Generate answer using LLM
        answer = llm_chain.run(context=context, question=question)
        print("hehe\ns,d\n")
        
        return {
            "question": question,
            "context": context,
            "answer": answer
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/list-books/")
async def list_books():
    try:
        # Get all collections (books)
        collections = chroma_client.list_collections()
        return {
            "books": [collection.name.replace("book_", "") for collection in collections]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 