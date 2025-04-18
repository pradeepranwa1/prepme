from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
import os
import PyPDF2
from pathlib import Path
import chromadb
from chromadb.config import Settings
import shutil
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from utils.prompt_utils import split_into_sentences, create_chunks, get_embeddings
from utils.s3_utils import upload_file_to_s3, check_file_in_s3
from pydantic import BaseModel
from utils.auth_utils import hash_password, verify_password, create_access_token, decode_access_token
from datetime import timedelta
from database import SessionLocal, User, get_db
from sqlalchemy.orm import Session

# Load environment variables
load_dotenv()

# Set tokenizers parallelism to false to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

class UserCreate(BaseModel):
    username: str
    password: str
    email: str
    full_name: str

class UserLogin(BaseModel):
    username: str
    password: str

# Initialize ChromaDB client
chroma_client = chromadb.Client(Settings(
    persist_directory="db",
    is_persistent=True
))

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

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    # Check if username already exists
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Check if email already exists
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user
    hashed_password = hash_password(user.password)
    db_user = User(
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        hashed_password=hashed_password
    )
    
    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return {"message": "User registered successfully"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/login")
def login(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.username}, expires_delta=timedelta(minutes=30))
    return {"access_token": token, "token_type": "bearer"}

@app.get("/protected")
def protected_route(token: str = Depends(decode_access_token)):
    if not token:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    return {"message": "You have access!", "user": token["sub"]}

@app.post("/upload-book/")
async def upload_book(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save the uploaded file
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    if check_file_in_s3(f"books/{file.filename}"):
        raise HTTPException(status_code=400, detail="File already exists in S3")

    # Upload file to S3 using utility function"""
    file_url = upload_file_to_s3(file.file, f"books/{file.filename}")
    
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
            "total_chunks": len(chunks),
            "url": file_url
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
        answer = llm_chain.invoke({"context": context, "question": question})["text"]
        
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