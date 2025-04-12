from typing import Optional, List
import re
from sentence_transformers import SentenceTransformer
# Initialize the sentence transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

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
