"""
Chunking Module
Splits documents into semantic chunks with configurable size and overlap
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """Represents a chunk of text"""
    text: str
    chunk_id: str = field(default_factory=lambda: f"chunk_{uuid.uuid4().hex[:12]}")
    source_file: str = ""
    page_number: int = 0
    start_char: int = 0
    end_char: int = 0
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def __repr__(self):
        return f"Chunk(id={self.chunk_id}, page={self.page_number}, size={len(self.text)})"


class ChunkingStrategy:
    """Base class for chunking strategies"""
    
    def __init__(self, chunk_size: int = 1024, overlap: int = 128):
        """
        Initialize chunking strategy
        
        Args:
            chunk_size: Target size in tokens (~4 chars per token)
            overlap: Overlap between chunks in tokens
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.char_size = chunk_size * 4  # Approximate chars per token
        self.char_overlap = overlap * 4
        
        logger.info(f"Initialized chunking: size={chunk_size} tokens, overlap={overlap} tokens")
    
    def chunk(self, text: str, source_file: str = "", page_number: int = 0) -> List[Chunk]:
        """
        Split text into chunks (to be implemented by subclasses)
        
        Args:
            text: Text to chunk
            source_file: Name of source file
            page_number: Page number in document
            
        Returns:
            List of Chunks
        """
        raise NotImplementedError


class FixedSizeChunker(ChunkingStrategy):
    """
    Simple fixed-size chunking strategy
    Splits text into fixed-size chunks with overlap
    """
    
    def chunk(self, text: str, source_file: str = "", page_number: int = 0) -> List[Chunk]:
        """Split text into fixed-size chunks"""
        
        chunks = []
        text_length = len(text)
        char_size = self.char_size
        char_overlap = self.char_overlap
        
        start = 0
        chunk_num = 0
        
        while start < text_length:
            end = min(start + char_size, text_length)
            
            # Extract chunk
            chunk_text = text[start:end]
            
            # Create chunk object
            chunk = Chunk(
                text=chunk_text,
                source_file=source_file,
                page_number=page_number,
                start_char=start,
                end_char=end,
                metadata={
                    "chunk_number": chunk_num,
                    "chunking_strategy": "fixed_size"
                }
            )
            chunks.append(chunk)
            
            # Move start position (with overlap)
            start = end - char_overlap
            chunk_num += 1
            
            # Avoid infinite loop if chunk is very small
            if start >= end:
                break
        
        return chunks


class SemanticChunker(ChunkingStrategy):
    """
    Semantic chunking strategy
    Splits on sentence/paragraph boundaries to preserve meaning
    """
    
    def chunk(self, text: str, source_file: str = "", page_number: int = 0) -> List[Chunk]:
        """Split text on semantic boundaries (paragraphs/sentences)"""
        
        chunks = []
        
        # Step 1: Split by paragraphs (double newline)
        paragraphs = text.split("\n\n")
        
        current_chunk_text = ""
        current_chunk_start = 0
        chunk_num = 0
        char_position = 0
        
        for para in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if (len(current_chunk_text) + len(para) > self.char_size 
                and current_chunk_text.strip()):
                
                chunk = Chunk(
                    text=current_chunk_text.strip(),
                    source_file=source_file,
                    page_number=page_number,
                    start_char=current_chunk_start,
                    end_char=char_position,
                    metadata={
                        "chunk_number": chunk_num,
                        "chunking_strategy": "semantic",
                        "boundary": "paragraph"
                    }
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap (last sentences of previous chunk)
                overlap_text = self._get_sentence_overlap(current_chunk_text)
                current_chunk_text = overlap_text + "\n\n" + para
                current_chunk_start = char_position - len(overlap_text)
                chunk_num += 1
            else:
                # Add paragraph to current chunk
                if current_chunk_text:
                    current_chunk_text += "\n\n"
                current_chunk_text += para
            
            char_position += len(para) + 2  # +2 for newlines
        
        # Add final chunk
        if current_chunk_text.strip():
            chunk = Chunk(
                text=current_chunk_text.strip(),
                source_file=source_file,
                page_number=page_number,
                start_char=current_chunk_start,
                end_char=char_position,
                metadata={
                    "chunk_number": chunk_num,
                    "chunking_strategy": "semantic",
                    "boundary": "document_end"
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_sentence_overlap(self, text: str) -> str:
        """Get last 1-2 sentences for overlap (roughly 128 tokens)"""
        sentences = text.split(". ")
        
        # Get last sentence(s) that fit in overlap budget
        overlap_sentences = []
        total_chars = 0
        
        for sentence in reversed(sentences):
            sentence_len = len(sentence)
            if total_chars + sentence_len > self.char_overlap:
                break
            overlap_sentences.insert(0, sentence)
            total_chars += sentence_len
        
        return ". ".join(overlap_sentences) + "." if overlap_sentences else ""


def create_chunker(strategy: str = "semantic", chunk_size: int = 1024, 
                   overlap: int = 128) -> ChunkingStrategy:
    """
    Factory function to create chunker instances
    
    Args:
        strategy: "semantic" or "fixed"
        chunk_size: Chunk size in tokens
        overlap: Overlap in tokens
        
    Returns:
        ChunkingStrategy instance
    """
    if strategy == "semantic":
        return SemanticChunker(chunk_size=chunk_size, overlap=overlap)
    elif strategy == "fixed":
        return FixedSizeChunker(chunk_size=chunk_size, overlap=overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")
