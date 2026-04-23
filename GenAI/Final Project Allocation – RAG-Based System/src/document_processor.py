"""
Document Processor Module
Handles PDF loading and text extraction
"""

import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pdfplumber
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TextBlock:
    """Represents a text block extracted from PDF"""
    text: str
    page_number: int
    section: Optional[str] = None
    char_position: int = 0
    source_file: str = ""


@dataclass
class PDFDocument:
    """Represents a loaded PDF document"""
    filename: str
    file_path: str
    total_pages: int
    text_blocks: List[TextBlock]
    metadata: Dict[str, Any]
    loaded_at: datetime


class DocumentProcessor:
    """Loads and processes PDF documents"""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def load_pdf(self, pdf_path: str) -> PDFDocument:
        """
        Load a PDF file and extract text with metadata
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            PDFDocument with extracted content
            
        Raises:
            FileNotFoundError: If PDF doesn't exist
            Exception: If PDF is corrupted
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        self.logger.info(f"Loading PDF: {pdf_path}")
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                text_blocks = []
                
                # Extract text from each page
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    
                    if text:
                        block = TextBlock(
                            text=text,
                            page_number=page_num,
                            source_file=pdf_path.name,
                            char_position=0
                        )
                        text_blocks.append(block)
                
                # Extract metadata
                metadata = {
                    "title": pdf.metadata.get("Title", "Unknown"),
                    "author": pdf.metadata.get("Author", "Unknown"),
                    "created": pdf.metadata.get("CreationDate", "Unknown"),
                    "subject": pdf.metadata.get("Subject", "Unknown")
                }
                
                doc = PDFDocument(
                    filename=pdf_path.name,
                    file_path=str(pdf_path),
                    total_pages=total_pages,
                    text_blocks=text_blocks,
                    metadata=metadata,
                    loaded_at=datetime.now()
                )
                
                self.logger.info(f"Successfully loaded PDF with {total_pages} pages, {len(text_blocks)} text blocks")
                return doc
                
        except Exception as e:
            self.logger.error(f"Error loading PDF {pdf_path}: {str(e)}")
            raise
    
    def batch_process_pdfs(self, pdf_directory: str) -> List[PDFDocument]:
        """
        Process all PDFs in a directory
        
        Args:
            pdf_directory: Path to directory with PDFs
            
        Returns:
            List of PDFDocuments
        """
        pdf_dir = Path(pdf_directory)
        
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory not found: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        self.logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        documents = []
        for pdf_file in pdf_files:
            try:
                doc = self.load_pdf(str(pdf_file))
                documents.append(doc)
            except Exception as e:
                self.logger.warning(f"Skipping {pdf_file}: {str(e)}")
        
        self.logger.info(f"Successfully processed {len(documents)} out of {len(pdf_files)} PDFs")
        return documents
    
    def extract_text_with_metadata(self, pdf: PDFDocument) -> List[Dict[str, Any]]:
        """
        Extract text blocks with full metadata
        
        Args:
            pdf: PDFDocument object
            
        Returns:
            List of dictionaries with text and metadata
        """
        results = []
        
        for block in pdf.text_blocks:
            result = {
                "text": block.text,
                "page": block.page_number,
                "section": block.section,
                "source_file": block.source_file,
                "char_position": block.char_position,
                "total_pages": pdf.total_pages,
                "doc_metadata": pdf.metadata
            }
            results.append(result)
        
        return results
