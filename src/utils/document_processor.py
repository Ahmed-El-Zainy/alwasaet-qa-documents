import os
import logging
from typing import List, Optional
from pathlib import Path
import PyPDF2 # mypdf2
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_parse import LlamaParse
import yaml
import sys 
import os 


# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

try:
    from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("document_processor")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("document processor")
    logger.info("Using standard logger - custom logger not available")

class DocumentProcessor:
    """Handles document parsing and preprocessing"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.use_llama_parse = self.config['parsing']['use_llama_parse']
        self.chunk_size = self.config['embeddings']['chunk_size']
        self.chunk_overlap = self.config['embeddings']['chunk_overlap']
        
        # Initialize LlamaParse if enabled
        if self.use_llama_parse and os.getenv("LLAMA_CLOUD_API_KEY"):
            self.parser = LlamaParse(
                result_type="text",
                verbose=True,
                language="en"
            )
        else:
            self.parser = None
        
        # Initialize node parser
        self.node_parser = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def extract_text_from_pdf(self, file_path: str) -> List[Document]:
        """Extract text from PDF using PyPDF2 with page tracking"""
        documents = []
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        doc = Document(
                            text=text,
                            metadata={
                                "file_name": Path(file_path).name,
                                "page_number": page_num,
                                "source": file_path
                            }
                        )
                        documents.append(doc)
                        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
        
        return documents
    
    def process_with_llama_parse(self, file_path: str) -> List[Document]:
        """Process document using LlamaParse"""
        try:
            documents = self.parser.load_data(file_path)
            
            # Add metadata
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    "file_name": Path(file_path).name,
                    "page_number": i + 1,
                    "source": file_path
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error processing with LlamaParse {file_path}: {str(e)}")
            # Fallback to PyPDF2
            return self.extract_text_from_pdf(file_path)
    
    def process_document(self, file_path: str) -> List[Document]:
        """Process a single document"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            if self.parser and self.use_llama_parse:
                documents = self.process_with_llama_parse(file_path)
            else:
                documents = self.extract_text_from_pdf(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Split into nodes
        nodes = []
        for doc in documents:
            doc_nodes = self.node_parser.get_nodes_from_documents([doc])
            # Preserve metadata in nodes
            for node in doc_nodes:
                node.metadata.update(doc.metadata)
            nodes.extend(doc_nodes)
        
        return nodes



if __name__=="__main__":
    logger.info("test ..")