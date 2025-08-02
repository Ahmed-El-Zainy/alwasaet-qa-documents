import os
import logging
import re
from typing import List, Optional, Dict, Any
from pathlib import Path
import PyPDF2
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
import yaml
import sys 
import unicodedata

# Add path for custom logger
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

try:
    from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("bilingual_processor")
    logger.info("Custom Logger Start Working.....")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("bilingual processor")
    logger.info("Using standard logger - custom logger not available")

class LanguageDetector:
    """Simple language detection for Arabic and English"""
    
    def __init__(self):
        # Arabic Unicode ranges
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        # English pattern
        self.english_pattern = re.compile(r'[a-zA-Z]+')
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect primary language of text"""
        if not text.strip():
            return {"language": "unknown", "confidence": 0.0, "mixed": False}
        
        # Count Arabic and English characters
        arabic_chars = len(self.arabic_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        total_chars = arabic_chars + english_chars
        
        if total_chars == 0:
            return {"language": "unknown", "confidence": 0.0, "mixed": False}
        
        arabic_ratio = arabic_chars / total_chars
        english_ratio = english_chars / total_chars
        
        # Determine primary language
        if arabic_ratio > 0.6:
            language = "arabic"
            confidence = arabic_ratio
        elif english_ratio > 0.6:
            language = "english" 
            confidence = english_ratio
        else:
            language = "mixed"
            confidence = max(arabic_ratio, english_ratio)
        
        # Check if text is mixed language
        is_mixed = min(arabic_ratio, english_ratio) > 0.1
        
        return {
            "language": language,
            "confidence": confidence,
            "mixed": is_mixed,
            "arabic_ratio": arabic_ratio,
            "english_ratio": english_ratio
        }

class BilingualDocumentProcessor:
    """Enhanced document processor for Arabic and English content"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.language_detector = LanguageDetector()
        
        # Get language-specific settings
        self.lang_config = self.config.get('language_support', {})
        self.arabic_config = self.lang_config.get('arabic', {})
        self.english_config = self.lang_config.get('english', {})
        
        # Initialize node parsers for different languages
        self._initialize_parsers()
    
    def _initialize_parsers(self):
        """Initialize language-specific text splitters"""
        # Arabic parser - smaller chunks
        self.arabic_parser = SentenceSplitter(
            chunk_size=self.arabic_config.get('chunk_size', 400),
            chunk_overlap=self.arabic_config.get('chunk_overlap', 80)
        )
        
        # English parser - larger chunks
        self.english_parser = SentenceSplitter(
            chunk_size=self.english_config.get('chunk_size', 800),
            chunk_overlap=self.english_config.get('chunk_overlap', 150)
        )
        
        # Default parser - balanced
        default_chunk_size = self.config['embeddings']['chunk_size']
        default_chunk_overlap = self.config['embeddings']['chunk_overlap']
        self.default_parser = SentenceSplitter(
            chunk_size=default_chunk_size,
            chunk_overlap=default_chunk_overlap
        )
    
    def preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text based on language"""
        if not text:
            return text
        
        # Common preprocessing
        text = unicodedata.normalize('NFKC', text)  # Unicode normalization
        
        if language == "arabic":
            # Arabic-specific preprocessing
            # Remove excessive whitespace but preserve Arabic text structure
            text = re.sub(r'\s+', ' ', text)
            # Remove unwanted characters but keep Arabic punctuation
            text = re.sub(r'[^\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\u0020-\u007E\s]', '', text)
            
        elif language == "english":
            # English-specific preprocessing
            text = re.sub(r'\s+', ' ', text)
            # Keep standard English characters and punctuation
            text = re.sub(r'[^\w\s\.,!?;:()\[\]{}"\'-]', ' ', text)
        
        return text.strip()
    
    def extract_text_from_pdf(self, file_path: str) -> List[Document]:
        """Extract text from PDF with language detection"""
        documents = []
        file_name = Path(file_path).name
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        text = page.extract_text()
                        if not text.strip():
                            continue
                        
                        # Detect language for this page
                        lang_info = self.language_detector.detect_language(text)
                        detected_language = lang_info['language']
                        
                        # Preprocess text based on detected language
                        processed_text = self.preprocess_text(text, detected_language)
                        
                        if not processed_text.strip():
                            continue
                        
                        doc = Document(
                            text=processed_text,
                            metadata={
                                "file_name": file_name,
                                "page_number": page_num,
                                "source": file_path,
                                "language": detected_language,
                                "language_confidence": lang_info['confidence'],
                                "is_mixed_language": lang_info['mixed'],
                                "arabic_ratio": lang_info.get('arabic_ratio', 0),
                                "english_ratio": lang_info.get('english_ratio', 0)
                            }
                        )
                        documents.append(doc)
                        
                        logger.info(f"Page {page_num}: {detected_language} "
                                  f"(confidence: {lang_info['confidence']:.2f})")
                        
                    except Exception as e:
                        logger.warning(f"Error processing page {page_num}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise
        
        logger.info(f"Extracted {len(documents)} pages from {file_name}")
        return documents
    
    def get_appropriate_parser(self, language: str) -> SentenceSplitter:
        """Get the appropriate parser based on language"""
        if language == "arabic":
            return self.arabic_parser
        elif language == "english":
            return self.english_parser
        else:
            return self.default_parser
    
    def process_document(self, file_path: str) -> List[Document]:
        """Process a document with language-aware chunking"""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension != '.pdf':
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        # Extract documents with language detection
        documents = self.extract_text_from_pdf(file_path)
        
        if not documents:
            logger.warning(f"No text extracted from {file_path}")
            return []
        
        # Process each document with appropriate parser
        all_nodes = []
        language_stats = {"arabic": 0, "english": 0, "mixed": 0, "unknown": 0}
        
        for doc in documents:
            doc_language = doc.metadata.get('language', 'unknown')
            language_stats[doc_language] += 1
            
            # Get appropriate parser
            parser = self.get_appropriate_parser(doc_language)
            
            # Split document into nodes
            doc_nodes = parser.get_nodes_from_documents([doc])
            
            # Enhance nodes with language information
            for node in doc_nodes:
                node.metadata.update(doc.metadata)
                # Re-detect language for the chunk (might be more accurate)
                chunk_lang_info = self.language_detector.detect_language(node.text)
                node.metadata['chunk_language'] = chunk_lang_info['language']
                node.metadata['chunk_confidence'] = chunk_lang_info['confidence']
            
            all_nodes.extend(doc_nodes)
        
        # Log language distribution
        total_pages = sum(language_stats.values())
        logger.info(f"Language distribution for {Path(file_path).name}:")
        for lang, count in language_stats.items():
            if count > 0:
                percentage = (count / total_pages) * 100
                logger.info(f"  {lang}: {count} pages ({percentage:.1f}%)")
        
        logger.info(f"Created {len(all_nodes)} nodes from {len(documents)} pages")
        return all_nodes
    
    def get_processing_stats(self, nodes: List[Document]) -> Dict[str, Any]:
        """Get statistics about processed documents"""
        stats = {
            "total_nodes": len(nodes),
            "languages": {"arabic": 0, "english": 0, "mixed": 0, "unknown": 0},
            "avg_chunk_size": 0,
            "files": set()
        }
        
        total_chars = 0
        for node in nodes:
            lang = node.metadata.get('chunk_language', 'unknown')
            stats['languages'][lang] += 1
            total_chars += len(node.text)
            stats['files'].add(node.metadata.get('file_name', 'unknown'))
        
        if nodes:
            stats['avg_chunk_size'] = total_chars // len(nodes)
        
        stats['files'] = list(stats['files'])
        return stats

if __name__ == "__main__":
    # Test the bilingual processor
    processor = BilingualDocumentProcessor()
    
    # Test language detection
    detector = LanguageDetector()
    
    test_texts = [
        "This is an English document about data protection laws.",
        "هذه وثيقة باللغة العربية حول قوانين حماية البيانات الشخصية.",
        "This document contains both English and Arabic text. هذا النص يحتوي على اللغتين.",
        "123 Numbers and symbols !@#$"
    ]
    
    for text in test_texts:
        result = detector.detect_language(text)
        print(f"Text: {text[:50]}...")
        print(f"Language: {result['language']} (confidence: {result['confidence']:.2f})")
        print()