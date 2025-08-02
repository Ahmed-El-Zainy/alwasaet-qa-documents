import logging
from typing import List, Dict, Any, Optional
import yaml
import os
import re
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Response
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.prompts import PromptTemplate
import google.generativeai as genai
import sys 
import os 

# Add path for custom logger
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

try:
    from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("bilingual_rag_engine")
    logger.info("Custom Logger Start Working.....")
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("bilingual rag engine")
    logger.info("Using standard logger - custom logger not available")


class LanguageDetector:
    """Detect query language for appropriate processing"""
    
    def __init__(self):
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
        self.english_pattern = re.compile(r'[a-zA-Z]+')
    
    def detect_query_language(self, query: str) -> str:
        """Detect the primary language of a query"""
        arabic_chars = len(self.arabic_pattern.findall(query))
        english_chars = len(self.english_pattern.findall(query))
        total_chars = arabic_chars + english_chars
        
        if total_chars == 0:
            return "english"  # Default fallback
        
        arabic_ratio = arabic_chars / total_chars
        
        if arabic_ratio > 0.3:  # If more than 30% Arabic, consider it Arabic
            return "arabic"
        else:
            return "english"


class BilingualGeminiLLM(CustomLLM):
    """Enhanced Gemini LLM with bilingual support"""
    
    model_name: str = "gemini-1.5-flash"
    temperature: float = 0.1
    max_tokens: int = 2048
    
    def __init__(self, model_name: str = "gemini-1.5-flash", temperature: float = 0.1, max_tokens: int = 2048):
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure the API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        genai.configure(api_key=api_key)
        
        # Initialize the model with bilingual instructions
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        
        # Bilingual system instruction
        system_instruction = """You are a bilingual AI assistant that can understand and respond in both Arabic and English.

Guidelines:
- If the question is in Arabic, respond in Arabic
- If the question is in English, respond in English  
- If the context contains both languages, you may use the language of the question
- Always provide accurate information based on the given context
- When citing sources, include page numbers and file names
- If no relevant information is found, clearly state this in the appropriate language

Arabic response format:
بناءً على الوثائق المتوفرة: [الإجابة]

English response format:
Based on the available documents: [Answer]"""
        
        self.model = genai.GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            # system_instruction=system_instruction
        )
        
        logger.info(f"Initialized Bilingual Gemini model: {self.model_name}")
    
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=1000000,  # Gemini 1.5 has large context window
            num_output=self.max_tokens,
            model_name=self.model_name,
        )
    
    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Generate completion for the given prompt."""
        try:
            response = self.model.generate_content(prompt)
            
            if response.parts and response.text:
                text = response.text
            else:
                # Try to extract any available text
                text = "No response generated"
                if hasattr(response, 'candidates') and response.candidates:
                    for candidate in response.candidates:
                        if hasattr(candidate, 'content') and candidate.content.parts:
                            text = candidate.content.parts[0].text
                            break
                
            return CompletionResponse(text=text)
            
        except Exception as e:
            logger.error(f"Error generating completion: {str(e)}")
            return CompletionResponse(text=f"Error: {str(e)}")
    
    @llm_completion_callback()
    def stream_complete(self, prompt: str, **kwargs: Any):
        """Stream completion for the given prompt."""
        try:
            response = self.model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.parts and chunk.text:
                    yield CompletionResponse(text=chunk.text, delta=chunk.text)
                    
        except Exception as e:
            logger.error(f"Error in stream completion: {str(e)}")
            yield CompletionResponse(text=f"Error: {str(e)}")


class BilingualRAGEngine:
    """Enhanced RAG Engine with full bilingual support"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.similarity_top_k = self.config['retrieval']['similarity_top_k']
        self.similarity_threshold = self.config['retrieval']['similarity_threshold']
        
        # Initialize language detector
        self.language_detector = LanguageDetector()
        
        # Get language-specific thresholds
        self.lang_config = self.config.get('language_support', {})
        self.arabic_threshold = self.lang_config.get('arabic', {}).get('similarity_threshold', 0.25)
        self.english_threshold = self.lang_config.get('english', {}).get('similarity_threshold', 0.15)
        
        # Initialize Bilingual Gemini LLM
        self._initialize_llm()
        
        self.index = None
        self.query_engine = None
    
    def _initialize_llm(self):
        """Initialize Bilingual Gemini LLM"""
        gemini_config = self.config.get('gemini', {}).get('llm', {})
        model_name = gemini_config.get('model', 'gemini-1.5-flash')
        
        # Clean model name
        if model_name.startswith("models/"):
            model_name = model_name.replace("models/", "")
        
        # Valid Gemini models
        valid_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "gemini-pro",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro-latest"
        ]
        
        if model_name not in valid_models:
            logger.warning(f"Invalid model '{model_name}', falling back to 'gemini-1.5-flash'")
            model_name = "gemini-1.5-flash"
        
        logger.info(f"Using Bilingual Gemini model: {model_name}")
        
        try:
            self.llm = BilingualGeminiLLM(
                model_name=model_name,
                temperature=gemini_config.get('temperature', 0.1),
                max_tokens=gemini_config.get('max_output_tokens', 2048)
            )
            logger.info(f"Successfully initialized Bilingual Gemini LLM: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini with {model_name}: {str(e)}")
            raise
    
    def _create_bilingual_prompt(self, query_language: str) -> PromptTemplate:
        """Create language-appropriate prompt template"""
        
        if query_language == "arabic":
            prompt_text = """معلومات السياق أدناه.
---------------------
{context_str}
---------------------
بناءً على معلومات السياق المذكورة أعلاه وليس على المعرفة المسبقة، أجب على السؤال.
إذا كان السياق لا يحتوي على معلومات ذات صلة للإجابة على السؤال، أجب بـ "لا توجد إجابة في الوثائق المتوفرة".
كن دقيقاً واستشهد بمعلومات محددة من السياق عند الإمكان.
السؤال: {query_str}
الإجابة: """
        else:
            prompt_text = """Context information is below.
---------------------
{context_str}
---------------------
Given the context information and not prior knowledge, answer the query.
If the context doesn't contain relevant information to answer the query, respond with 'No answer found in the provided documents'.
Be precise and cite specific information from the context when possible.
Include relevant details and maintain accuracy.
Query: {query_str}
Answer: """
        
        return PromptTemplate(prompt_text)
    
    def setup_engine(self, index: VectorStoreIndex):
        """Setup the bilingual query engine"""
        self.index = index
        
        # Configure retriever
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=self.similarity_top_k
        )
        
        # Configure postprocessor
        postprocessor = SimilarityPostprocessor(
            similarity_cutoff=self.similarity_threshold
        )
        
        # Create default prompt (will be dynamically updated based on query language)
        default_prompt = self._create_bilingual_prompt("english")
        
        # Create response synthesizer
        response_synthesizer = get_response_synthesizer(
            llm=self.llm,
            text_qa_template=default_prompt
        )
        
        # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[postprocessor]
        )
        
        # Log setup information
        embedding_provider = self.config.get('embedding_provider', 'huggingface')
        embedding_model = self.config.get('huggingface', {}).get('embeddings', {}).get('model', 'unknown')
        
        logger.info(f"Bilingual RAG engine setup complete:")
        logger.info(f"  - LLM: Bilingual Gemini ({self.config.get('gemini', {}).get('llm', {}).get('model', 'gemini-1.5-flash')})")
        logger.info(f"  - Embeddings: {embedding_provider} ({embedding_model})")
        logger.info(f"  - Vector DB: Qdrant (bilingual)")
        logger.info(f"  - Arabic threshold: {self.arabic_threshold}")
        logger.info(f"  - English threshold: {self.english_threshold}")
        logger.info(f"  - Top-k retrieval: {self.similarity_top_k}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process bilingual query with language-aware optimization"""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call setup_engine first.")
        
        try:
            # Detect query language
            query_language = self.language_detector.detect_query_language(question)
            logger.info(f"Processing {query_language} query: {question}")
            
            # Adjust similarity threshold based on language
            current_threshold = self.arabic_threshold if query_language == "arabic" else self.english_threshold
            
            # Update postprocessor threshold dynamically
            if hasattr(self.query_engine, 'node_postprocessors'):
                for postprocessor in self.query_engine.node_postprocessors:
                    if isinstance(postprocessor, SimilarityPostprocessor):
                        postprocessor._similarity_cutoff = current_threshold
            
            # Create language-appropriate prompt
            bilingual_prompt = self._create_bilingual_prompt(query_language)
            
            # Update the response synthesizer prompt
            if hasattr(self.query_engine, 'response_synthesizer'):
                self.query_engine.response_synthesizer.update_prompts(
                    {"text_qa_template": bilingual_prompt}
                )
            
            # Execute query
            response = self.query_engine.query(question)
            
            # Extract citations with language information
            citations = self._extract_bilingual_citations(response)
            
            # Check if answer was found
            answer_text = response.response if response.response else ""
            
            # Language-specific no-answer detection
            no_answer_indicators = [
                "no answer found", "not found in the provided documents",
                "لا توجد إجابة", "غير موجود في الوثائق", "لا يحتوي السياق"
            ]
            
            if not answer_text or any(indicator in answer_text.lower() for indicator in no_answer_indicators):
                no_answer_msg = "لا توجد إجابة في الوثائق المتوفرة." if query_language == "arabic" else "No answer found in the provided documents."
                logger.info(f"No relevant answer found for {query_language} query")
                return {
                    "answer": no_answer_msg,
                    "citations": [],
                    "source_nodes": [],
                    "query_language": query_language,
                    "threshold_used": current_threshold
                }
            
            logger.info(f"Found answer with {len(citations)} citations for {query_language} query")
            return {
                "answer": answer_text,
                "citations": citations,
                "source_nodes": response.source_nodes,
                "query_language": query_language,
                "threshold_used": current_threshold
            }
            
        except Exception as e:
            logger.error(f"Error during bilingual query processing: {str(e)}")
            return {
                "answer": f"خطأ في معالجة الاستعلام: {str(e)}" if query_language == "arabic" else f"Error processing query: {str(e)}",
                "citations": [],
                "source_nodes": [],
                "query_language": query_language if 'query_language' in locals() else "unknown",
                "threshold_used": self.similarity_threshold
            }
    
    def _extract_bilingual_citations(self, response: Response) -> List[Dict[str, Any]]:
        """Extract citations with language information"""
        citations = []
        
        for node in response.source_nodes:
            citation = {
                "file_name": node.metadata.get("file_name", "Unknown"),
                "page_number": node.metadata.get("page_number", "Unknown"),
                "score": round(node.score, 3) if hasattr(node, 'score') and node.score else None,
                "text_snippet": node.text[:200] + "..." if len(node.text) > 200 else node.text,
                "language": node.metadata.get("language", "unknown"),
                "chunk_language": node.metadata.get("chunk_language", "unknown"),
                "language_confidence": node.metadata.get("language_confidence", 0.0)
            }
            citations.append(citation)
        
        return citations
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the bilingual models being used"""
        embedding_provider = self.config.get('embedding_provider', 'huggingface')
        embedding_model = self.config.get('huggingface', {}).get('embeddings', {}).get('model', 'unknown')
        
        return {
            "llm_provider": "gemini",
            "llm_model": self.config.get('gemini', {}).get('llm', {}).get('model', 'gemini-1.5-flash'),
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "vector_store": "qdrant",
            "architecture": "Bilingual: Arabic + English Support",
            "language_detection": "Automatic",
            "supported_languages": ["Arabic", "English", "Mixed"]
        }
    
    def validate_bilingual_setup(self) -> Dict[str, Any]:
        """Validate the bilingual RAG engine setup"""
        validation_result = {
            "llm_initialized": bool(self.llm),
            "index_available": bool(self.index),
            "query_engine_ready": bool(self.query_engine),
            "embedding_provider": self.config.get('embedding_provider', 'unknown'),
            "llm_provider": "gemini (bilingual)",
            "language_support": ["Arabic", "English"]
        }
        
        # Test LLM with both languages
        try:
            # Test English
            en_response = self.llm.complete("Hello, how are you?")
            validation_result["english_llm_test"] = "passed"
            
            # Test Arabic
            ar_response = self.llm.complete("مرحبا، كيف حالك؟")
            validation_result["arabic_llm_test"] = "passed"
            
        except Exception as e:
            validation_result["llm_test"] = f"failed: {str(e)}"
        
        return validation_result

if __name__ == "__main__":
    logger.info("Testing Bilingual RAG engine initialization...")
    try:
        engine = BilingualRAGEngine("src/config.yaml")
        info = engine.get_model_info()
        print(f"Model Info: {info}")
        
        validation = engine.validate_bilingual_setup()
        print(f"Validation: {validation}")
        
        # Test language detection
        detector = LanguageDetector()
        test_queries = [
            "What is this document about?",
            "ما هو موضوع هذه الوثيقة؟",
            "Tell me about البيانات الشخصية"
        ]
        
        for query in test_queries:
            lang = detector.detect_query_language(query)
            print(f"Query: {query}")
            print(f"Detected language: {lang}\n")
        
    except Exception as e:
        print(f"Error: {e}")