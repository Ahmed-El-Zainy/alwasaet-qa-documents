import logging
from typing import List, Dict, Any, Optional
import yaml
import os
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Response
# from llama_index.llms.gemini import Gemini
import sys 
import os 
import google.generativeai as genai

# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

try:
    from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("rag engine")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("rag engine")
    logger.info("Using standard logger - custom logger not available")

class RAGEngine:
    """Retrieval-Augmented Generation Engine with Gemini LLM and Hugging Face Embeddings"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.similarity_top_k = self.config['retrieval']['similarity_top_k']
        self.similarity_threshold = self.config['retrieval']['similarity_threshold']
        
        # Initialize Gemini LLM (keeping Gemini for generation)
        self._initialize_llm()
        
        self.index = None
        self.query_engine = None
    
    def _initialize_llm(self):
        """Initialize Gemini LLM"""
        gemini_config = self.config.get('gemini', {}).get('llm', {})
        model_name = gemini_config.get('model', 'gemini-1.5-flash')
        
        # Validate model name
        valid_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro", 
            "gemini-pro"
        ]
        
        if model_name not in valid_models:
            logger.warning(f"Invalid model '{model_name}', falling back to 'gemini-1.5-flash'")
            model_name = "models/gemini-1.5-flash"
        
        logger.info(f"Loading Gemini LLM configuration: {gemini_config}")
        logger.info(f"Using Gemini model: {model_name}")
        
        # Check for API key
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini LLM")
        
        try:
            self.llm = genai.GenerativeModel(
                model_name)
            
            # Gemini(
            #     model=model_name,
            #     api_key=api_key,
            #     temperature=gemini_config.get('temperature', 0.1),
            #     max_tokens=gemini_config.get('max_output_tokens', 2048)
            # )
            logger.info(f"Successfully initialized Gemini LLM: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini with {model_name}: {str(e)}")
            # Try fallback model
            fallback_model = "gemini-1.5-flash"
            if model_name != fallback_model:
                logger.info(f"Trying fallback model: {fallback_model}")
                try:
                    self.llm = genai.GenerativeModel(
                        model=fallback_model
                        # api_key=api_key,
                        # temperature=gemini_config.get('temperature', 0.1),
                        # max_tokens=gemini_config.get('max_output_tokens', 2048)
                    )
                    logger.info(f"Successfully initialized Gemini with fallback model: {fallback_model}")
                except Exception as fallback_error:
                    logger.error(f"Failed to initialize Gemini with fallback model: {str(fallback_error)}")
                    raise
            else:
                raise
    
    def setup_engine(self, index: VectorStoreIndex):
        """Setup the query engine with the provided index"""
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
        
        # Create query engine with custom prompt
        from llama_index.core.prompts import PromptTemplate
        
        qa_prompt = PromptTemplate(
            "Context information is below.\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Given the context information and not prior knowledge, "
            "answer the query. If the context doesn't contain relevant information "
            "to answer the query, respond with 'No answer found'.\n"
            "Be precise and cite specific information from the context when possible.\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        
        # Create query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            node_postprocessors=[postprocessor],
            llm=self.llm
        )
        
        # Update the query engine's prompt
        self.query_engine.update_prompts({"response_synthesizer:text_qa_template": qa_prompt})
        
        # Log the setup information
        embedding_provider = self.config.get('embedding_provider', 'huggingface')
        embedding_model = self.config.get('huggingface', {}).get('embeddings', {}).get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        
        logger.info(f"RAG engine setup complete:")
        logger.info(f"  - LLM: Gemini ({self.config.get('gemini', {}).get('llm', {}).get('model', 'gemini-1.5-flash')})")
        logger.info(f"  - Embeddings: {embedding_provider} ({embedding_model})")
        logger.info(f"  - Vector DB: Qdrant")
        logger.info(f"  - Similarity threshold: {self.similarity_threshold}")
        logger.info(f"  - Top-k retrieval: {self.similarity_top_k}")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG engine"""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call setup_engine first.")
        
        try:
            logger.info(f"Processing query: {question}")
            response = self.query_engine.query(question)
            
            # Extract citations
            citations = self._extract_citations(response)
            
            # Check if answer was found
            answer_text = response.response if response.response else ""
            if not answer_text or "no answer found" in answer_text.lower() or "sorry" in answer_text.lower():
                logger.info("No relevant answer found in documents")
                return {
                    "answer": "No answer found in the provided documents.",
                    "citations": [],
                    "source_nodes": []
                }
            
            logger.info(f"Found answer with {len(citations)} citations")
            return {
                "answer": answer_text,
                "citations": citations,
                "source_nodes": response.source_nodes
            }
            
        except Exception as e:
            logger.error(f"Error during query processing: {str(e)}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "citations": [],
                "source_nodes": []
            }
    
    def _extract_citations(self, response: Response) -> List[Dict[str, Any]]:
        """Extract citations from response source nodes"""
        citations = []
        
        for node in response.source_nodes:
            citation = {
                "file_name": node.metadata.get("file_name", "Unknown"),
                "page_number": node.metadata.get("page_number", "Unknown"),
                "score": node.score if hasattr(node, 'score') else None,
                "text_snippet": node.text[:200] + "..." if len(node.text) > 200 else node.text
            }
            citations.append(citation)
        
        return citations
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the models being used"""
        embedding_provider = self.config.get('embedding_provider', 'huggingface')
        if embedding_provider == 'huggingface':
            embedding_model = self.config.get('huggingface', {}).get('embeddings', {}).get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        else:
            embedding_model = self.config.get('gemini', {}).get('embeddings', {}).get('model', 'models/embedding-001')
        
        return {
            "llm_provider": "gemini",
            "llm_model": self.config.get('gemini', {}).get('llm', {}).get('model', 'gemini-1.5-flash'),
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "vector_store": "qdrant"
        }

if __name__=="__main__":
    logger.info("test....")
    # Test the RAG engine
    try:
        engine = RAGEngine("src/config.yaml")
        info = engine.get_model_info()
        print(f"Model Info: {info}")
    except Exception as e:
        print(f"Error: {e}")