import logging
from typing import List, Dict, Any, Optional
import yaml
import os
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core import Response
from llama_index.llms.gemini import Gemini
import sys 
import os 

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
    """Retrieval-Augmented Generation Engine with Gemini"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.similarity_top_k = self.config['retrieval']['similarity_top_k']
        self.similarity_threshold = self.config['retrieval']['similarity_threshold']
        
        # Initialize Gemini LLM
        logger.info(f"Loading Gemini configuration: {self.config['gemini']['llm']}")
        model_name = self.config['gemini']['llm']['model']
        logger.info(f"Using model: {model_name}")
        
        self.llm = Gemini(
            model=model_name,
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=self.config['gemini']['llm']['temperature'],
            max_tokens=self.config['gemini']['llm']['max_output_tokens'],
            safety_settings=[
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
                },
            ]
        )
        
        self.index = None
        self.query_engine = None
    
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
        
        logger.info("RAG engine setup complete with Gemini")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG engine"""
        if not self.query_engine:
            raise ValueError("Query engine not initialized. Call setup_engine first.")
        
        try:
            response = self.query_engine.query(question)
            
            # Extract citations
            citations = self._extract_citations(response)
            
            # Check if answer was found
            if not response.response or "no answer found" in response.response.lower() or "sorry" in response.response.lower():
                return {
                    "answer": "No answer found",
                    "citations": [],
                    "source_nodes": []
                }
            
            return {
                "answer": response.response,
                "citations": citations,
                "source_nodes": response.source_nodes
            }
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return {
                "answer": "Error processing query",
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
    


if __name__=="__main__":
    logger.info("test....")