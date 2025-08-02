import logging
from typing import List, Optional
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.core.schema import BaseNode
import os 
import sys 

# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

try:
    from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("vector store")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("vector store")
    logger.info("Using standard logger - custom logger not available")

class VectorStoreManager:
    """Manages vector store operations with Gemini embeddings"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.collection_name = self.config['vector_store']['collection_name']
        self.vector_size = self.config['vector_store']['vector_size']
        self.host = self.config['vector_store']['host']
        self.port = self.config['vector_store']['port']
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=self.host,
            port=self.port
        )
        
        # Initialize Gemini embedding model with proper model name
        logger.info(f"Loading Gemini embeddings configuration: {self.config['huggingface']['embeddings']}")
        # embedding_model = self.config['huggingface']['embeddings']['model']
        embedding_model = "models/text-embedding-004"
        
        # Validate embedding model name
        valid_embedding_models = [
            "models/text-embedding-004",
            "models/embedding-001"
        ]
        
        if embedding_model not in valid_embedding_models:
            logger.warning(f"Invalid embedding model '{embedding_model}', falling back to 'models/text-embedding-004'")
            embedding_model = "models/text-embedding-004"
            
        logger.info(f"Using embedding model: {embedding_model}")
        
        try:
            self.embed_model = GeminiEmbedding(
                model_name=embedding_model,
                api_key=os.getenv("GOOGLE_API_KEY"),
                title="Agentic RAG Embeddings"
            )
            logger.info(f"Successfully initialized Gemini embeddings with: {embedding_model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings with {embedding_model}: {str(e)}")
            # Try fallback
            fallback_model = "models/embedding-001"
            try:
                self.embed_model = GeminiEmbedding(
                    model_name=fallback_model,
                    api_key=os.getenv("GOOGLE_API_KEY"),
                    title="Agentic RAG Embeddings"
                )
                logger.info(f"Successfully initialized Gemini embeddings with fallback: {fallback_model}")
                # Update vector size for fallback model
                self.vector_size = 768  # embedding-001 uses 768 dimensions
            except Exception as fallback_error:
                logger.error(f"Failed to initialize Gemini embeddings with fallback: {str(fallback_error)}")
                raise
        
        # Create collection if it doesn't exist
        self._create_collection()
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name
        )
    
    def _create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name} with vector size: {self.vector_size}")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
                
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def create_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """Create vector store index from nodes"""
        try:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True
            )
            
            logger.info(f"Created index with {len(nodes)} nodes")
            return index
            
        except Exception as e:
            logger.error(f"Error creating index: {str(e)}")
            raise
    
    def load_index(self) -> Optional[VectorStoreIndex]:
        """Load existing index from vector store"""
        try:
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                embed_model=self.embed_model
            )
            
            logger.info("Loaded existing index")
            return index
            
        except Exception as e:
            logger.warning(f"Could not load existing index: {str(e)}")
            return None
    
    def clear_collection(self):
        """Clear all vectors from collection"""
        try:
            self.client.delete_collection(self.collection_name)
            self._create_collection()
            logger.info(f"Cleared collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")

if __name__=="__main__":
    logger.info("test...")