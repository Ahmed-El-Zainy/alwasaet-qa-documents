import logging
from typing import List, Optional
import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import BaseNode
import os 
import sys 

# Add path for custom logger
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
    """Manages vector store operations with Hugging Face embeddings"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.collection_name = self.config['vector_store']['collection_name']
        self.host = self.config['vector_store']['host']
        self.port = self.config['vector_store']['port']
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            host=self.host,
            port=self.port
        )
        
        # Initialize Hugging Face embedding model
        self._initialize_embeddings()
        
        # Create collection with correct dimensions
        self._create_collection()
        
        # Initialize vector store
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name
        )
    
    def _initialize_embeddings(self):
        """Initialize Hugging Face embedding model with automatic dimension detection"""
        # Get embedding configuration
        embedding_config = self.config.get('huggingface', {}).get('embeddings', {})
        model_name = embedding_config.get('model', 'sentence-transformers/all-MiniLM-L6-v2')
        device = embedding_config.get('device', 'cpu')
        
        # Handle 'auto' device selection more safely
        if device == 'auto':
            try:
                import torch
                if torch.cuda.is_available():
                    device = 'cuda'
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    device = 'mps'  # Apple Silicon
                else:
                    device = 'cpu'
                logger.info(f"Auto-detected device: {device}")
            except Exception as e:
                logger.warning(f"Device auto-detection failed: {e}, using CPU")
                device = 'cpu'
        
        logger.info(f"Initializing Hugging Face embeddings: {model_name}")
        logger.info(f"Using device: {device}")
        
        try:
            # Initialize the embedding model with proper device handling
            self.embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                device=device,
                trust_remote_code=True
            )
            
            # Get the actual embedding dimension by testing with sample text
            test_embedding = self.embed_model.get_text_embedding("test")
            self.vector_size = len(test_embedding)
            
            logger.info(f"Successfully initialized Hugging Face embeddings")
            logger.info(f"Model: {model_name}")
            logger.info(f"Vector dimension: {self.vector_size}")
            logger.info(f"Device: {device}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hugging Face embeddings: {str(e)}")
            # Fallback to CPU with simple model
            logger.info("Falling back to all-MiniLM-L6-v2 model on CPU")
            try:
                self.embed_model = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    device="cpu"
                )
                test_embedding = self.embed_model.get_text_embedding("test")
                self.vector_size = len(test_embedding)
                logger.info(f"Fallback successful. Vector dimension: {self.vector_size}")
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                raise RuntimeError("Could not initialize any embedding model")
    
    def _create_collection(self):
        """Create Qdrant collection with correct vector dimensions"""
        try:
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                # Check if existing collection has correct dimensions
                collection_info = self.client.get_collection(self.collection_name)
                existing_size = collection_info.config.params.vectors.size
                
                if existing_size != self.vector_size:
                    logger.warning(f"Existing collection has wrong vector size ({existing_size} vs {self.vector_size})")
                    logger.info("Deleting and recreating collection with correct dimensions")
                    self.client.delete_collection(self.collection_name)
                    collection_names.remove(self.collection_name)
                else:
                    logger.info(f"Collection {self.collection_name} exists with correct dimensions ({self.vector_size})")
                    return
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection: {self.collection_name} with vector size: {self.vector_size}")
                
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def create_index(self, nodes: List[BaseNode]) -> VectorStoreIndex:
        """Create vector store index from nodes"""
        try:
            # Ensure we have the correct collection
            self._create_collection()
            
            storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            
            # Create index with progress tracking
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model,
                show_progress=True
            )
            
            logger.info(f"Successfully created index with {len(nodes)} nodes")
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
            
            logger.info("Successfully loaded existing index")
            return index
            
        except Exception as e:
            logger.warning(f"Could not load existing index: {str(e)}")
            return None
    
    def clear_collection(self):
        """Clear all vectors from collection"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
            self._create_collection()
            logger.info("Recreated empty collection")
        except Exception as e:
            logger.error(f"Error clearing collection: {str(e)}")
    
    def get_embedding_info(self) -> dict:
        """Get information about the embedding configuration"""
        return {
            "provider": "huggingface",
            "model": self.embed_model.model_name,
            "vector_size": self.vector_size,
            "collection_name": self.collection_name
        }
    
    def test_embedding(self, text: str = "This is a test") -> dict:
        """Test embedding generation"""
        try:
            embedding = self.embed_model.get_text_embedding(text)
            return {
                "success": True,
                "dimension": len(embedding),
                "sample_values": embedding[:5]  # First 5 values
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

if __name__ == "__main__":
    # Test the vector store manager
    try:
        manager = VectorStoreManager("config.yaml")
        info = manager.get_embedding_info()
        print(f"Embedding Info: {info}")
        
        # Test embedding
        test_result = manager.test_embedding()
        print(f"Embedding Test: {test_result}")
        
    except Exception as e:
        print(f"Error: {e}")