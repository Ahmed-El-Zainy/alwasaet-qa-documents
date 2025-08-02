import os
import sys
import logging
from pathlib import Path
import yaml
# from dotenv import load_dotenv


# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

# print(f"SCRIPT_DIR: {SCRIPT_DIR}")
# print(f"os.path.dirname(SCRIPT_DIR): {os.path.dirname(SCRIPT_DIR)}")
# print(f"os.path.dirname(os.path.dirname(SCRIPT_DIR)): {os.path.dirname(os.path.dirname(SCRIPT_DIR))}")

from utils.document_processor import DocumentProcessor
from utils.vector_store import VectorStoreManager
from utils.agent import AgenticRAG

# Load environment variables
# load_dotenv()


try:
    from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("main pipeline")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("main pipeline")
    logger.info("Using standard logger - custom logger not available")


class RAGApplication:
    """Main RAG Application class with Gemini integration"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.doc_processor = DocumentProcessor(config_path)
        self.vector_manager = VectorStoreManager(config_path)
        self.agent = AgenticRAG(config_path)
        
        self.index = None
        
        logger.info("RAG Application initialized with Gemini")
    
    def process_documents(self, file_paths: list) -> bool:
        """Process and index documents"""
        try:
            all_nodes = []
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    logger.warning(f"File not found: {file_path}")
                    continue
                
                logger.info(f"Processing document: {file_path}")
                nodes = self.doc_processor.process_document(file_path)
                all_nodes.extend(nodes)
                logger.info(f"Processed {len(nodes)} nodes from {file_path}")
            
            if not all_nodes:
                logger.error("No nodes were processed from the documents")
                return False
            
            # Create index
            logger.info(f"Creating index with {len(all_nodes)} total nodes")
            self.index = self.vector_manager.create_index(all_nodes)
            
            # Setup agent
            self.agent.setup(self.index)
            
            logger.info("Document processing and indexing complete")
            return True
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return False
    
    def query(self, question: str) -> dict:
        """Query the RAG system"""
        if not self.index:
            return {
                "answer": "No documents have been processed yet. Please upload documents first.",
                "citations": [],
                "source_nodes": []
            }
        
        return self.agent.process_query(question)
    
    def clear_memory(self):
        """Clear conversation memory"""
        self.agent.clear_memory()
    
    def reset_system(self):
        """Reset the entire system"""
        self.vector_manager.clear_collection()
        self.agent.clear_memory()
        self.index = None
        logger.info("System reset complete")

def main():
    """Main function for testing"""
    # Check environment variables
    required_vars = ["GOOGLE_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing environment variables: {missing_vars}")
        return
    
    # Initialize application
    app = RAGApplication()
    
    # Example usage
    print("RAG Application initialized successfully with Gemini!")
    print("Use the Gradio interface to interact with the system.")

if __name__ == "__main__":
    main()
    # Test the code with example in the following api of gemini
    app = RAGApplication()
    app.process_documents(["assets/Documents/2020-october-ksa-transparency-report-saudi-arabia-ar.pdf"])
    result = app.query("can you tell me about the Saudi Arabia transparency report?")
    print(result)
    app.clear_memory()
    app.reset_system()




# """
# import logging

# from core.model_runtime.entities.model_entities import ModelType
# from core.model_runtime.errors.validate import CredentialsValidateFailedError
# from core.model_runtime.model_providers.__base.model_provider import ModelProvider

# logger = logging.getLogger(__name__)


# class GoogleProvider(ModelProvider):
#     def validate_provider_credentials(self, credentials: dict) -> None:
#         """
#         Validate provider credentials.

#         If validation fails, raise an exception.

#         :param credentials: provider credentials defined in `provider_credential_schema`.
#         """
#         try:
#             model_instance = self.get_model_instance(ModelType.LLM)

#             # Use `gemini-1.5-pro` for validation instead of the default.
#             model_instance.validate_credentials(model="gemini-1.5-pro",
#                                                 credentials=credentials)
#         except CredentialsValidateFailedError as ex:
#             raise ex
#         except Exception as ex:
#             logger.exception(
#                 f"{self.get_provider_schema().provider} credentials validation failed"
#             )
#             raise ex
# """