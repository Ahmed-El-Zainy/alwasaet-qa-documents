#!/usr/bin/env python3
"""
Setup script for Hugging Face embeddings integration
This script will install required packages and test the embedding model
"""

import os
import sys
import subprocess
import logging


# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))


try:
    from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("test huggingface")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("test huggingface")
    logger.info("Using standard logger - custom logger not available")




def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install {package}: {e}")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    packages_to_test = [
        ("sentence_transformers", "sentence-transformers"),
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("llama_index.embeddings.huggingface", "llama-index-embeddings-huggingface")
    ]
    
    success = True
    for module, package in packages_to_test:
        try:
            __import__(module)
            logger.info(f"✅ {package} imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import {module}: {e}")
            success = False
    
    return success

def test_embedding_model():
    """Test the Hugging Face embedding model"""
    try:
        from sentence_transformers import SentenceTransformer
        
        # Test loading the model
        logger.info("Loading sentence-transformers/all-MiniLM-L6-v2...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test embedding generation
        test_texts = ["Hello world", "This is a test document"]
        embeddings = model.encode(test_texts)
        
        logger.info(f"✅ Model loaded successfully!")
        logger.info(f"✅ Generated embeddings with shape: {embeddings.shape}")
        logger.info(f"✅ Embedding dimension: {embeddings.shape[1]}")
        
        return embeddings.shape[1]  # Return dimension
        
    except Exception as e:
        logger.error(f"❌ Failed to test embedding model: {e}")
        return None

def test_llama_index_integration():
    """Test LlamaIndex Hugging Face embedding integration"""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        # Initialize the embedding model
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="auto"
        )
        
        # Test embedding
        test_text = "This is a test for LlamaIndex integration"
        embedding = embed_model.get_text_embedding(test_text)
        
        logger.info(f"✅ LlamaIndex integration working!")
        logger.info(f"✅ Embedding dimension: {len(embedding)}")
        
        return len(embedding)
        
    except Exception as e:
        logger.error(f"❌ LlamaIndex integration failed: {e}")
        return None

def setup_environment():
    """Setup the environment for Hugging Face embeddings"""
    logger.info("🚀 Setting up Hugging Face embeddings environment...")
    
    # Required packages
    packages = [
        "sentence-transformers>=2.2.2",
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "llama-index-embeddings-huggingface>=0.2.0"
    ]
    
    # Install packages
    logger.info("📦 Installing required packages...")
    all_installed = True
    for package in packages:
        if not install_package(package):
            all_installed = False
    
    if not all_installed:
        logger.error("❌ Some packages failed to install")
        return False
    
    # Test imports
    logger.info("🧪 Testing package imports...")
    if not test_imports():
        logger.error("❌ Some imports failed")
        return False
    
    # Test embedding model
    logger.info("🤖 Testing embedding model...")
    embedding_dim = test_embedding_model()
    if embedding_dim is None:
        logger.error("❌ Embedding model test failed")
        return False
    
    # Test LlamaIndex integration
    logger.info("🔗 Testing LlamaIndex integration...")
    llama_dim = test_llama_index_integration()
    if llama_dim is None:
        logger.error("❌ LlamaIndex integration test failed")
        return False
    
    # Verify dimensions match
    if embedding_dim != llama_dim:
        logger.warning(f"⚠️ Dimension mismatch: SentenceTransformers={embedding_dim}, LlamaIndex={llama_dim}")
    else:
        logger.info(f"✅ Embedding dimensions match: {embedding_dim}")
    
    logger.info("🎉 Setup completed successfully!")
    logger.info(f"📊 Embedding dimension: {embedding_dim}")
    logger.info("💡 Update your config.yaml vector_size to: {}".format(embedding_dim))
    
    return True

def check_existing_setup():
    """Check if setup is already complete"""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
        test_embedding = embed_model.get_text_embedding("test")
        logger.info("✅ Hugging Face embeddings already set up correctly!")
        logger.info(f"📊 Embedding dimension: {len(test_embedding)}")
        return len(test_embedding)
    except Exception:
        return None

def main():
    """Main setup function"""
    print("🤗 Hugging Face Embeddings Setup for Agentic RAG")
    print("=" * 50)
    
    # Check if already set up
    existing_dim = check_existing_setup()
    if existing_dim:
        print(f"✅ Already set up! Embedding dimension: {existing_dim}")
        print("💡 Make sure your config.yaml vector_size is set to:", existing_dim)
        return
    
    # Run setup
    if setup_environment():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Update your config.yaml with the provided configuration")
        print("2. Replace vector_store.py and rag_engine.py with the updated versions")
        print("3. Clear existing vector data: rm -rf qdrant_storage/")
        print("4. Restart your application")
        print("\n🚀 You're now using Hugging Face embeddings with Gemini LLM!")
    else:
        print("❌ Setup failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()