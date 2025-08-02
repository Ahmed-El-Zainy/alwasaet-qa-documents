#!/usr/bin/env python3
"""
Test script to verify Hugging Face embeddings setup
Run this after updating your configuration
"""

import os
import sys
import logging

# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))


try:
    from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("test embedding model")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("test embedding model")
    logger.info("Using standard logger - custom logger not available")



def test_config():
    """Test configuration loading"""
    try:
        import yaml
        with open('src/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        embedding_provider = config.get('embedding_provider', 'unknown')
        vector_size = config.get('vector_store', {}).get('vector_size', 'unknown')
        model = config.get('huggingface', {}).get('embeddings', {}).get('model', 'unknown')
        
        logger.info(f"✅ Config loaded successfully")
        logger.info(f"📊 Embedding provider: {embedding_provider}")
        logger.info(f"📊 Vector size: {vector_size}")
        logger.info(f"📊 HF model: {model}")
        
        return embedding_provider == 'huggingface'
        
    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        return False

def test_vector_store():
    """Test vector store initialization"""
    try:
        from utils.vector_store import VectorStoreManager
        
        manager = VectorStoreManager()
        info = manager.get_embedding_info()
        
        logger.info(f"✅ Vector store initialized successfully")
        logger.info(f"📊 Embedding info: {info}")
        
        return info['provider'] == 'huggingface'
        
    except Exception as e:
        logger.error(f"❌ Vector store test failed: {e}")
        return False

def test_rag_engine():
    """Test RAG engine initialization"""
    try:
        from utils.rag_engine import RAGEngine
        
        engine = RAGEngine()
        info = engine.get_model_info()
        
        logger.info(f"✅ RAG engine initialized successfully")
        logger.info(f"📊 Model info: {info}")
        
        expected = info['embedding_provider'] == 'huggingface' and info['llm_provider'] == 'gemini'
        return expected
        
    except Exception as e:
        logger.error(f"❌ RAG engine test failed: {e}")
        return False

def test_embedding_generation():
    """Test actual embedding generation"""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="auto"
        )
        
        test_texts = [
            "This is a test document about machine learning",
            "Artificial intelligence and natural language processing",
            "Vector embeddings and semantic search"
        ]
        
        embeddings = []
        for text in test_texts:
            embedding = embed_model.get_text_embedding(text)
            embeddings.append(embedding)
        
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        logger.info(f"📊 Embedding dimension: {len(embeddings[0])}")
        
        # Test similarity
        import numpy as np
        sim1 = np.dot(embeddings[0], embeddings[1])
        sim2 = np.dot(embeddings[0], embeddings[2])
        
        logger.info(f"📊 Similarity scores: {sim1:.3f}, {sim2:.3f}")
        
        return len(embeddings[0]) == 384  # Expected dimension for all-MiniLM-L6-v2
        
    except Exception as e:
        logger.error(f"❌ Embedding generation test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing Hugging Face Embeddings Setup")
    print("=" * 45)
    
    tests = [
        ("Configuration", test_config),
        ("Vector Store", test_vector_store),
        ("RAG Engine", test_rag_engine),
        ("Embedding Generation", test_embedding_generation)
    ]
    
    results = {}
    for name, test_func in tests:
        print(f"\n🔍 Testing {name}...")
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"❌ {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    print("\n📊 Test Results:")
    print("-" * 30)
    passed = 0
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Summary: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\n🚀 You can now:")
        print("1. Upload documents to your RAG system")
        print("2. The system will use Hugging Face embeddings for indexing")
        print("3. Gemini will generate responses based on retrieved context")
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)