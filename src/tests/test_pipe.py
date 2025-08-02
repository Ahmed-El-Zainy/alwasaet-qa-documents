#!/usr/bin/env python3
"""
Setup script to fix vector dimension issues and configure Hugging Face embeddings
This script will:
1. Install required packages
2. Test embedding models and their dimensions
3. Update configuration 
4. Clear existing Qdrant data with wrong dimensions
5. Validate the setup
"""

import os
import sys
import subprocess
import logging
import yaml
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        logger.info(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install {package}: {e}")
        return False

def install_required_packages():
    """Install all required packages for Hugging Face embeddings"""
    packages = [
        "sentence-transformers>=2.2.2",
        "torch>=1.9.0", 
        "transformers>=4.20.0",
        "llama-index-embeddings-huggingface>=0.2.0"
    ]
    
    logger.info("📦 Installing required packages...")
    success = True
    for package in packages:
        if not install_package(package):
            success = False
    
    return success

def test_embedding_model(model_name):
    """Test an embedding model and return its dimension"""
    try:
        from sentence_transformers import SentenceTransformer
        
        logger.info(f"🧪 Testing model: {model_name}")
        model = SentenceTransformer(model_name)
        test_embedding = model.encode("test text")
        dimension = len(test_embedding) if hasattr(test_embedding, '__len__') else test_embedding.shape[0]
        
        logger.info(f"✅ {model_name}: {dimension} dimensions")
        return dimension
        
    except Exception as e:
        logger.error(f"❌ Failed to test {model_name}: {e}")
        return None

def test_llama_index_integration(model_name):
    """Test LlamaIndex integration with the embedding model"""
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            device="cpu"
        )
        
        test_embedding = embed_model.get_text_embedding("test text")
        dimension = len(test_embedding)
        
        logger.info(f"✅ LlamaIndex integration working for {model_name}: {dimension} dimensions")
        return dimension
        
    except Exception as e:
        logger.error(f"❌ LlamaIndex integration failed for {model_name}: {e}")
        return None

def clear_qdrant_data():
    """Clear existing Qdrant data to avoid dimension conflicts"""
    qdrant_storage_path = Path("qdrant_storage")
    
    if qdrant_storage_path.exists():
        logger.info("🗑️ Clearing existing Qdrant data...")
        try:
            shutil.rmtree(qdrant_storage_path)
            logger.info("✅ Qdrant data cleared successfully")
        except Exception as e:
            logger.error(f"❌ Failed to clear Qdrant data: {e}")
            return False
    else:
        logger.info("ℹ️ No existing Qdrant data found")
    
    return True

def update_config_file(recommended_model, dimension):
    """Update the config file with the recommended model"""
    config_path = Path("src/config.yaml")
    
    try:
        # Read current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update embedding configuration
        if 'huggingface' not in config:
            config['huggingface'] = {}
        if 'embeddings' not in config['huggingface']:
            config['huggingface']['embeddings'] = {}
        
        config['huggingface']['embeddings']['model'] = recommended_model
        config['embedding_provider'] = 'huggingface'
        
        # Update vector store configuration
        if 'vector_store' not in config:
            config['vector_store'] = {}
        config['vector_store']['vector_size'] = dimension
        
        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"✅ Updated config.yaml with {recommended_model} ({dimension} dimensions)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update config: {e}")
        return False

def restart_docker_qdrant():
    """Restart Qdrant Docker container"""
    try:
        logger.info("🔄 Restarting Qdrant container...")
        
        # Stop existing container
        subprocess.run(["docker", "stop", "qdrant"], capture_output=True)
        subprocess.run(["docker", "rm", "qdrant"], capture_output=True)
        
        # Start fresh container
        subprocess.check_call([
            "docker", "run", "-d", "--name", "qdrant",
            "-p", "6333:6333", "-p", "6334:6334",
            "-v", f"{os.getcwd()}/qdrant_storage:/qdrant/storage",
            "qdrant/qdrant:latest"
        ])
        
        # Wait a moment for startup
        import time
        time.sleep(3)
        
        logger.info("✅ Qdrant container restarted")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to restart Qdrant: {e}")
        return False

def validate_final_setup():
    """Validate the complete setup"""
    try:
        # Test vector store manager
        sys.path.append("src")
        from utils.vector_store import VectorStoreManager
        
        manager = VectorStoreManager("src/config.yaml")
        info = manager.get_embedding_info()
        test_result = manager.test_embedding()
        
        logger.info(f"✅ Final validation successful:")
        logger.info(f"   - Provider: {info['provider']}")
        logger.info(f"   - Model: {info['model']}")
        logger.info(f"   - Dimensions: {info['vector_size']}")
        logger.info(f"   - Embedding test: {'✅ Passed' if test_result['success'] else '❌ Failed'}")
        
        return test_result['success']
        
    except Exception as e:
        logger.error(f"❌ Final validation failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Fixing Vector Dimension Issues - Hugging Face Embeddings Setup")
    print("=" * 70)
    
    # Step 1: Install required packages
    if not install_required_packages():
        print("❌ Package installation failed. Aborting.")
        return 1
    
    # Step 2: Test available embedding models
    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",  # 384 dims, fast
        # "sentence-transformers/all-mpnet-base-v2",  # 768 dims, quality
        # "sentence-transformers/paraphrase-MiniLM-L6-v2"  # 384 dims, alternative
    ]
    
    logger.info("🧪 Testing embedding models...")
    working_models = {}
    
    for model in models_to_test:
        # Test with sentence-transformers directly
        st_dim = test_embedding_model(model)
        if st_dim:
            # Test with LlamaIndex integration
            li_dim = test_llama_index_integration(model)
            if li_dim and st_dim == li_dim:
                working_models[model] = st_dim
                logger.info(f"✅ {model} fully working: {st_dim} dimensions")
            else:
                logger.warning(f"⚠️ {model} dimension mismatch or LlamaIndex issue")
    
    if not working_models:
        logger.error("❌ No working embedding models found!")
        return 1
    
    # Step 3: Choose recommended model (prefer smaller dimensions for efficiency)
    recommended_model = min(working_models.items(), key=lambda x: x[1])
    model_name, dimension = recommended_model
    
    logger.info(f"🎯 Recommended model: {model_name} ({dimension} dimensions)")
    
    # Step 4: Clear existing Qdrant data
    if not clear_qdrant_data():
        logger.warning("⚠️ Could not clear Qdrant data, continuing anyway...")
    
    # Step 5: Update configuration
    if not update_config_file(model_name, dimension):
        logger.error("❌ Failed to update configuration!")
        return 1
    
    # Step 6: Restart Qdrant (optional, only if Docker is available)
    try:
        if subprocess.run(["docker", "--version"], capture_output=True).returncode == 0:
            restart_docker_qdrant()
    except:
        logger.warning("⚠️ Docker not available, please restart Qdrant manually")
    
    # Step 7: Final validation
    logger.info("🔍 Running final validation...")
    if validate_final_setup():
        print("\n🎉 Setup completed successfully!")
        print("\n📋 Summary:")
        print(f"✅ Embedding Model: {model_name}")
        print(f"✅ Vector Dimensions: {dimension}")
        print(f"✅ Provider: Hugging Face (local)")
        print(f"✅ LLM: Gemini (API)")
        print(f"✅ Vector DB: Qdrant")
        
        print("\n🚀 Next steps:")
        print("1. Your system is now configured correctly")
        print("2. Run your application: ./run.sh")
        print("3. Upload documents - they will be embedded with correct dimensions")
        print("4. No more dimension mismatch errors!")
        
        print("\n💡 Benefits of this setup:")
        print("- No API costs for embeddings (runs locally)")
        print("- Consistent vector dimensions")
        print("- Fast and efficient")
        print("- Works offline for embeddings")
        
    else:
        print("❌ Final validation failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)