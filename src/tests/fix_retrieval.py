#!/usr/bin/env python3
"""
Script to fix retrieval issues and test different Gemini models
"""

import os
import sys
import yaml
import google.generativeai as genai
from qdrant_client import QdrantClient

def test_gemini_models():
    """Test which Gemini models are available"""
    print("🤖 Testing Available Gemini Models")
    print("=" * 40)
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("❌ GOOGLE_API_KEY not found")
        return []
    
    genai.configure(api_key=api_key)
    
    # List all available models
    available_models = []
    try:
        models = genai.list_models()
        print("📋 All available models:")
        for model in models:
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.replace('models/', '')
                available_models.append(model_name)
                print(f"   ✅ {model_name}")
    except Exception as e:
        print(f"❌ Error listing models: {e}")
    
    # Test specific models
    models_to_test = [
        "gemini-1.5-flash",
        "gemini-1.5-pro", 
        "gemini-1.5-flash-8b",
        "gemini-2.0-flash-exp",  # Experimental
        "gemini-exp-1114"        # Experimental
    ]
    
    working_models = []
    print(f"\n🧪 Testing specific models:")
    
    for model_name in models_to_test:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content("Hello! Respond with just 'Working'")
            
            if response.parts and response.text:
                working_models.append(model_name)
                print(f"   ✅ {model_name}: {response.text.strip()}")
            else:
                print(f"   ⚠️ {model_name}: No response")
                
        except Exception as e:
            print(f"   ❌ {model_name}: {str(e)}")
    
    return working_models

def test_retrieval():
    """Test if vector retrieval is working"""
    print("\n🔍 Testing Vector Retrieval")
    print("=" * 30)
    
    try:
        # Test Qdrant connection
        client = QdrantClient(host="localhost", port=6333)
        collections = client.get_collections()
        
        if not collections.collections:
            print("❌ No collections found")
            return False
        
        collection_name = "documents"
        found_collection = False
        
        for collection in collections.collections:
            if collection.name == collection_name:
                found_collection = True
                info = client.get_collection(collection_name)
                print(f"✅ Collection '{collection_name}' found")
                print(f"   Vectors: {info.points_count}")
                print(f"   Vector size: {info.config.params.vectors.size}")
                break
        
        if not found_collection:
            print(f"❌ Collection '{collection_name}' not found")
            return False
        
        # Test if there are any vectors
        if info.points_count == 0:
            print("⚠️ Collection is empty - no documents indexed")
            return False
        
        # Test search
        print(f"\n🧪 Testing search...")
        import random
        test_vector = [random.random() for _ in range(384)]
        
        search_results = client.search(
            collection_name=collection_name,
            query_vector=test_vector,
            limit=3,
            with_payload=True
        )
        
        print(f"✅ Search successful, found {len(search_results)} results")
        
        for i, result in enumerate(search_results):
            print(f"   Result {i+1}: Score={result.score:.3f}")
            if result.payload:
                file_name = result.payload.get('file_name', 'Unknown')
                page = result.payload.get('page_number', 'Unknown')
                print(f"      File: {file_name}, Page: {page}")
        
        return True
        
    except Exception as e:
        print(f"❌ Retrieval test failed: {e}")
        return False

def fix_similarity_threshold():
    """Fix similarity threshold in config"""
    print("\n⚙️ Fixing Similarity Threshold")
    print("=" * 35)
    
    config_path = "src/config.yaml"
    
    try:
        # Read current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update retrieval settings
        if 'retrieval' not in config:
            config['retrieval'] = {}
        
        # Lower the threshold for better retrieval
        config['retrieval']['similarity_threshold'] = 0.1  # Very low threshold
        config['retrieval']['similarity_top_k'] = 5
        
        # Write back
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print("✅ Updated similarity threshold to 0.1")
        print("✅ Updated top_k to 5")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def test_rag_pipeline():
    """Test the complete RAG pipeline"""
    print("\n🔧 Testing Complete RAG Pipeline")
    print("=" * 40)
    
    try:
        # Import your modules
        sys.path.append('src')
        from main import RAGApplication
        
        # Initialize
        app = RAGApplication()
        
        # Test with a simple query
        test_queries = [
            "What is this document about?",
            "Tell me about Saudi Arabia",
            "What are the main findings?",
            "transparency report"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: '{query}'")
            try:
                result = app.query(query)
                
                print(f"   Answer length: {len(result['answer'])} chars")
                print(f"   Citations: {len(result['citations'])}")
                
                if result['citations']:
                    print("   ✅ Citations found:")
                    for i, citation in enumerate(result['citations'][:2]):
                        print(f"      {i+1}. {citation['file_name']} (page {citation['page_number']})")
                else:
                    print("   ⚠️ No citations found")
                
                # Show first 100 chars of answer
                answer_preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
                print(f"   Answer: {answer_preview}")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline test failed: {e}")
        return False

def update_model_config(model_name):
    """Update config to use a specific model"""
    print(f"\n⚙️ Updating Config to Use {model_name}")
    print("=" * 45)
    
    config_path = "src/config.yaml"
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update model
        if 'gemini' not in config:
            config['gemini'] = {}
        if 'llm' not in config['gemini']:
            config['gemini']['llm'] = {}
        
        config['gemini']['llm']['model'] = model_name
        
        # Also update legacy config
        if 'llm' in config:
            config['llm']['model'] = model_name
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"✅ Updated config to use {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config: {e}")
        return False

def main():
    """Main function"""
    print("🚀 RAG System Diagnostics and Model Testing")
    print("=" * 50)
    
    # Test 1: Available models
    working_models = test_gemini_models()
    
    # Test 2: Retrieval system
    retrieval_ok = test_retrieval()
    
    # Test 3: Fix similarity threshold if needed
    if not retrieval_ok or True:  # Always run to improve retrieval
        fix_similarity_threshold()
    
    # Test 4: Test RAG pipeline
    test_rag_pipeline()
    
    # Recommendations
    print("\n📋 Recommendations:")
    print("=" * 20)
    
    if working_models:
        print(f"✅ Working models: {', '.join(working_models)}")
        
        # Recommend best model
        if "gemini-1.5-pro" in working_models:
            recommended = "gemini-1.5-pro"
            reason = "Best quality for complex reasoning"
        elif "gemini-1.5-flash" in working_models:
            recommended = "gemini-1.5-flash"
            reason = "Good balance of speed and quality"
        else:
            recommended = working_models[0]
            reason = "Available model"
        
        print(f"🎯 Recommended: {recommended} ({reason})")
        
        # Ask if user wants to update
        print(f"\n💡 To use {recommended}, update your config:")
        print(f"   model: \"{recommended}\"")
        
    else:
        print("❌ No working models found - check your API key")
    
    if not retrieval_ok:
        print("⚠️ Retrieval issues detected:")
        print("   1. Make sure documents are uploaded")
        print("   2. Check Qdrant has vectors")
        print("   3. Lower similarity threshold")
    
    print("\n🔄 After making changes, restart your application:")
    print("   Ctrl+C to stop, then ./run.sh")

if __name__ == "__main__":
    main()