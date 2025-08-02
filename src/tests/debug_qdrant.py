#!/usr/bin/env python3
"""
Debug script to diagnose and fix Qdrant issues
"""

import requests
import json
import sys
import time
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def test_qdrant_health():
    """Test if Qdrant is healthy"""
    try:
        response = requests.get("http://localhost:6333/health", timeout=5)
        if response.status_code == 200:
            print("✅ Qdrant health check passed")
            return True
        else:
            print(f"❌ Qdrant health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Qdrant health check error: {e}")
        return False

def test_qdrant_info():
    """Get Qdrant cluster info"""
    try:
        response = requests.get("http://localhost:6333/", timeout=5)
        if response.status_code == 200:
            info = response.json()
            print(f"✅ Qdrant version: {info.get('version', 'unknown')}")
            return True
        else:
            print(f"❌ Qdrant info failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Qdrant info error: {e}")
        return False

def test_collections():
    """Test collections operations"""
    try:
        client = QdrantClient(host="localhost", port=6333)
        
        # Get collections
        collections = client.get_collections()
        print(f"✅ Current collections: {[c.name for c in collections.collections]}")
        
        # Test creating a small collection
        test_collection = "test_collection"
        
        # Delete if exists
        try:
            client.delete_collection(test_collection)
            print(f"🗑️ Deleted existing test collection")
        except:
            pass
        
        # Create new collection
        client.create_collection(
            collection_name=test_collection,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"✅ Created test collection")
        
        # Test inserting a small vector
        from qdrant_client.models import PointStruct
        import random
        
        test_vector = [random.random() for _ in range(384)]
        client.upsert(
            collection_name=test_collection,
            points=[
                PointStruct(
                    id=1,
                    vector=test_vector,
                    payload={"test": "data"}
                )
            ]
        )
        print(f"✅ Successfully inserted test vector")
        
        # Test search
        search_result = client.search(
            collection_name=test_collection,
            query_vector=test_vector,
            limit=1
        )
        print(f"✅ Search successful, found {len(search_result)} results")
        
        # Cleanup
        client.delete_collection(test_collection)
        print(f"🗑️ Cleaned up test collection")
        
        return True
        
    except Exception as e:
        print(f"❌ Collection test failed: {e}")
        return False

def diagnose_error(error_msg):
    """Diagnose specific error patterns"""
    if "OutputTooSmall" in error_msg:
        print("\n🔍 Diagnosis: OutputTooSmall Error")
        print("This usually means:")
        print("1. Vector dimension mismatch during search")
        print("2. Corrupted index or collection")
        print("3. Memory/buffer issues in Qdrant")
        print("\n💡 Solutions:")
        print("- Clear and recreate collections")
        print("- Reduce batch size and similarity_top_k")
        print("- Restart Qdrant container")
        return True
    
    if "Service internal error" in error_msg:
        print("\n🔍 Diagnosis: Qdrant Internal Error")
        print("This usually means:")
        print("1. Qdrant service instability")
        print("2. Resource constraints")
        print("3. Configuration issues")
        print("\n💡 Solutions:")
        print("- Restart Qdrant with more memory")
        print("- Check Docker logs: docker logs qdrant")
        print("- Use smaller batch sizes")
        return True
    
    return False

def fix_qdrant_issues():
    """Apply common fixes"""
    print("\n🔧 Applying fixes...")
    
    import subprocess
    import os
    
    try:
        # Stop current container
        subprocess.run(["docker", "stop", "qdrant"], capture_output=True)
        subprocess.run(["docker", "rm", "qdrant"], capture_output=True)
        
        # Remove storage
        if os.path.exists("qdrant_storage"):
            import shutil
            shutil.rmtree("qdrant_storage")
            print("✅ Cleared Qdrant storage")
        
        # Start with memory limits and better config
        cmd = [
            "docker", "run", "-d", "--name", "qdrant",
            "-p", "6333:6333", "-p", "6334:6334",
            "-v", f"{os.getcwd()}/qdrant_storage:/qdrant/storage",
            "--memory=2g",  # Limit memory
            "--memory-swap=2g",
            "-e", "QDRANT__SERVICE__HTTP_PORT=6333",
            "-e", "QDRANT__SERVICE__GRPC_PORT=6334",
            "-e", "QDRANT__LOG_LEVEL=INFO",
            "-e", "QDRANT__STORAGE__RECOVERY_MODE=true",
            "qdrant/qdrant:latest"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Started Qdrant with optimized configuration")
            
            # Wait for startup
            print("⏳ Waiting for Qdrant to start...")
            for i in range(20):
                if test_qdrant_health():
                    break
                time.sleep(1)
            
            return True
        else:
            print(f"❌ Failed to start Qdrant: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Fix attempt failed: {e}")
        return False

def main():
    """Main diagnostic routine"""
    print("🔍 Qdrant Diagnostic Tool")
    print("=" * 30)
    
    # Check if error message provided
    if len(sys.argv) > 1:
        error_msg = " ".join(sys.argv[1:])
        diagnose_error(error_msg)
    
    print("\n1. Testing Qdrant Health...")
    health_ok = test_qdrant_health()
    
    print("\n2. Testing Qdrant Info...")
    info_ok = test_qdrant_info()
    
    print("\n3. Testing Collections...")
    collections_ok = test_collections()
    
    if not (health_ok and info_ok and collections_ok):
        print("\n❌ Issues detected. Attempting fixes...")
        
        if fix_qdrant_issues():
            print("\n✅ Fixes applied. Testing again...")
            
            time.sleep(3)
            health_ok = test_qdrant_health()
            collections_ok = test_collections()
            
            if health_ok and collections_ok:
                print("\n🎉 Qdrant is now working correctly!")
            else:
                print("\n⚠️ Some issues remain. Check Docker logs:")
                print("docker logs qdrant")
        else:
            print("\n❌ Automatic fixes failed. Manual intervention needed.")
    else:
        print("\n🎉 All tests passed! Qdrant is working correctly.")
    
    print("\n💡 If problems persist:")
    print("1. Check Docker logs: docker logs qdrant")
    print("2. Try different Qdrant version: qdrant/qdrant:v1.7.0")
    print("3. Increase Docker memory limits")
    print("4. Use smaller batch sizes in your config")

if __name__ == "__main__":
    main()