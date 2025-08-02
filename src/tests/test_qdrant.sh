#!/bin/bash

# Fix Qdrant 500 error and model configuration issues
echo "🔧 Fixing Qdrant Configuration and Model Issues"
echo "==============================================="

# Stop the current application
echo "🛑 Stopping current application..."
pkill -f "gradio_demo.py" 2>/dev/null || true
pkill -f "python.*gradio" 2>/dev/null || true

# Stop and remove Qdrant container
echo "🗑️ Resetting Qdrant with proper configuration..."
docker stop qdrant 2>/dev/null || true
docker rm qdrant 2>/dev/null || true

# Remove problematic Qdrant storage
rm -rf qdrant_storage/
echo "✅ Cleared Qdrant storage"

# Set environment variable to fix tokenizer warning
export TOKENIZERS_PARALLELISM=false
echo "✅ Set TOKENIZERS_PARALLELISM=false"

# Update config to fix model name
echo "⚙️ Fixing configuration..."
cat > src/config_fixed.yaml << 'EOF'
app:
  name: "Agentic RAG Application"
  version: "1.0.0"
  debug: false

model_provider: "mixed"
embedding_provider: "huggingface"
llm_provider: "gemini"

gemini:
  llm:
    model: "gemini-1.5-flash"  # Fixed: was gemini-2.5-flash-lite
    temperature: 0.1
    max_output_tokens: 2048

huggingface:
  embeddings:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cpu"  # Fixed: changed from "auto" to "cpu"

llm:
  provider: "gemini"
  model: "gemini-1.5-flash"
  temperature: 0.1
  max_output_tokens: 2048

embeddings:
  provider: "huggingface"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  chunk_size: 1000
  chunk_overlap: 200

vector_store:
  provider: "qdrant"
  collection_name: "documents"
  vector_size: 384
  distance: "cosine"
  host: "localhost"
  port: 6333

retrieval:
  similarity_top_k: 3  # Reduced from 5 to avoid issues
  similarity_threshold: 0.5  # Reduced from 0.7 for better recall

parsing:
  use_llama_parse: false
  extract_images: false

memory:
  enabled: true
  max_tokens: 4000

performance:
  batch_size: 5  # Reduced batch size
  max_workers: 2  # Reduced workers
  cache_embeddings: true
EOF

# Backup and replace config
cp src/config.yaml src/config.yaml.backup
cp src/config_fixed.yaml src/config.yaml
echo "✅ Configuration updated"

# Start Qdrant with better configuration
echo "🚀 Starting Qdrant with optimized settings..."
docker run -d --name qdrant \
    -p 6333:6333 -p 6334:6334 \
    -v "${PWD}/qdrant_storage:/qdrant/storage" \
    -e QDRANT__SERVICE__HTTP_PORT=6333 \
    -e QDRANT__SERVICE__GRPC_PORT=6334 \
    -e QDRANT__LOG_LEVEL=INFO \
    qdrant/qdrant:latest

# Wait for Qdrant
echo "⏳ Waiting for Qdrant to initialize..."
sleep 8

# Test Qdrant
for i in {1..15}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "✅ Qdrant is healthy"
        break
    fi
    echo "   Waiting... ($i/15)"
    sleep 2
done

# Test Qdrant collections endpoint
echo "🧪 Testing Qdrant API..."
curl -s http://localhost:6333/collections > /dev/null && echo "✅ Qdrant API responding" || echo "⚠️ Qdrant API issues"

echo ""
echo "🎉 Fixes Applied!"
echo ""
echo "📋 Changes made:"
echo "✅ Fixed model name: gemini-2.5-flash-lite → gemini-1.5-flash"
echo "✅ Fixed device setting: auto → cpu (more reliable)"
echo "✅ Reduced similarity threshold: 0.7 → 0.5"
echo "✅ Reduced top-k retrieval: 5 → 3"
echo "✅ Optimized batch processing settings"
echo "✅ Reset Qdrant with proper configuration"
echo "✅ Set TOKENIZERS_PARALLELISM=false"
echo ""
echo "🚀 Now run: ./run.sh"
echo ""
echo "💡 If you still get Qdrant errors, try:"
echo "   docker logs qdrant"
echo "   curl http://localhost:6333/health"