#!/bin/bash

# Quick fix for 0 citations issue
echo "🔧 Quick Fix for 0 Citations Issue"
echo "=================================="

# Update config to lower similarity threshold
echo "⚙️ Lowering similarity threshold for better retrieval..."

cat > src/config_retrieval_fix.yaml << 'EOF'
app:
  name: "Agentic RAG Application"
  version: "1.0.0"
  debug: false

model_provider: "mixed"
embedding_provider: "huggingface"
llm_provider: "gemini"

gemini:
  llm:
    model: "gemini-1.5-flash"  # You can change this to other models
    temperature: 0.1
    max_output_tokens: 2048

huggingface:
  embeddings:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    device: "cpu"

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
  similarity_top_k: 8           # Increased for more results
  similarity_threshold: 0.1     # Much lower threshold

parsing:
  use_llama_parse: false
  extract_images: false

memory:
  enabled: true
  max_tokens: 4000

performance:
  batch_size: 5
  max_workers: 2
  cache_embeddings: true
EOF

# Apply the fix
cp src/config.yaml src/config.yaml.backup
cp src/config_retrieval_fix.yaml src/config.yaml

echo "✅ Updated configuration:"
echo "   - similarity_threshold: 0.5 → 0.1"
echo "   - similarity_top_k: 3 → 8"
echo ""
echo "🔄 Now restart your application:"
echo "   1. Stop current app (Ctrl+C)"
echo "   2. Run: ./run.sh"
echo "   3. Upload your document again"
echo "   4. Test with questions"
echo ""
echo "💡 Alternative Gemini models you can try:"
echo "   - gemini-1.5-flash (current, fast)"
echo "   - gemini-1.5-pro (best quality, slower)"
echo "   - gemini-1.5-flash-8b (fastest, lightweight)"
echo ""
echo "🧪 To test models and retrieval, run:"
echo "   python fix_retrieval.py"

# chmod +x src/tests/quick_citation_fix.sh
# ./src/tests/quick_citation_fix.sh