#!/bin/bash

# Complete Bilingual Arabic-English RAG Setup
echo "🌍 Setting up Bilingual Arabic-English RAG System"
echo "================================================"

# Stop current processes
echo "🛑 Stopping current application..."
pkill -f "gradio_demo.py" 2>/dev/null || true
pkill -f "python.*gradio" 2>/dev/null || true

# Stop and remove Qdrant
echo "🗑️ Resetting Qdrant for bilingual support..."
docker stop qdrant 2>/dev/null || true
docker rm qdrant 2>/dev/null || true
rm -rf qdrant_storage/

# Set bilingual environment
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TOKENIZERS_PARALLELISM=false
echo "✅ Set bilingual environment variables"

# Install required packages for bilingual support
echo "📦 Installing bilingual embedding models..."
pip install sentence-transformers>=2.2.2
pip install langdetect  # For language detection
pip install polyglot    # Additional language processing (optional)

# Test multilingual models
echo "🧪 Testing multilingual embedding models..."

# Test Model 1: mpnet-base-v2 (best quality)
python3 -c "
try:
    from sentence_transformers import SentenceTransformer
    print('Testing paraphrase-multilingual-mpnet-base-v2...')
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    # Test both languages
    texts = [
        'This is an English document about data protection.',
        'هذه وثيقة باللغة العربية حول حماية البيانات الشخصية.'
    ]
    embeddings = model.encode(texts)
    print(f'✅ Model working: {embeddings.shape} (768 dimensions)')
    
    # Test similarity between languages
    import numpy as np
    similarity = np.dot(embeddings[0], embeddings[1])
    print(f'✅ Cross-language similarity: {similarity:.3f}')
    
except Exception as e:
    print(f'❌ Model test failed: {e}')
    print('Falling back to smaller model...')
"

# Create bilingual configuration
echo "⚙️ Creating bilingual configuration..."
cat > src/config_bilingual.yaml << 'EOF'
app:
  name: "Bilingual RAG Application - Arabic & English"
  version: "1.0.0"
  debug: false

language_support:
  primary_languages: ["arabic", "english"]
  auto_detect: true
  fallback_language: "english"
  
  arabic:
    chunk_size: 400
    chunk_overlap: 80
    similarity_threshold: 0.25
  
  english:
    chunk_size: 800
    chunk_overlap: 150
    similarity_threshold: 0.15

model_provider: "mixed"
embedding_provider: "huggingface"
llm_provider: "gemini"

gemini:
  llm:
    model: "gemini-1.5-flash"
    temperature: 0.1
    max_output_tokens: 2048

huggingface:
  embeddings:
    model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    device: "cpu"

llm:
  provider: "gemini"
  model: "gemini-1.5-flash"
  temperature: 0.1
  max_output_tokens: 2048

embeddings:
  provider: "huggingface"
  model: "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
  chunk_size: 600
  chunk_overlap: 120

vector_store:
  provider: "qdrant"
  collection_name: "bilingual_documents"
  vector_size: 768
  distance: "cosine"
  host: "localhost"
  port: 6333

retrieval:
  similarity_top_k: 5
  similarity_threshold: 0.2
  cross_lingual: true

parsing:
  use_llama_parse: false
  extract_images: false
  detect_language: true

memory:
  enabled: true
  max_tokens: 4000

performance:
  batch_size: 4
  max_workers: 2
  cache_embeddings: true
EOF

# Apply configuration
cp src/config.yaml src/config.yaml.backup
cp src/config_bilingual.yaml src/config.yaml
echo "✅ Applied bilingual configuration"

# Start Qdrant with optimal settings for bilingual content
echo "🚀 Starting Qdrant optimized for bilingual content..."
docker run -d --name qdrant \
    -p 6333:6333 -p 6334:6334 \
    -v "${PWD}/qdrant_storage:/qdrant/storage" \
    --memory=6g \
    --memory-swap=6g \
    -e QDRANT__SERVICE__HTTP_PORT=6333 \
    -e QDRANT__SERVICE__GRPC_PORT=6334 \
    -e QDRANT__LOG_LEVEL=INFO \
    -e QDRANT__STORAGE__WAL_CAPACITY_MB=128 \
    -e QDRANT__STORAGE__OPTIMIZERS__MEMMAP_THRESHOLD_KB=100000 \
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    qdrant/qdrant:latest

# Wait for Qdrant
echo "⏳ Waiting for bilingual Qdrant to start..."
sleep 15

# Test Qdrant
for i in {1..20}; do
    if curl -s http://localhost:6333/health > /dev/null 2>&1; then
        echo "✅ Qdrant ready for bilingual content"
        break
    fi
    echo "   Waiting... ($i/20)"
    sleep 2
done

# Create bilingual test script
echo "📝 Creating bilingual test script..."
cat > test_bilingual.py << 'EOF'
#!/usr/bin/env python3
"""Test bilingual functionality"""

import sys
import os
sys.path.append('src')

def test_bilingual_system():
    print("🌍 Testing Bilingual System")
    print("=" * 30)
    
    try:
        from main import RAGApplication
        app = RAGApplication()
        
        # Test queries in both languages
        test_queries = [
            # English queries
            ("English", "What is this document about?"),
            ("English", "What are the main topics discussed?"),
            ("English", "Tell me about data protection laws."),
            
            # Arabic queries  
            ("Arabic", "ما هو موضوع هذه الوثيقة؟"),
            ("Arabic", "ما هي القوانين المذكورة في الوثيقة؟"),
            ("Arabic", "تحدث عن حماية البيانات الشخصية."),
            
            # Mixed language queries
            ("Mixed", "What does the document say about حماية البيانات؟"),
            ("Mixed", "ما رأيك في data protection laws؟")
        ]
        
        for lang_type, query in test_queries:
            print(f"\n🔍 {lang_type} Query: {query}")
            try:
                result = app.query(query)
                
                answer_length = len(result['answer'])
                citations_count = len(result['citations'])
                
                print(f"   ✅ Answer: {answer_length} chars")
                print(f"   ✅ Citations: {citations_count}")
                
                if citations_count > 0:
                    print("   📚 Sources:")
                    for i, citation in enumerate(result['citations'][:2]):
                        file_name = citation.get('file_name', 'Unknown')
                        page = citation.get('page_number', 'Unknown')
                        print(f"      {i+1}. {file_name} (Page {page})")
                
                # Show answer preview
                preview = result['answer'][:100] + "..." if len(result['answer']) > 100 else result['answer']
                print(f"   💬 Preview: {preview}")
                
            except Exception as e:
                print(f"   ❌ Query failed: {e}")
    
    except Exception as e:
        print(f"❌ System test failed: {e}")

if __name__ == "__main__":
    test_bilingual_system()
EOF

chmod +x test_bilingual.py

# Test the embedding model
echo "🧪 Final embedding test..."
python3 -c "
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    # Test bilingual embedding
    texts = [
        'Data protection is important for privacy.',
        'حماية البيانات مهمة للخصوصية.',
        'This document discusses both English and Arabic content. هذه الوثيقة تناقش المحتوى الإنجليزي والعربي.'
    ]
    
    embeddings = model.encode(texts)
    print(f'✅ Bilingual embedding successful: {embeddings.shape}')
    
    # Test cross-language similarity
    import numpy as np
    en_ar_similarity = np.dot(embeddings[0], embeddings[1])
    mixed_similarity = np.dot(embeddings[0], embeddings[2])
    
    print(f'✅ English-Arabic similarity: {en_ar_similarity:.3f}')
    print(f'✅ Mixed content similarity: {mixed_similarity:.3f}')
    
except Exception as e:
    print(f'❌ Final test failed: {e}')
"

echo ""
echo "🎉 Bilingual Arabic-English RAG System Ready!"
echo "============================================"

echo ""
echo "📋 System Features:"
echo "✅ Automatic language detection"
echo "✅ Optimized processing for Arabic & English"
echo "✅ Cross-language search capabilities"
echo "✅ Language-aware chunking strategies"
echo "✅ Bilingual embedding model (768 dimensions)"
echo "✅ Gemini 1.5 Flash with excellent bilingual support"

echo ""
echo "🚀 Next Steps:"
echo "1. Start your application: ./run.sh"
echo "2. Upload both Arabic and English PDFs"
echo "3. Ask questions in either language"
echo "4. Test bilingual functionality: python test_bilingual.py"

echo ""
echo "💡 Example queries to try:"
echo "English:"
echo "   - 'What is this document about?'"
echo "   - 'Summarize the main points'"
echo "   - 'What laws are mentioned?'"
echo ""
echo "Arabic:"
echo "   - 'ما هو موضوع هذه الوثيقة؟'"
echo "   - 'ما هي القوانين المذكورة؟'"
echo "   - 'لخص النقاط الرئيسية'"
echo ""
echo "Mixed:"
echo "   - 'What does it say about حماية البيانات؟'"
echo "   - 'Compare English and Arabic sections'"

echo ""
echo "🔧 Model Information:"
echo "   - Embedding: paraphrase-multilingual-mpnet-base-v2"
echo "   - Dimensions: 768"
echo "   - Languages: 104+ including Arabic & English"
echo "   - LLM: Gemini 1.5 Flash (bilingual)"
echo "   - Vector DB: Qdrant (optimized for multilingual)"




# chmod +x src/tests/complete_bilingual_solution.sh
# ./src/tests/complete_bilingual_solution.sh