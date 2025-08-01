import logging
from typing import Dict, Any, List, Optional
import yaml
from .rag_engine import RAGEngine
import os 
import sys 


# fmt: off
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

try:
    from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("rag engine")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("rag engine")
    logger.info("Using standard logger - custom logger not available")


class ConversationMemory:
    """Simple conversation memory management"""
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        self.conversation_history = []
    
    def add_exchange(self, question: str, answer: str):
        """Add a question-answer exchange to memory"""
        self.conversation_history.append({
            "question": question,
            "answer": answer
        })
        
        # Simple token management (rough estimate)
        total_tokens = sum(len(ex["question"]) + len(ex["answer"]) for ex in self.conversation_history)
        
        while total_tokens > self.max_tokens and self.conversation_history:
            removed = self.conversation_history.pop(0)
            total_tokens -= len(removed["question"]) + len(removed["answer"])
    
    def get_context(self) -> str:
        """Get conversation context for the current query"""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context_parts.append(f"Previous Q: {exchange['question']}")
            context_parts.append(f"Previous A: {exchange['answer']}")
        
        return "\n".join(context_parts)
    
    def clear(self):
        """Clear conversation history"""
        self.conversation_history = []

class AgenticRAG:
    """Agentic RAG system with conversation memory and multi-step reasoning"""
    
    def __init__(self, config_path: str = "src/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.rag_engine = RAGEngine(config_path)
        
        # Initialize memory if enabled
        if self.config['memory']['enabled']:
            self.memory = ConversationMemory(
                max_tokens=self.config['memory']['max_tokens']
            )
        else:
            self.memory = None
        
        logger.info("Agentic RAG system initialized with Gemini")
    
    def setup(self, index):
        """Setup the agentic system with vector index"""
        self.rag_engine.setup_engine(index)
    
    def process_query(self, question: str) -> Dict[str, Any]:
        """Process query with conversation memory and context"""
        try:
            # Enhance question with conversation context if memory is enabled
            enhanced_question = self._enhance_question_with_context(question)
            
            # Get response from RAG engine
            response = self.rag_engine.query(enhanced_question)
            
            # Add to memory if enabled
            if self.memory and response["answer"] != "No answer found":
                self.memory.add_exchange(question, response["answer"])
            
            # Add conversation context to response
            response["original_question"] = question
            response["enhanced_question"] = enhanced_question
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "answer": "Error processing query",
                "citations": [],
                "source_nodes": [],
                "original_question": question,
                "enhanced_question": question
            }
    
    def _enhance_question_with_context(self, question: str) -> str:
        """Enhance question with conversation context"""
        if not self.memory:
            return question
        
        context = self.memory.get_context()
        if not context:
            return question
        
        # Simple context enhancement
        enhanced = f"""Previous conversation context:
{context}

Current question: {question}

Please answer the current question considering the previous conversation context where relevant."""
        
        return enhanced
    
    def clear_memory(self):
        """Clear conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Conversation memory cleared")



if __name__=="__main__":
    logger.info("test....")