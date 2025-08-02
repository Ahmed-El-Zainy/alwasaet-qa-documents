# src/gradio_demo.py
import os
import sys
import gradio as gr
import logging
from pathlib import Path
from typing import List, Tuple
import tempfile
from main import RAGApplication


try:
    from logger.custom_logger import CustomLoggerTracker
    custom = CustomLoggerTracker()
    logger = custom.get_logger("gradio demo")
    logger.info("Custom Logger Start Working.....")

except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("grado demo")
    logger.info("Using standard logger - custom logger not available")



class GradioRAGDemo:
    """Gradio interface for the RAG application with Gemini"""
    
    def __init__(self):
        self.app = RAGApplication()
        self.uploaded_files = []
    
    def upload_files(self, files) -> str:
        """Handle file uploads"""
        if not files:
            return "No files uploaded."
        
        try:
            # Clear previous files
            self.uploaded_files = []
            
            # Process uploaded files
            for file in files:
                if file is not None:
                    # Copy file to temp location for processing
                    temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(file.name))
                    
                    # Copy file content
                    with open(file.name, 'rb') as src, open(temp_path, 'wb') as dst:
                        dst.write(src.read())
                    
                    self.uploaded_files.append(temp_path)
            
            # Process documents
            success = self.app.process_documents(self.uploaded_files)
            
            if success:
                file_names = [os.path.basename(f) for f in self.uploaded_files]
                return f"✅ Successfully processed {len(self.uploaded_files)} files with Gemini: {', '.join(file_names)}"
            else:
                return "❌ Error processing files. Please check the logs."
                
        except Exception as e:
            logger.error(f"Error in upload_files: {str(e)}")
            return f"❌ Error uploading files: {str(e)}"
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """Handle chat messages"""
        if not message.strip():
            return "", history
        
        try:
            # Query the RAG system
            response = self.app.query(message)
            
            # Format response with citations
            answer = response["answer"]
            citations = response["citations"]
            
            if citations:
                citation_text = "\n\n**📚 Sources:**\n"
                for i, citation in enumerate(citations, 1):
                    score_text = f" (Score: {citation['score']:.3f})" if citation['score'] else ""
                    citation_text += f"{i}. 📄 **{citation['file_name']}** - Page {citation['page_number']}{score_text}\n"
                answer += citation_text
            
            # Update history
            history.append((message, answer))
            
            return "", history
            
        except Exception as e:
            logger.error(f"Error in chat: {str(e)}")
            error_msg = f"❌ Error processing query: {str(e)}"
            history.append((message, error_msg))
            return "", history
    
    def clear_chat(self) -> List[Tuple[str, str]]:
        """Clear chat history"""
        self.app.clear_memory()
        return []
    
    def reset_system(self) -> Tuple[str, List[Tuple[str, str]]]:
        """Reset the entire system"""
        try:
            self.app.reset_system()
            self.uploaded_files = []
            return "🔄 System reset successfully.", []
        except Exception as e:
            return f"❌ Error resetting system: {str(e)}", []
    
    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(
            title="Agentic RAG with Gemini", 
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
            }
            """
        ) as interface:
            gr.Markdown(
                """
                # 🤖 Agentic RAG Application with Google Gemini
                Upload documents and ask questions to get accurate answers with citations powered by **Gemini 1.5 Pro**.
                """,
                elem_id="header"
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # File upload section
                    gr.Markdown("## 📁 Document Upload")
                    file_upload = gr.File(
                        label="Upload PDF Documents",
                        file_count="multiple",
                        file_types=[".pdf"],
                        height=120
                    )
                    upload_status = gr.Textbox(
                        label="Upload Status",
                        interactive=False,
                        lines=4,
                        placeholder="Upload status will appear here..."
                    )
                    
                    # Control buttons
                    gr.Markdown("## 🎛️ Controls")
                    with gr.Row():
                        clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary", size="sm")
                        reset_btn = gr.Button("🔄 Reset System", variant="stop", size="sm")
                    
                    # Model info
                    gr.Markdown(
                        """
                        ## 🔧 Model Information
                        - **LLM**: Gemini 1.5 Pro
                        - **Embeddings**: Gemini Embedding-001
                        - **Vector DB**: Qdrant
                        - **Framework**: LlamaIndex
                        """
                    )
                
                with gr.Column(scale=2):
                    # Chat interface
                    gr.Markdown("## 💬 Chat with Your Documents")
                    chatbot = gr.Chatbot(
                        label="Gemini RAG Assistant",
                        height=600,
                        show_label=True,
                        bubble_full_width=False,
                        avatar_images=(None, "https://www.gstatic.com/lamda/images/gemini_sparkle_v002_d4735304ff6292a690345.svg")
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Ask a question",
                            placeholder="Type your question about the uploaded documents...",
                            lines=2,
                            scale=4,
                            container=False
                        )
                        send_btn = gr.Button("📤 Send", variant="primary", scale=1, size="lg")
            
            # Event handlers
            file_upload.change(
                fn=self.upload_files,
                inputs=[file_upload],
                outputs=[upload_status]
            )
            
            msg_input.submit(
                fn=self.chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            send_btn.click(
                fn=self.chat,
                inputs=[msg_input, chatbot],
                outputs=[msg_input, chatbot]
            )
            
            clear_btn.click(
                fn=self.clear_chat,
                outputs=[chatbot]
            )
            
            reset_btn.click(
                fn=self.reset_system,
                outputs=[upload_status, chatbot]
            )
            
            # Instructions
            gr.Markdown("""
            ## 📋 How to Use
            
            ### 1. 🔑 Setup
            - Ensure you have set the `GOOGLE_API_KEY` environment variable
            - Make sure Qdrant is running (Docker: `docker run -p 6333:6333 qdrant/qdrant`)
            
            ### 2. 📄 Upload Documents
            - Click "Upload PDF Documents" and select your files
            - Wait for the "Successfully processed" confirmation
            - Documents are chunked and embedded using Gemini
            
            ### 3. 💬 Ask Questions
            - Type questions in the chat interface
            - Get answers with precise source citations (file + page number)
            - Ask follow-up questions - the system remembers context
            
            ### 4. 🎯 Features
            - **Smart Retrieval**: Semantic search with Gemini embeddings
            - **Accurate Citations**: Every answer includes source references
            - **Conversation Memory**: Maintains context across questions
            - **No Hallucinations**: Returns "No answer found" when appropriate
            - **Multi-language Support**: Powered by Gemini's multilingual capabilities
            
            ### 5. 💡 Example Questions
            ```
            "What is the main topic of this document?"
            "Can you summarize the key findings from page 5?"
            "What does the document say about [specific topic]?"
            "Based on our previous discussion, can you elaborate on [topic]?"
            ```
            """)
        
        return interface

def main():
    """Main function to run the Gradio demo"""
    # Check environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY environment variable is required")
        print("Get your API key from: https://makersuite.google.com/app/apikey")
        return
    
    # Create and launch interface
    demo = GradioRAGDemo()
    interface = demo.create_interface()
    
    # Launch with appropriate settings
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        favicon_path=None
    )

if __name__ == "__main__":
    main()
