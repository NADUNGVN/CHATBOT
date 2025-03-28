import gradio as gr
from similarity_search import load_retrievers, build_rag_chatbots, process_query
from document_manager import DocumentManager
from config.settings import Config
import os

class ChatbotManager:
    def __init__(self):
        self.vectordbs = None
        self.retrievers = None
        self.qa_chains = None
        self.doc_manager = DocumentManager(Config.PDF_DIRECTORY, Config.CHROMA_PERSIST_DIRECTORY)
        
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(Config.PDF_DIRECTORY, exist_ok=True)
        os.makedirs(Config.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
        
        # Thử load vector databases nếu đã có
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize retrievers and qa_chains if documents exist"""
        if os.path.exists(Config.CHROMA_PERSIST_DIRECTORY) and os.listdir(Config.CHROMA_PERSIST_DIRECTORY):
            self.vectordbs, self.retrievers = load_retrievers(Config.CHROMA_PERSIST_DIRECTORY)
            if self.vectordbs:
                self.qa_chains = build_rag_chatbots(self.vectordbs)
                return True
        return False
    
    def process_chat(self, message, history):
        if not self.qa_chains:
            return history + [(message, "Vui lòng tải lên tài liệu PDF trước khi chat.")]
        
        try:
            # Validate input
            if not message or not message.strip():
                return history
                
            # Get context from recent history
            recent_context = history[-3:] if history else []
            
            # Process query
            result = process_query(message, self.qa_chains, self.retrievers)
            
            # Format response with sources
            response = result["answer"]
            if result.get("source_info"):
                response += "\n\nNguồn tham khảo:"
                for i, info in enumerate(result["source_info"], 1):
                    response += f"\n{i}. {info['source']} (Trang {info['page']})"
            
            # Update history
            history = history or []
            history.append((message, response))
            
            # Limit history length
            max_history = 10
            if len(history) > max_history:
                history = history[-max_history:]
                
            return history
        except Exception as e:
            error_msg = f"Xin lỗi, có lỗi xảy ra: {str(e)}"
            return history + [(message, error_msg)]
    
    def handle_upload(self, files):
        try:
            self.doc_manager.add_documents(files)
            success = self.initialize_components()
            if success:
                return self.doc_manager.get_document_list(), "Tải lên và xử lý tài liệu thành công"
            else:
                return None, "Lỗi: Không thể tạo vector database"
        except Exception as e:
            return None, f"Lỗi: {str(e)}"
    
    def handle_delete(self, selected_rows):
        try:
            if selected_rows is None or len(selected_rows) == 0:
                return self.doc_manager.get_document_list(), "Vui lòng chọn tài liệu cần xóa"
                
            self.doc_manager.delete_documents(selected_rows)
            success = self.initialize_components()
            if success:
                return self.doc_manager.get_document_list(), "Xóa tài liệu thành công"
            else:
                # Reset components if no documents remain
                self.vectordbs = None
                self.retrievers = None
                self.qa_chains = None
                return self.doc_manager.get_document_list(), "Đã xóa tất cả tài liệu"
        except Exception as e:
            return None, f"Lỗi: {str(e)}"

# Create global chatbot manager
chatbot_manager = ChatbotManager()

def main():
    with gr.Blocks(css="style.css") as demo:
        with gr.Tabs() as tabs:
            # Removed all authentication code
            with gr.Tab("Quản lý tài liệu"):
                gr.Markdown("""### Quản lý tài liệu PDF""")
                
                with gr.Row():
                    pdf_files = gr.File(
                        file_count="multiple",
                        file_types=[".pdf"],
                        label="Upload tài liệu PDF"
                    )
                    upload_button = gr.Button("Tải lên", variant="primary")
                
                file_list = gr.Dataframe(
                    headers=["Tên tài liệu", "Ngày tải lên", "Kích thước", "Trạng thái"],
                    interactive=True,  # Make it interactive to allow selection
                    label="Danh sách tài liệu"
                )
                
                delete_button = gr.Button("Xóa tài liệu đã chọn", variant="secondary")
                status_text = gr.Textbox(label="Trạng thái", interactive=False)
                
                upload_button.click(
                    chatbot_manager.handle_upload,
                    inputs=[pdf_files],
                    outputs=[file_list, status_text]
                )
                
                delete_button.click(
                    chatbot_manager.handle_delete,
                    inputs=[file_list],
                    outputs=[file_list, status_text]
                )
                
                # Update file list on tab change
                tabs.change(
                    lambda: chatbot_manager.doc_manager.get_document_list(),
                    None,
                    file_list
                )
            
            with gr.Tab("Chat"):
                gr.Markdown(
                    """
                    # RAG Chatbot Hỗ Trợ Thủ Tục Hành Chính
                    Hãy đặt câu hỏi về các thủ tục hành chính liên quan đến hộ tịch
                    """
                )
                
                chatbot = gr.Chatbot(
                    height=600,
                    show_label=False,
                    container=True,
                    bubble_full_width=False,
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        show_label=False,
                        placeholder="Nhập câu hỏi của bạn...",
                        container=False
                    )
                    submit = gr.Button("Gửi", variant="primary")
                
                with gr.Row():
                    clear = gr.Button("Xóa lịch sử")
                
                # Set up event handlers with proper history management
                submit_click = submit.click(
                    chatbot_manager.process_chat,
                    inputs=[msg, chatbot],
                    outputs=chatbot,
                )
                
                # Clear text box after sending
                submit_click.then(lambda: "", None, msg)
                
                # Handle Enter key submission
                msg.submit(
                    chatbot_manager.process_chat,
                    inputs=[msg, chatbot],
                    outputs=chatbot,
                ).then(lambda: "", None, msg)
                
                # Clear history properly
                clear.click(lambda: [], None, chatbot)

    # Simplified launch settings for public access
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        height=800
    )

if __name__ == "__main__":
    main()