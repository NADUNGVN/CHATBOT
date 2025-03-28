def format_response(answer, source_info):
    """Format the chatbot's response for better readability."""
    formatted_answer = f"Answer: {answer}\n\nSource Information:\n"
    for info in source_info:
        formatted_answer += f"- Source: {info['source']}, Page: {info['page']}, Text: \"{info['text']}\"\n"
    return formatted_answer

def load_pdf_files(pdf_directory):
    """Load PDF files from a specified directory."""
    import os
    pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    return pdf_files

def clean_text(text):
    """Remove unnecessary whitespace and special characters from text."""
    import re
    return re.sub(r'\s+', ' ', text).strip()