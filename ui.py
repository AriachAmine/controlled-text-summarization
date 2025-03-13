import gradio as gr
from model import generate_stylized_summary

def create_ui(model, tokenizer):
    """Create a Gradio interface for the stylized summarizer"""
    def summarize(text, style):
        return generate_stylized_summary(text, model, tokenizer, style)
    
    interface = gr.Interface(
        fn=summarize,
        inputs=[
            gr.Textbox(lines=10, placeholder="Enter text to summarize..."),
            gr.Dropdown(["formal", "informal", "humorous", "poetic"], label="Summary Style")
        ],
        outputs="text",
        title="Creative Text Summarization with Style Control",
        description="Generate summaries in different styles (formal, informal, humorous, poetic)"
    )
    
    return interface
