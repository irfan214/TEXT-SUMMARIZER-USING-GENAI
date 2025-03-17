import torch
import gradio as gr
from transformers import pipeline

# Load summarization model
pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

# Define a function for summarization
def summarize_text(text):
    summary = pipe(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Create Gradio UI
iface = gr.Interface(
    fn=summarize_text,  # Function to call
    inputs=gr.Textbox(lines=5, placeholder="Enter text to summarize..."),  # Input box
    outputs=gr.Textbox(),  # Output box
    title="AI Text Summarizer",
    description="Enter a long text and get a summarized version using AI.",
)

# Launch the web app
if __name__ == "__main__":
    iface.launch()

