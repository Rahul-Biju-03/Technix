import os
import re
import string
import torch
import gradio as gr
from transformers import pipeline, LlamaTokenizer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join(text.split())
    return text

# Postprocessing function
def postprocess_response(response):
    response = response.strip()
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', response)
    sentences = [sentence.capitalize() for sentence in sentences]
    response = ' '.join(sentences)
    return response

try:
    # Load the tokenizer and quantized model from the saved directory
    model_path = 'C:\\Users\\HP\\Desktop\\Intel_LLM\\.venv\\Pipeline\\FINAL'
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    logger.info("Tokenizer loaded successfully")
    

    logger.info(f"Loading model from {model_path}")
    pipe = pipeline("text-generation", model=model_path, tokenizer=tokenizer)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model or tokenizer: {e}")
    raise e

# Function to generate response
def generate_response(user_input):
    try:
        user_input = preprocess_text(user_input)
        logger.info(f"Generating response for input: {user_input}")
        response = pipe(user_input, max_length=100, temperature=0.7, top_k=50, top_p=0.9, num_return_sequences=1)[0]['generated_text']
        response = postprocess_response(response)
        logger.info(f"Response generated: {response}")
        return response
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Error generating response"

# Sample text generation prompts
sample_prompts = [
    "Python is one of the",
    "The Machine learning is",
    "OpenVINO is a",
    "The largest animal in the planet is",
    "Indian food is famous"
]

# Define Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Input your message here..."),
    outputs=gr.Textbox(label="Response"),
    title="TinyLlama Chatbot",
    description="Welcome to the TinyLlama Chatbot! How can I assist you today?",
    examples=sample_prompts,
    theme="default",
    allow_flagging="never"
)

# Launch the Gradio interface
try:
    logger.info("Launching Gradio interface...")
    iface.launch(share=True)
    logger.info("Gradio interface launched successfully")
except Exception as e:
    logger.error(f"Error launching Gradio interface: {e}")
