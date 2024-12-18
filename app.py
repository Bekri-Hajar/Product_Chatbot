from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pinecone import Pinecone
import tiktoken

# Loading the environment variables from the dotenv file
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "medicalchatbot"
index = pc.Index(index_name)
# Load the tokenizer and model from HuggingFace
model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name,token=os.getenv("HuggingFace_API_KEY"),legacy=False)
model = AutoModelForCausalLM.from_pretrained(model_name,token=os.getenv("HuggingFace_API_KEY"),legacy=False)

# Preprocessing the PDF content
def preprocess_text(text):
    text = re.sub(r"Page \d+|Chapter \d+", "", text)  # Remove page numbers and chapters
    text = re.sub(r"\s+", " ", text)  # Normalize spaces
    return text

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Preload the PDF text into the application
pdf_path = "C:\\Users\\hajar\\OneDrive\\Documents\\natureleaveschatbot\\data\\book.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
cleaned_text = preprocess_text(extracted_text)

# Function to generate embeddings for the text (using an embedding model)
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=512, temperature=0.7, num_return_sequences=1)
    embedding = outputs[0].detach().cpu().numpy()  # Example: you should extract the embedding here
    return embedding

# Function to index chunks of text in Pinecone
def index_text_in_pinecone(text):
    chunks = text.split('\n')  # Split text into chunks, 
    for chunk in chunks:
        embedding = embed_text(chunk)
        index.upsert([(str(hash(chunk)), embedding)])  # Using hash as an ID for each chunk

# Index the book text (only need to do this once)
index_text_in_pinecone(cleaned_text)

# Function to generate chatbot responses using Llama2 model
def chat_with_llama2(prompt, context, template=None):
    if not template:
        template = """You are an expert in cosmetics products and skincare. The user will ask you questions about a cosmetic product or their skin type, 
        what they can buy or do to have better skin. You will answer them, providing recommendations based on the book and the context. 
        Please answer concisely and professionally, including relevant examples or details when appropriate."""
    
    full_prompt = f"{context}\n\n{template}\n\nUser: {prompt}\nChatbot:"

    # Tokenize the input and generate a response
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    
    with torch.no_grad():
        outputs = model.generate(inputs["input_ids"], max_length=512, temperature=0.7, num_return_sequences=1)
    
    # Decode the model output and return the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response[len(full_prompt):].strip()  # Get the part after the prompt
    
    return answer if answer else "I couldn't generate a response."

# API endpoint for the chatbot
@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get('question')
    if user_query:
        # Searching for the most relevant context using Pinecone (vector search)
        query_embedding = embed_text(user_query)
        result = index.query([query_embedding], top_k=1)  # Query Pinecone for the top chunk
        best_match = result['matches'][0]['id']  # Get the top matching chunk

        # Using the matched chunk as the context for generating a response
        context = f"The following is information from a skincare book:\n{best_match}"
        answer = chat_with_llama2(user_query, context, template="Provide skin care advice based on the book.")
        return jsonify({'answer': answer})
    return jsonify({'error': 'No question provided'}), 400

if __name__ == "__main__":
    app.run(debug=True)


