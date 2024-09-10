from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer
import redis
import json

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer
model = BartForConditionalGeneration.from_pretrained('./model')
tokenizer = BartTokenizer.from_pretrained('./model')

# Initialize Redis for caching
cache = redis.StrictRedis(host='redis', port=6379, decode_responses=True)

# Function to summarize text
def summarize(text):
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# API endpoint for document summarization
@app.route('/summarize', methods=['POST'])
def get_summary():
    document = request.json.get('document')
    
    # Check cache for response
    cached_summary = cache.get(document)
    if cached_summary:
        return jsonify({"summary": cached_summary, "cached": True})
    
    # Summarize the document
    summary = summarize(document)
    
    # Store summary in Redis cache
    cache.set(document, summary)
    
    return jsonify({"summary": summary, "cached": False})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
