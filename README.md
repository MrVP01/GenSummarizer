# **GenSummarizer: LLM-Based Real-Time Summarization**

**GenSummarizer** is a real-time document summarization system built using **PyTorch**, **Hugging Face Transformers**, **Flask**, **Redis**, and **Docker**. It leverages a fine-tuned pre-trained **BART** (Bidirectional and Auto-Regressive Transformers) model to generate summaries of input documents with **sub-second latency** for up to **10,000+ documents**. The system includes optimizations such as caching via Redis to reduce response time by **40%**.

## **Features**

- Fine-tuned BART model for document summarization.
- Deployed API to summarize documents in real-time.
- Optimized for handling large-scale queries of 10,000+ documents.
- Sub-second response time via Redis caching and optimized inference.
- Dockerized for easy setup and deployment.

## **Technologies Used**

- **PyTorch**: For model fine-tuning and inference.
- **Hugging Face Transformers**: Pre-trained BART model for summarization.
- **Flask**: API deployment for real-time document summarization.
- **Redis**: Caching mechanism to reduce response time and handle repeated queries.
- **Docker**: To containerize the entire application for easy setup and deployment.

## **Project Structure**

```plaintext
gen-summarizer/
│
├── docker-compose.yml         # Docker Compose file
├── Dockerfile                 # Dockerfile for Flask and model
│
├── model/                     # Model training and loading scripts
│   └── train.py               # Script to fine-tune the LLM for summarization
│   └── load_model.py          # Script to load the trained model for inference
│
├── app/                       # Flask app for real-time summarization
│   └── app.py                 # Flask application to handle requests
│
├── data/                      # Directory for document data
│   └── sample_documents.csv   # Sample document data
│
├── cache/                     # Cache handling
│   └── redis_cache.py         # Redis cache logic for optimizing latency
│
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation and setup instructions
