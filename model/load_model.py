import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load the fine-tuned model and tokenizer
def load_summarization_model(model_path='./model'):
    """
    Load the fine-tuned BART model and tokenizer for summarization.
    """
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartForConditionalGeneration.from_pretrained(model_path)
    
    return tokenizer, model

def summarize(text, tokenizer, model):
    """
    Summarize the input text using the loaded model and tokenizer.
    """
    # Tokenize the input text
    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    
    # Generate summary (you can tweak num_beams and max_length for better results)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)
    
    # Decode the generated summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return summary

if __name__ == "__main__":
    # Example usage of loading and summarizing
    tokenizer, model = load_summarization_model()
    
    sample_text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents": any device that perceives its environment and takes actions that maximize its chance of successfully achieving its goals.
    """
    
    summary = summarize(sample_text, tokenizer, model)
    print("Summary:", summary)
