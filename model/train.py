import pandas as pd
from transformers import BartForConditionalGeneration, BartTokenizer, Trainer, TrainingArguments

# Load the dataset (assuming 'text' and 'summary' columns)
data = pd.read_csv('./data/sample_documents.csv')

# Preprocess the dataset
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')

def preprocess_function(examples):
    inputs = [doc for doc in examples['text']]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples['summary'], max_length=150, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Define the model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Training arguments
training_args = TrainingArguments(
    output_dir='./model',
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_steps=10_000,
    save_total_limit=2,
    logging_dir='./logs',
)

# Define the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=data["train"],
    eval_dataset=data["test"],
    tokenizer=tokenizer,
)

# Fine-tune the model
trainer.train()

# Save the model
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
