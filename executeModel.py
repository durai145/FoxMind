# executeModel.py
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
import torch

# Load the trained model and tokenizer
model_path = "./qa-model"
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Example paragraph and question
paragraph = "The University of Notre Dame is a Catholic research university located in Notre Dame, Indiana. It was founded in 1842 by Father Edward Sorin."
question = "Where is the University of Notre Dame located?"

# Answer the question
result = qa_pipeline(question=question, context=paragraph)
print(f"Question: {question}")
print(f"Answer: {result['answer']}")
