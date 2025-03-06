# generateQuestionDataSet.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json

# Load pre-trained T5 model and tokenizer for question generation
model_name = "valhalla/t5-small-qg-hl"  # Pre-trained model for question generation
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def generate_questions(paragraph, num_questions=5):
    # Preprocess the input
    input_text = "generate questions: " + paragraph
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate questions
    outputs = model.generate(
        input_ids,
        max_length=64,
        num_return_sequences=num_questions,
        num_beams=5,
        early_stopping=True
    )

    # Decode the generated questions
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

def create_dataset(paragraph, num_questions=5):
    questions = generate_questions(paragraph, num_questions)
    dataset = []
    for question in questions:
        dataset.append({
            "context": paragraph,
            "question": question,
            "answers": {
                "text": [paragraph],  # Placeholder for answers (can be improved)
                "answer_start": [0]   # Placeholder for answer start position
            }
        })
    return dataset

# Example paragraph
paragraph = "The University of Notre Dame is a Catholic research university located in Notre Dame, Indiana. It was founded in 1842 by Father Edward Sorin."

# Generate dataset
dataset = create_dataset(paragraph, num_questions=5)

# Save dataset to a JSON file
with open("question_answer_dataset.json", "w") as f:
    json.dump(dataset, f, indent=4)

print("Dataset generated and saved to 'question_answer_dataset.json'.")
