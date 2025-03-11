from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModelForQuestionAnswering, AutoTokenizer, pipeline, Trainer, TrainingArguments
from sentence_transformers import SentenceTransformer, util
from datasets import Dataset
import json
import os
import numpy as np
import subprocess  # To run the training script

app = Flask(__name__)
CORS(app)

# File to store the dataset
DATASET_FILE = "question_answer_dataset.json"

# Load pre-trained T5 model and tokenizer for question generation
model_name = "valhalla/t5-small-qg-hl"  # Pre-trained model for question generation
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)

# Load the trained QA model and tokenizer
qa_model_path = "./qa-model"  # Path where your QA model is saved
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_path)
qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_path)

# Create a question-answering pipeline
qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

# Load Sentence Transformer for context retrieval
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the dataset (context) from question_answer_dataset.json
def load_dataset():
    """Load the existing dataset from the file."""
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r") as f:
            return json.load(f)
    return []

def save_dataset(dataset):
    """Save the dataset to the file."""
    with open(DATASET_FILE, "w") as f:
        json.dump(dataset, f, indent=4)

def generate_questions(paragraph, num_questions=5):
    """Generate questions using the T5 model."""
    input_text = "generate questions: " + paragraph
    input_ids = t5_tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = t5_model.generate(
        input_ids,
        max_length=640,
        num_return_sequences=num_questions,
        num_beams=5,
        early_stopping=True
    )

    questions = [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

def generate_when_where_questions(paragraph):
    """Generate 'when' and 'where' questions using rule-based methods."""
    questions = []
    sentences = paragraph.split(". ")  # Split paragraph into sentences

    for sentence in sentences:
        if " in " in sentence or " at " in sentence:  # Detect location-related phrases
            questions.append(f"Where {sentence}?")
        if " in " in sentence or " on " in sentence or " during " in sentence:  # Detect time-related phrases
            questions.append(f"When {sentence}?")
    return questions

def create_dataset(paragraph, num_questions=10):
    """Create a dataset by generating questions and appending to the existing dataset."""
    # Load the existing dataset
    dataset = load_dataset()

    # Generate questions using the model
    model_questions = generate_questions(paragraph, num_questions)

    # Generate "when" and "where" questions using rule-based method
    when_where_questions = generate_when_where_questions(paragraph)

    # Combine all questions
    all_questions = model_questions + when_where_questions

    # Append new questions to the dataset
    for question in all_questions:
        dataset.append({
            "context": paragraph,
            "question": question,
            "answers": {
                "text": [paragraph],  # Placeholder for answers (can be improved)
                "answer_start": [0]   # Placeholder for answer start position
            }
        })

    # Save the updated dataset
    save_dataset(dataset)
    return dataset

def train_model():
    """Train the QA model using the updated dataset."""
    # Load the dataset
    with open(DATASET_FILE, "r") as f:
        dataset = json.load(f)

    # Convert to Hugging Face Dataset
    dataset = Dataset.from_dict({
        "context": [item["context"] for item in dataset],
        "question": [item["question"] for item in dataset],
        "answers": [item["answers"] for item in dataset]
    })

    # Load pre-trained model and tokenizer
    model_name = "bert-base-uncased"  # Use BERT for question answering
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess the dataset
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Extract start and end positions
        start_positions = []
        end_positions = []
        for i, offset in enumerate(inputs["offset_mapping"]):
            answer = examples["answers"][i]
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            # Find the token positions for the start and end of the answer
            start_token = -1
            end_token = -1
            for idx, (start, end) in enumerate(offset):
                if start <= start_char < end:
                    start_token = idx
                if start < end_char <= end:
                    end_token = idx
            start_positions.append(start_token)
            end_positions.append(end_token)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True)

    # Training arguments (disable evaluation)
    training_args = TrainingArguments(
        output_dir="./qa-model",
        evaluation_strategy="no",  # Disable evaluation
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        save_steps=500,
        logging_dir="./logs",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained("./qa-model")
    tokenizer.save_pretrained("./qa-model")

    print("Model trained and saved to './qa-model'.")

@app.route('/api/generatedataset', methods=['POST'])
def generate_dataset():
    """Endpoint to generate and append questions to the dataset."""
    # Get the JSON data from the POST request
    data = request.get_json()

    # Extract paragraph and number of questions from the request
    paragraph = data.get('paragraph', '')
    num_questions = data.get('num_questions', 10)

    # Generate dataset and append new questions
    dataset = create_dataset(paragraph, num_questions=int(num_questions))

    # Train the model after updating the dataset
    train_model()

    # Return the updated dataset as JSON
    return jsonify(dataset), 200

@app.route('/ask', methods=['POST'])
def ask():
    """Endpoint to answer questions using the QA model."""
    # Get the question from the API request
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Load the dataset (context) from question_answer_dataset.json
    dataset = load_dataset()
    knowledge_base = [item["context"] for item in dataset]

    # If no paragraphs exist, return an error
    if not knowledge_base:
        return jsonify({"error": "No paragraphs available in the knowledge base"}), 400

    # Create an embedding for the question
    question_embedding = sentence_model.encode([question], convert_to_tensor=True)

    # Compute embeddings for all available paragraphs in the knowledge base
    corpus_embeddings = sentence_model.encode(knowledge_base, convert_to_tensor=True)

    # Compute cosine similarity between the question embedding and the context corpus
    similarities = util.cos_sim(question_embedding.cpu().numpy(), corpus_embeddings.cpu().numpy())

    # Get the index of the most similar context
    best_match_idx = np.argmax(similarities)

    # Retrieve the context (most relevant to the question)
    context = knowledge_base[best_match_idx]

    # Perform question-answering using the context
    try:
        result = qa_pipeline(question=question, context=context)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
