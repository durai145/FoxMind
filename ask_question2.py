from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from sentence_transformers import SentenceTransformer, util
import json
import numpy as np

app = Flask(__name__)
CORS(app)
# Load the trained QA model and tokenizer
model_path = "./qa-model"  # Path where your model is saved
model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Create a question-answering pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Load Sentence Transformer for context retrieval
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the dataset (context) from question_answer_dataset.json
with open("question_answer_dataset.json", "r") as f:
    dataset = json.load(f)

# Extract the context (paragraphs) from the dataset
knowledge_base = [item["context"] for item in dataset]

@app.route('/ask', methods=['POST'])
def ask():
    # Get the question from the API request
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({"error": "Question is required"}), 400

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

