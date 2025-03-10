from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORSi
# generateQuestionDataSet.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
import json
app = Flask(__name__)
CORS(app)

# Load pre-trained T5 model and tokenizer for question generation
model_name = "valhalla/t5-small-qg-hl"  # Pre-trained model for question generation
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

def generate_questions(paragraph, num_questions=5):
    # Preprocess the input
    input_text = "generate questions: " + paragraph
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate questions
    #64
    outputs = model.generate(
        input_ids,
        max_length=6400,
        num_return_sequences=num_questions,
        num_beams=5,
        early_stopping=True
    )

    # Decode the generated questions
    questions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

def generate_when_where_questions(paragraph):
    # Rule-based question generation for "when" and "where"
    questions = []
    sentences = paragraph.split(". ")  # Split paragraph into sentences

    for sentence in sentences:
        if " in " in sentence or " at " in sentence:  # Detect location-related phrases
            questions.append(f"Where {sentence}?")
        if " in " in sentence or " on " in sentence or " during " in sentence:  # Detect time-related phrases
            questions.append(f"When {sentence}?")
    return questions

def create_dataset(paragraph, num_questions=10):
    # Generate questions using the model
    model_questions = generate_questions(paragraph, num_questions)

    # Generate "when" and "where" questions using rule-based method
    when_where_questions = generate_when_where_questions(paragraph)

    # Combine all questions
    all_questions = model_questions + when_where_questions

    # Create dataset
    dataset = []
    for question in all_questions:
        dataset.append({
            "context": paragraph,
            "question": question,
            "answers": {
                "text": [paragraph],  # Placeholder for answers (can be improved)
                "answer_start": [0]   # Placeholder for answer start position
            }
        })
    return dataset


# Save dataset to a JSON file
#with open("question_answer_dataset.json", "w") as f:
#    json.dump(dataset, f, indent=4)

#print("Dataset generated and saved to 'question_answer_dataset.json'.")

# Initialize Flask app
#app = Flask(__name__)

# Define the POST endpoint
@app.route('/api/generatedataset', methods=['POST'])
def post_example():
    # Get the JSON data from the POST request
    data = request.get_json()

    # Example of processing the data (you can modify it according to your need)
    paragraph = data.get('paragraph', 'Anonymous')
    num_questions = data.get('num_questions', 'Anonymous')

    # Example paragraph
    #paragraph = "The University of Notre Dame is a Catholic research university located in Notre Dame, Indiana. It was founded in 1842 by Father Edward Sorin."

    # Generate dataset
    dataset = create_dataset(paragraph, num_questions=int(num_questions))
    with open("question_answer_dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)

    '''
    # Create a response
    response = {
        'message': f'Hello, {name}. You are {age} years old.',
        'status': 'success'
    }
    '''
    # Return the response as JSON
    return jsonify(dataset), 200

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
