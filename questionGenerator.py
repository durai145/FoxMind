from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load pre-trained T5 model and tokenizer
model_name = "valhalla/t5-small-qa-qg-hl"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Example context
context = "The Eiffel Tower is located in Paris, France."

# Tokenize input
input_text = f"generate questions: {context}"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate questions
outputs = model.generate(input_ids, max_length=50, num_return_sequences=1)
for question in outputs:
    print(question)
    questions = tokenizer.decode(question, skip_special_tokens=True)
    print(f"Generated Questions: {questions}")
