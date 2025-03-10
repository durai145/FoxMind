# trainModel.py
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

# Load the dataset
with open("question_answer_dataset.json", "r") as f:
    dataset = json.load(f)

# Convert to Hugging Face Dataset
dataset = Dataset.from_dict({
    "context": [item["context"] for item in dataset],
    "question": [item["question"] for item in dataset],
    "answers": [item["answers"] for item in dataset]
})

# Load pre-trained model and tokenizer
'''
model_name = "bert-base-uncased"  # Use BERT for question answering
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
'''
model_name = "deepseek-ai/deepseek-llm-7b-base"  # Use BERT for question answering
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
#384
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
