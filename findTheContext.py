from sentence_transformers import SentenceTransformer, util

# Load a pre-trained model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode paragraphs and question
paragraphs = ["Paragraph 1 text...", "Paragraph 2 text...", "Paragraph 3 text...", "The University of Notre Dame is a Catholic research university located in Notre Dame, Indiana. It was founded in 1842 by Father Edward Sorin."]
question = "Where is the University of Notre Dame located?"

paragraph_embeddings = model.encode(paragraphs)
question_embedding = model.encode(question)

# Find the most relevant paragraph
scores = util.cos_sim(question_embedding, paragraph_embeddings)
print(scores)
most_relevant_index = scores.argmax()
context = paragraphs[most_relevant_index]

print(f"context:{context}")
