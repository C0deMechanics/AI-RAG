from langchain_ollama import OllamaEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

embeddings = OllamaEmbeddings(model="nomic-embed-text")

texts = [
    "What is RAG?",
    "Isn't avocados delicious?",
    ("RAG (Retrieval Augmented Generation) is an AI technique that combines "
    "information retrieval from external source with a generative LLM "
    "to produce accurate responses."),
]

vectors = embeddings.embed_documents(texts)

similarity = cosine_similarity(vectors)

print("Cosine Similarity:")

print(similarity)