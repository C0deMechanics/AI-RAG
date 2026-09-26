from langchain_ollama import OllamaEmbeddings
from sklearn.decomposition import PCA

embeddings = OllamaEmbeddings(model="nomic-embed-text")

texts = [
    "What is RAG?",
    "Isn't avocados delicious?",
    ("RAG (Retrieval Augmented Generation) is an AI technique that combines "
    "information retrieval from external source with a generative LLM "
    "to produce accurate responses."),
]

# Generate 768-dimensional embeddings
vectors = embeddings.embed_documents(texts)

# Reduce 768 → 3 dimensions
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(vectors)

# Print each vector
for text, vector in zip(texts, vectors_3d):
    print(f"\nText: {text}")
    print(f"3D Vector: {vector}")