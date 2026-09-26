from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings( model="nomic-embed-text")

vector = embeddings.embed_query("What is Retrieval Augmented Generation?")

print(f"Dimensions: {len(vector)}")

print(vector[:10])