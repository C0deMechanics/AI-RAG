from langchain_ollama import OllamaEmbeddings, ChatOllama
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. Initialize models
# --------------------------------------------------
embeddings = OllamaEmbeddings(model="nomic-embed-text")

llm = ChatOllama(model="qwen2.5:7b", temperature=0)

# --------------------------------------------------
# 2. Knowledge base
# --------------------------------------------------
documents = [
    (
        "RAG (Retrieval Augmented Generation) is an AI technique "
        "that combines information retrieval from external source "
        "with a generative LLM to produce accurate responses."
    ),
    (
        "RAG allows an LLM to use external information instead "
        "of relying only on information contained in its model."
    ),
    (
        "A typical RAG system first converts documents into "
        "embeddings and stores those embeddings in a vector database."
    ),
    (
        "During a query, the question is converted into an embedding "
        "and compared with document embeddings to find relevant information."
    ),
    (
        "The retrieved information is then provided to the language "
        "model as context for generating the final answer."
    ),
    (
        "Qwen 2.5 is a family of large language models that can "
        "generate natural language responses."
    ),
]

# --------------------------------------------------
# 3. Generate document embeddings
# --------------------------------------------------
document_vectors = embeddings.embed_documents(documents)

# --------------------------------------------------
# 4. User question
# --------------------------------------------------
question = "What is Retrieval Augmented Generation? Anser in 1 sentence."

# --------------------------------------------------
# 5. Embed question
# --------------------------------------------------
question_vector = embeddings.embed_query(question)

# --------------------------------------------------
# 6. Calculate similarity
# --------------------------------------------------
similarities = cosine_similarity([question_vector], document_vectors)[0]

# --------------------------------------------------
# 7. Retrieve top documents
# --------------------------------------------------
top_k = 1

top_indices = similarities.argsort()[-top_k:][::-1]

retrieved_documents = []

for index in top_indices:
    retrieved_documents.append(documents[index])
    print(f"Similarity: {similarities[index]:.4f}")
    print(f"Document: {documents[index]}\n")

# --------------------------------------------------
# 8. Build RAG context
# --------------------------------------------------
context = "\n\n".join(retrieved_documents)

# --------------------------------------------------
# 9. Create prompt for Qwen
# --------------------------------------------------
prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

# --------------------------------------------------
# 10. Ask Qwen
# --------------------------------------------------
response = llm.invoke(prompt)

# --------------------------------------------------
# 11. Display result
# --------------------------------------------------
print("=" * 80)
print("QUESTION")
print("=" * 80)
print(question)
print("\n" + "=" * 80)
print("RETRIEVED CONTEXT")
print("=" * 80)
print(context)
print("\n" + "=" * 80)
print("QWEN RAG RESPONSE")
print("=" * 80)
print(response.content)