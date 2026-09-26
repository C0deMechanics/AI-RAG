from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b",vtemperature=0)

question = "What is Retrieval Augmented Generation? Anser in 1 sentence."

response = llm.invoke(question)

print("=" * 80)
print("QUESTION")
print("=" * 80)
print(question)

print("\n" + "=" * 80)
print("QWEN RESPONSE")
print("=" * 80)
print(response.content)