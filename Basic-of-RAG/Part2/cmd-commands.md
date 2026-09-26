#### **\[Hardware Check]**

**VRAM → determines how large a model/workload can fit on the GPU and affects performance.**

nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,driver\_version --format=csv



**CUDA support → determines whether NVIDIA-accelerated software like PyTorch, WhisperX, Ollama, etc. can use the GPU.**

nvidia-smi | findstr /I "CUDA Version"



**System RAM - Holds models/data that don't fit in VRAM**

systeminfo | findstr /C:"Total Physical Memory"



**CPU - Handles preprocessing, tokenization, RAG, file processing, and CPU offloading**

echo %NUMBER\_OF\_PROCESSORS%

#### 

#### **\[Ollama Installation]**



**Ollama Download**

https://ollama.com/download/windows



**Ollama Version:**

ollama --version



**Verify Ollama is running**

ollama list



**Download Qwen 2.5:7B**

ollama pull qwen2.5:7b



**Download Nomic Embed Text**

ollama pull nomic-embed-text



**Test Qwen directly**

ollama run qwen2.5:7b



**Query:**

What is Retrieval Augmented Generation? Answer in 1 sentence.



**Exit with:**

/bye



#### **\[Testing RAG]**

**Python Dependencies:**

pip install langchain langchain-ollama scikit-learn





#### 





