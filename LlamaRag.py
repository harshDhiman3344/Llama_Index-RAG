import os
import sys
import logging
from llama_index.core import VectorStoreIndex,SimpleDirectoryReader, Settings
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

#LOCAL EMBEDDING KARUNGA AAAAA
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


#Logging Setup
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


os.environ["GOOGLE_API_KEY"] = "Your key"


Settings.llm = Gemini(model="gemini-2.5-flash")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")



print("Loading Docs")
documents = SimpleDirectoryReader("data").load_data()
print("Docs Loaded Len: ",len(documents))

print('Now Indexing')

try:
    # show_progress=True adds a progress bar so you know it's not stuck
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    print("Indexed successfully")
except Exception as e:
    print(f"Error during indexing: {e}")
    sys.exit(1)
query_engine = index.as_query_engine()

print("chat ready")
while True:
    question = input("\nYou:")

    if question.lower() in ["exit"]:
        break

    try:
        response = query_engine.query(question)
        print("\nAI: ", response)
    except Exception as e:
        print(f"\nError querying: {e}")

    


