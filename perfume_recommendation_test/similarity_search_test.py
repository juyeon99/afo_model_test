import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
import os
from langchain_community.embeddings import OpenAIEmbeddings
import hashlib
import json
import torch

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

chroma_client = chromadb.PersistentClient(path="perfume_db")

def initialize_vector_db(perfume_data):
    """Initialize Chroma DB and store embeddings."""
    collection = chroma_client.get_or_create_collection("perfume_embeddings")

    # Use OpenAI for text embeddings
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Insert vectors for each perfume
    for perfume in perfume_data:
        combined_text = f"{perfume['brand']}\n{perfume['name_kr']} ({perfume['name_en']})\n{perfume['main_accord']}\n{perfume['content']}"
        
        # Compute the embedding
        # embedding = embeddings.embed_query(combined_text)

        # Store in Chroma
        collection.add(
            documents=[combined_text],
            metadatas=[{"id": perfume["id"], "name_kr": perfume["name_kr"], "brand": perfume["brand"], "category_id": perfume["category_id"]}],
            ids=[str(perfume["id"])]
        )

    print("Perfume data has been embedded and stored in Chroma.")

    return collection

CACHE_DIR = "./cache"
CACHE_FILE_PATH = os.path.join(CACHE_DIR, "product_cache.json")
HASH_FILE_PATH = os.path.join(CACHE_DIR, "product_hash.txt")

def compute_file_hash(file_path):
    """Compute the MD5 hash of a file."""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except FileNotFoundError:
        return None

def load_perfume_data(json_path):
    """Load perfume data with caching support."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)

    # Compute hash of the current product.json
    current_hash = compute_file_hash(json_path)

    # Check if cache exists and matches the file hash
    if os.path.exists(CACHE_FILE_PATH) and os.path.exists(HASH_FILE_PATH):
        with open(HASH_FILE_PATH, "r") as hash_file:
            cached_hash = hash_file.read().strip()

        # If hash matches, load from cache
        if cached_hash == current_hash:
            try:
                with open(CACHE_FILE_PATH, "r", encoding="utf-8") as cache_file:
                    print("Loading data from cache.")
                    return json.load(cache_file)
            except json.JSONDecodeError:
                print("Cache file is corrupted. Reloading from source.")
    
    # Load from source and update cache
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Save to cache
            with open(CACHE_FILE_PATH, "w", encoding="utf-8") as cache_file:
                json.dump(data, cache_file, ensure_ascii=False, indent=4)
            # Save hash
            with open(HASH_FILE_PATH, "w") as hash_file:
                hash_file.write(current_hash)
            print("Data loaded and cache updated.")
            return data
    except FileNotFoundError:
        print(f"File {json_path} not found.")
        return []
    except json.JSONDecodeError:
        print(f"Invalid JSON format in {json_path}.")
        return []

if __name__ == "__main__":
    perfume_data_path = "product.json"
    perfume_data = load_perfume_data(perfume_data_path)
    
    if not perfume_data:
        print("No perfume data available.")
        exit()
    
    collection = initialize_vector_db(perfume_data)

    query_text = "상쾌한 향, 샤넬 제품을 선호"
    results = collection.query(
        query_texts=[query_text],
        n_results=3,
        # where={"brand": "크리드"},  # Optional filter
    )

    print("Query Results:")
    print(results)





# import json
# import os
# import chromadb

# # Initialize Chroma Persistent Client
# client = chromadb.PersistentClient(path="perfume_db")
# collection = client.get_or_create_collection("perfume_collection")

# # Load perfume data from JSON
# def load_perfume_data(json_path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         return json.load(f)

# perfume_data_path = "product.json"  # Update this to your JSON file path
# perfume_data = load_perfume_data(perfume_data_path)

# # Prepare documents, metadata, and IDs for ChromaDB
# documents = []
# metadatas = []
# ids = []

# for idx, perfume in enumerate(perfume_data):
#     # Format the document string as required
#     document_text = f"{perfume['brand']}\n{perfume['name_kr']} ({perfume['name_en']})\n{perfume['main_accord']}\n{perfume['content']}"
#     documents.append(document_text)
    
#     # Metadata for filtering
#     metadata = {
#         "brand": perfume["brand"],
#         "category_id": perfume["category_id"],
#         "id": perfume["id"]
#     }
#     metadatas.append(metadata)
    
#     # Unique ID
#     ids.append(str(perfume["id"]))

# # Add data to the collection
# collection.add(
#     documents=documents,
#     metadatas=metadatas,
#     ids=ids,
# )

# # Query/search for perfumes
# query_text = "고급진 향"
# results = collection.query(
#     query_texts=[query_text],
#     n_results=3,
#     where={"brand": "크리드"},  # Optional filter
# )

# print("Query Results:")
# print(results)
