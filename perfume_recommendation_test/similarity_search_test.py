import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import hashlib, json, torch, os
from langchain.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

chroma_client = chromadb.PersistentClient(path="chroma_db")

embedding_model = SentenceTransformer("snunlp/KLUE-SRoBERTa-Large-SNUExtended-klueNLI-klueSTS") # truncate_dim=384
# embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="snunlp/KLUE-SRoBERTa-Large-SNUExtended-klueNLI-klueSTS")
# embedding_function = SentenceTransformerEmbeddings(model_name="snunlp/KLUE-SRoBERTa-Large-SNUExtended-klueNLI-klueSTS")
# embedding_function = SentenceTransformerEmbeddings(model_name="distiluse-base-multilingual-cased-v2")
# paraphrase-multilingual-MiniLM-L12-v2
embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model_name="snunlp/KLUE-SRoBERTa-Large-SNUExtended-klueNLI-klueSTS"
)

# TODO: 향료정보 가져와서 향의 느낌을 GPT에게 설명시킨 후 캐시에 저장.
#       저장된 향 설명을 가져와 Chroma DB에 임베딩 저장

def initialize_vector_db(perfume_data):
    """Initialize Chroma DB and store embeddings."""
    # collection = chroma_client.get_or_create_collection("perfume_embeddings", embedding_function=get_sentence_embedding)
    # collection = chroma_client.get_or_create_collection("perfume_embeddings", embedding_function=embedding_function)
    # collection = chroma_client.get_or_create_collection("perfume_embeddings")
    collection = chroma_client.get_or_create_collection(name="embeddings", embedding_function=embedding_function)

    # Use OpenAI for text embeddings
    # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Insert vectors for each perfume
    for perfume in perfume_data:
        combined_text = f"{perfume['brand']}\n{perfume['name_kr']} ({perfume['name_en']})\n{perfume['content']}"
        
        # Compute the embedding
        # embeddings = embeddings.embed_query(combined_text)
        # embeddings = embedding_model.encode([combined_text])
        # embeddings=get_sentence_embedding([combined_text])
        
        embeddings = embedding_model.encode([combined_text])
        # print("💟💟💟💟💟",embeddings.shape)

        # Store in Chroma
        collection.add(
            documents=[combined_text],
            # embeddings = embeddings,
            metadatas=[{"id": perfume["id"], "name_kr": perfume["name_kr"], "brand": perfume["brand"], "category_id": perfume["category_id"]}],
            ids=[str(perfume["id"])]
        )

    print("Diffuser data has been embedded and stored in Chroma.")

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

    query_text = "아쿠아 디 파르마 브랜드의 우디한 향을 가진 디퓨저를 추천해주세요."
    results = collection.query(
        query_texts=[query_text],
        n_results=5,
        # where={"brand": "딥티크"},  # Optional filter
    )

    print("Query Results:")
    print(results)