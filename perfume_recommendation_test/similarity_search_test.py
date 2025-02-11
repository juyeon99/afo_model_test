# Ref: https://cookbook.chromadb.dev/faq/#how-to-set-dimensionality-of-my-collections
import chromadb, logging
from chromadb.config import Settings
from dotenv import load_dotenv
from chromadb.utils import embedding_functions
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import hashlib, json, torch, os
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

chroma_client = chromadb.PersistentClient(path="chroma_db")

embedding_model = SentenceTransformer("snunlp/KLUE-SRoBERTa-Large-SNUExtended-klueNLI-klueSTS")
embedding_function = embedding_functions.HuggingFaceEmbeddingFunction(
    api_key=os.getenv("HUGGINGFACE_API_KEY"),
    model_name="snunlp/KLUE-SRoBERTa-Large-SNUExtended-klueNLI-klueSTS"
)

def load_diffuser_scent_data(json_path):
    """Load diffuser scent descriptions."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return {item["id"]: item["scent_description"] for item in json.load(f)}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading diffuser scent data: {e}")
        return {}

def initialize_vector_db(perfume_data, diffuser_scent_data):
    """Initialize Chroma DB and store embeddings."""
    collection = chroma_client.get_or_create_collection(name="embeddings", embedding_function=embedding_function)

    # Fetch existing IDs from the collection
    existing_ids = set()
    try:
        results = collection.get()
        existing_ids.update(results["ids"])
    except Exception as e:
        logger.error(f"Error fetching existing IDs: {e}")

    # Insert vectors for each perfume if not already in collection
    for perfume in perfume_data:
        if str(perfume["id"]) in existing_ids:
            # logger.info(f"Skipping perfume ID {perfume['id']} (already in collection).")
            continue
        
        logger.info(f"Inserting vectors for ID {perfume['id']}.")
        scent_description = diffuser_scent_data.get(perfume["id"], "")
        combined_text = f"{perfume['brand']}\n{perfume['name_kr']} ({perfume['name_en']})\n{scent_description}"

        # Store in Chroma
        collection.add(
            documents=[combined_text],
            metadatas=[{"id": perfume["id"], "name_kr": perfume["name_kr"], "brand": perfume["brand"], "category_id": perfume["category_id"]}],
            ids=[str(perfume["id"])]
        )
    logger.info(f"Perfume and diffuser data have been embedded and stored in Chroma.")

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
                    logger.info(f"Loading data from cache.")
                    return json.load(cache_file)
            except json.JSONDecodeError:
                logger.warning(f"Cache file is corrupted. Reloading from source.")
    
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
            logger.info(f"Data loaded and cache updated.")
            return data
    except FileNotFoundError:
        logger.info(f"File {json_path} not found.")
        return []
    except json.JSONDecodeError:
        logger.warning(f"Invalid JSON format in {json_path}.")
        return []

def get_distinct_brands(perfume_data):
    """Return all distinct diffuser brands from the perfume data."""
    brands = set()
    for perfume in perfume_data:
        brands.add(perfume.get("brand", "Unknown"))
    return brands

def get_fragrance_recommendation(user_input, caption, existing_brands):
    brands_str = ", ".join(existing_brands)

    prompt = f"""You are a fragrance expert with in-depth knowledge of various scents. Based on the User Input and Image Caption, **imagine** and provide a fragrance scent description that matches the room's description and the user's request. Your task is to creatively describe a fragrance that would fit well with the mood and characteristics of the room as described in the caption, as well as the user's scent preference. Do not mention specific diffuser or perfume products.

### Instructions:
- **Existing Brands**: {brands_str}
1. **If a specific brand is mentioned**, check if it exists in the list of existing brands above. If it does, acknowledge the brand name without referring to any specific product and describe a fitting scent that aligns with the user's request.  
**IF THE BRAND IS MENTIONED IN THE USER INPUT BUT IS NOT FOUND IN THE EXISTING BRANDS LIST, START BY 'Not Found' TO SAY THE BRAND DOES NOT EXIST.**
2. **If the brand is misspelled or doesn't exist**, please:
    - Correct the spelling if the brand is close to an existing brand (e.g., "아쿠아 파르마" -> "아쿠아 디 파르마").
    - **IF THE BRAND IS MENTIONED IN THE USER INPUT BUT IS NOT FOUND IN THE EXISTING BRANDS LIST, START BY 'Not Found' TO SAY THE BRAND DOES NOT EXIST.** Then, recommend a suitable fragrance based on the context and preferences described in the user input.
3. Provide the fragrance description in **Korean**, focusing on key scent notes and creative details that align with the mood and characteristics described in the user input and image caption. Do **not mention specific diffuser or perfume products.**

### Example Responses:

#### Example 1 (when a brand is mentioned, but with a minor spelling error):
- User Input: 아쿠아 파르마의 우디한 베이스를 가진 디퓨저를 추천해줘.
- Image Caption: The image shows a modern living room with a large window on the right side. The room has white walls and wooden flooring. On the left side of the room, there is a gray sofa and a white coffee table with a black and white patterned rug in front of it. In the center of the image, there are six black chairs arranged around a wooden dining table. The table is set with a vase and other decorative objects on it. Above the table, two large windows let in natural light and provide a view of the city outside. A white floor lamp is placed on the floor next to the sofa.
- Response:
  - Brand: 아쿠아 디 파르마
  - Scent Description: 우디한 느낌이 강조된 따뜻하고 차분한 향. 샌들우드와 시더우드의 풍부한 노트가 공간에 안정감을 더해줍니다. 가벼운 허브와 상쾌한 시트러스 노트가 은은하게 균형을 이루며 여유롭고 세련된 분위기를 연출합니다.

#### Example 2 (when no brand is mentioned):
- User Input: 우디한 베이스를 가진 디퓨저를 추천해줘.
- Image Caption: The image shows a modern living room with a large window on the right side. The room has white walls and wooden flooring. On the left side of the room, there is a gray sofa and a white coffee table with a black and white patterned rug in front of it. In the center of the image, there are six black chairs arranged around a wooden dining table. The table is set with a vase and other decorative objects on it. Above the table, two large windows let in natural light and provide a view of the city outside. A white floor lamp is placed on the floor next to the sofa.
- Response:
  - Brand: None
  - Scent Description: 우디한 느낌이 강조된 세련되고 따뜻한 향. 샌들우드, 시더우드, 오크모스의 풍부하고 자연적인 노트가 섬세하게 어우러져 차분하고 안정된 분위기를 만들어냅니다. 신선한 소나무와 가벼운 유칼립투스의 향이 상쾌함을 더해, 깊고 고요한 우디 향과 균형을 이루며 공간에 생기를 불어넣습니다. 이 향은 자연의 우아함을 그대로 담아내며, 평온하고 초대하는 느낌의 분위기를 완성합니다.

#### Example 3 (when a brand is mentioned but not in the list of existing brands):
- User Input: 샤넬 브랜드 제품의 우디한 베이스를 가진 디퓨저를 추천해줘.
- Image Caption: The image shows a modern living room with a large window on the right side. The room has white walls and wooden flooring. On the left side of the room, there is a gray sofa and a white coffee table with a black and white patterned rug in front of it. In the center of the image, there are six black chairs arranged around a wooden dining table. The table is set with a vase and other decorative objects on it. Above the table, two large windows let in natural light and provide a view of the city outside. A white floor lamp is placed on the floor next to the sofa.
- Response:
  - Brand: Not Found
  - Scent Description: 우디한 느낌이 강조된 따뜻하고 차분한 향. 샌들우드와 시더우드의 풍부한 노트가 공간에 안정감을 더해줍니다. 가벼운 허브와 상쾌한 시트러스 노트가 은은하게 균형을 이루며 여유롭고 세련된 분위기를 연출합니다.

- **User Input**: {user_input}
- **Image Caption**: {caption}
Response:"""
    
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_HOST")

    gpt_client = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=api_key,
        openai_api_base=api_base
    )
    fragrance_description = gpt_client.invoke(prompt).content.strip()

    return fragrance_description

if __name__ == "__main__":
    perfume_data_path = "product.json"
    diffuser_data_path = "./cache/diffuser_scent.json"

    perfume_data = load_perfume_data(perfume_data_path)
    diffuser_scent_data = load_diffuser_scent_data(diffuser_data_path)

    brands = get_distinct_brands(perfume_data)

    if not perfume_data:
        logger.info(f"No perfume data available.")
        exit()

    collection = initialize_vector_db(perfume_data, diffuser_scent_data)

    user_input = "우디한 향을 가진 디퓨저를 추천해주세요."
    caption = "The image shows a modern living room with a large window on the right side. The room has white walls and wooden flooring. On the left side of the room, there is a gray sofa and a white coffee table with a black and white patterned rug in front of it. In the center of the image, there are six black chairs arranged around a wooden dining table. The table is set with a vase and other decorative objects on it. Above the table, two large windows let in natural light and provide a view of the city outside. A white floor lamp is placed on the floor next to the sofa."
    
    # query_text 업데이트 => GPT에게 user input과 caption 전달 후 어울리는 향에 대한 설명 한국어로 반환(특정 브랜드 있으면 맨 앞에 적게끔 요청.)
    fragrance_description = get_fragrance_recommendation(user_input, caption, brands)
    logger.info(f"🎀 Generated Fragrance Description: {fragrance_description}")
    
    query_text = fragrance_description

    results = collection.query(
        query_texts=[query_text],
        n_results=5,
        # where={"brand": "딥티크"},  # Optional filter
    )

    logger.info(f"🎀 Query Results: {results}")