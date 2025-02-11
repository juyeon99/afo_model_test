# 향료정보 가져와서 향의 느낌을 GPT에게 설명 요청 후 diffuser_scent.json 캐시에 저장
import json
import os
from collections import defaultdict
from dotenv import load_dotenv
import openai
from langchain_openai import ChatOpenAI

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

NOTE_TYPES = ["TOP", "MIDDLE", "BASE", "SINGLE"]

CACHE_DIR = "./cache"
NOTE_CACHE_PATH = os.path.join(CACHE_DIR, "note_cache.json")
SPICE_CACHE_PATH = os.path.join(CACHE_DIR, "spice_cache.json")
PRODUCT_CACHE_PATH = os.path.join(CACHE_DIR, "product_cache.json")
DIFFUSER_SCENT_PATH = os.path.join(CACHE_DIR, "diffuser_scent.json")

def get_product_details(product_id, products):
    for product in products:
        if product["id"] == product_id:
            return product
    return None

# Load cache data
def load_cache(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

notes = load_cache(NOTE_CACHE_PATH)
spices = load_cache(SPICE_CACHE_PATH)
products = load_cache(PRODUCT_CACHE_PATH)

# Extract product IDs from the product cache
existing_product_ids = {product["id"] for product in products}

# Create spice ID to name mapping
spice_id_to_name = {spice["id"]: spice["name_kr"] for spice in spices}

# Group notes by product_id
product_notes = defaultdict(lambda: defaultdict(list))

for note in notes:
    note_type = note["note_type"].upper()
    product_id = note["product_id"]
    if product_id in existing_product_ids:
        spice_name = spice_id_to_name.get(note["spice_id"], "")
        if note_type in NOTE_TYPES and spice_name:
            product_notes[product_id][note_type].append(spice_name)

# Generate formatted scent descriptions
def format_notes(note_data):
    if "SINGLE" in note_data:
        single_notes = ", ".join(note_data["SINGLE"])
        return f"single: {single_notes}"
    else:
        formatted = []
        for note_type in ["TOP", "MIDDLE", "BASE"]:
            if note_data.get(note_type):
                notes_str = ", ".join(note_data[note_type])
                formatted.append(f"{note_type.lower()}: {notes_str}")
        return "\n".join(formatted)

def generate_scent_description(notes_text, diffuser_description):
    prompt = f"""Based on the following fragrance combination of the diffuser, describe the characteristics of the overall scent using common perfumery terms such as 우디, 플로럴, 스파이시, 시트러스, 허브, 머스크, 아쿠아, 그린, 구르망, etc. You do not need to break down each note, just focus on the overall scent impression.
        # EXAMPLE 1:
        - Note: Top: 이탈리안 레몬 잎, 로즈마리\nMiddle: 자스민, 라반딘\nBase: 시더우드, 머스크
        - Diffuser Description: 당신의 여정에 감각적이고 신선한 향기가 퍼집니다. 아침 햇살이 창문을 통해 들어올 때, 산들 바람과 함께 이탈리아 시골을 연상시키는 푸른 향기
        - Response: 상쾌한 시트러스와 허브의 조화, 플로럴한 우아함, 따뜻한 우디한 향과 부드러운 머스크가 어우러져 균형 잡힌 향기를 만들어냅니다. 전체적으로 이 향은 활력을 주면서도 동시에 편안함과 안정감을 느낄 수 있는, 다채롭고 매력적인 향입니다.

        # EXAMPLE 2:
        - Note: single: 이탈리안 베르가못, 이탈리안 레몬, 자몽, 무화과, 핑크 페퍼, 자스민 꽃잎, 무화과 나무, 시더우드, 벤조인
        - Diffuser Description: 당신의 여정에 감각적이고 신선한 향기가 퍼집니다. 아침 햇살이 창문을 통해 들어올 때, 산들 바람과 함께 이탈리아 시골을 연상시키는 푸른 향기
        - Response: 이 향은 상쾌하고 활기찬 느낌을 주면서도, 부드럽고 따뜻한 깊이를 지닌 균형 잡힌 향입니다. 밝고 톡톡 튀는 시트러스 향이 기분을 상쾌하게 해주고, 달콤하고 우아한 플로럴과 자연적인 우디한 느낌이 조화를 이루며 세련된 분위기를 만들어냅니다. 전체적으로 신선하고 세련되며, 따뜻하면서도 편안한 느낌을 주는 복합적인 향입니다.

        # Note: {notes_text}
        # Diffuser Description: {diffuser_description}
        # Response: """
    
    print("💟",notes_text)
    
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("OPENAI_HOST")

    gpt_client = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        openai_api_key=api_key,
        openai_api_base=api_base
    )
    response = gpt_client.invoke(prompt).content.strip()

    return response

# Load or initialize the scent cache as a dictionary
def load_diffuser_scent_cache():
    if os.path.exists(DIFFUSER_SCENT_PATH):
        with open(DIFFUSER_SCENT_PATH, "r", encoding="utf-8") as f:
            scent_cache = json.load(f)
            # Convert the list to a dictionary for easy look-up
            return {str(item["id"]): item["scent_description"] for item in scent_cache}
    return {}

# Update scent cache to a list before saving
def save_scent_cache(scent_cache):
    scent_cache_list = [{"id": int(product_id), "scent_description": scent_description} 
                        for product_id, scent_description in scent_cache.items()]
    with open(DIFFUSER_SCENT_PATH, "w", encoding="utf-8") as f:
        json.dump(scent_cache_list, f, ensure_ascii=False, indent=4)

# Load the scent cache as a dictionary
scent_cache = load_diffuser_scent_cache()

# Generate and cache scent descriptions
scent_cache_list = []

for product_id, note_data in product_notes.items():
    if str(product_id) in scent_cache:
        print(f"Product {product_id} already has a cached scent description.")
        scent_cache_list.append({
            "id": int(product_id),
            "scent_description": scent_cache[str(product_id)]
        })
        continue

    formatted_notes = format_notes(note_data)
    print(f"Generating scent description for product {product_id}...")

    product_details = get_product_details(product_id, products)
    if product_details:
        # Diffuser description is fetched from product details or assigned manually
        diffuser_description = product_details.get("content", "")

    scent_description = generate_scent_description(formatted_notes, diffuser_description)
    scent_cache[str(product_id)] = scent_description

    print(f"Scent description for product {product_id}: {scent_description}")

# Save the updated scent cache as a list
save_scent_cache(scent_cache)

print("All scent descriptions have been updated and saved.")