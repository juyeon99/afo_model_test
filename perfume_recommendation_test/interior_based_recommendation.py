import os
import random
import json
import re
import torch
import hashlib
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI

# 환경 변수 로드
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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

def analyze_image(image_path):
    """이미지 분석 및 캡션 생성"""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-large",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

    try:
        image = Image.open(image_path).convert('RGB')
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
        outputs = model.generate(**inputs, max_length=150, num_beams=4)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return ""

def recommend_perfumes(caption, user_message, perfume_data, max_perfumes=5):
    """향수 추천 및 응답 구조화"""
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-4o-mini",
        max_tokens=1000,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # TODO: 새로운 필터링 기준 적용 필요
    # perfume_list = perfume_data   # 전체 리스트를 프롬프트에 집어넣음 => EXPENSIVE! But does recommend the perfumes from certain brand if user requests.
    sampled_perfumes = random.sample(perfume_data, min(len(perfume_data), max_perfumes))    # 토큰 줄일 목적으로 랜덤으로 5개 넣기 => 필터링 기준 적용 필요
    perfume_list = "\n".join([f"- {p['id']} {p['name_kr']} ({p['brand']})" for p in sampled_perfumes])

    prompt_template = PromptTemplate(
        input_variables=["caption", "user_message", "perfume_list"],
        template=(
            "Based on the description of the interior design and the user's message, recommend perfumes.\n"
            "Please describe the common notes shared by the recommended perfumes and the emotions or atmosphere they evoke in detail.\n\n"
            "Please respond IN KOREAN in the following format:\n\n"
            "Common Feelings: [The common notes and characteristics of all the recommended perfumes, along with the atmosphere and emotions they evoke IN KOREAN. Example: '이 향수들은 시트러스 계열의 상쾌한 향과 따뜻한 우디 노트가 어우러져 세련된 느낌을 줍니다.']\n\n"
            "추천:\n"
            "1. [Perfume Name IN KOREAN]: [Detailed reason for the recommendation and situations where it fits IN KOREAN]\n"
            "2. [Perfume Name IN KOREAN]: [Detailed reason for the recommendation and situations where it fits IN KOREAN]\n"
            "3. [Perfume Name IN KOREAN]: [Detailed reason for the recommendation and situations where it fits IN KOREAN]\n\n"
            "설명:\n{caption}\n\n"
            "유저 메세지:\n{user_message}\n\n"
            "사용 가능한 향수:\n{perfume_list}"
        )
    )

    try:
        # Generate prompt and get response from OpenAI
        formatted_prompt = prompt_template.format(caption=caption, user_message=user_message, perfume_list=perfume_list)
        response = llm.invoke(formatted_prompt)
        common_feeling = ""
        recommendations = []

        def clean_name(name):
            return re.sub(r"^\d+\s+|\s*\([^)]*\)|:$|^\d+\.\s", "", name).strip()

        for line in response.content.split("\n"):
            if line.startswith("공통 느낌:"):
                common_feeling = line.replace("공통 느낌:", "").strip()
            elif line.startswith(("1.", "2.", "3.")):
                parts = line.split(": ", 1)
                if len(parts) == 2:
                    raw_name = parts[0].split(".", 1)[1].strip()
                    reason = parts[1].strip()
                    
                    matching_perfume = next(
                        (p for p in perfume_list if clean_name(p["name_kr"]) in clean_name(raw_name)),
                        None
                    )
                    
                    if matching_perfume:
                        recommendations.append({
                            "id": matching_perfume["id"],
                            "name_kr": matching_perfume["name_kr"],
                            "name_en": matching_perfume["name_en"],
                            "brand": matching_perfume["brand"],
                            "grade": matching_perfume["grade"],
                            "main_accord": matching_perfume["main_accord"],
                            "description": matching_perfume["content"],
                            "ingredients": matching_perfume["ingredients"],
                            "category_id": matching_perfume["category_id"],
                            "reason": reason
                        })

        return {
            "recommendations": recommendations,
            "content": common_feeling
        }

    except Exception as e:
        print(f"추천 생성 오류: {e}")
        return {
            "recommendations": [],
            "content": "추천을 생성할 수 없습니다."
        }

if __name__ == "__main__":
    # 데이터 로드
    perfume_data_path = "product.json"
    perfume_data = load_perfume_data(perfume_data_path)
    
    if not perfume_data:
        print("No perfume data available.")
        exit()

    # 이미지 분석
    image_path = "test_space.jpg"
    caption = analyze_image(image_path)
    # caption = "The image shows a modern living room with a large window on the right side. The room has white walls and wooden flooring. On the left side of the room, there is a gray sofa and a white coffee table with a black and white patterned rug in front of it. In the center of the image, there are six black chairs arranged around a wooden dining table. The table is set with a vase and other decorative objects on it. Above the table, two large windows let in natural light and provide a view of the city outside. A white floor lamp is placed on the floor next to the sofa."
    user_message = "크리드 브랜드의 향수를 추천해주세요."

    if caption:
        # print(f"\nImage Caption:\n{caption}")
        result = recommend_perfumes(caption, user_message, perfume_data, max_perfumes=5)
        print("\nResult:\n")
        print(json.dumps(result, ensure_ascii=False, indent=4))
    else:
        print("Failed to generate caption")