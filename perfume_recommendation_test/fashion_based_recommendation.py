import os
import random
import json
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_perfume_data(json_path):
    """perfume.json 데이터를 로드"""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
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

def recommend_perfumes(caption, perfume_data, max_perfumes=5):
    """향수 추천 및 응답 구조화"""
    llm = OpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo-instruct",
        max_tokens=1000,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    sampled_perfumes = random.sample(perfume_data, min(len(perfume_data), max_perfumes))
    perfume_list = "\n".join([f"- {p['id']} {p['name']} ({p['brand']})" for p in sampled_perfumes])

    prompt_template = PromptTemplate(
        input_variables=["caption", "perfume_list"],
        template=(
            "이미지 설명을 기반으로 향수를 추천해주세요.\n"
            "추천된 향수들이 공통적으로 어떤 향조를 가지고 있는지, "
            "그리고 이 향들이 어떤 감정이나 분위기를 불러일으키는지 구체적으로 묘사해주세요.\n\n"
            "아래 형식으로 한글로 답변해주세요:\n\n"
            "공통 느낌: [추천된 모든 향수의 공통적인 향조와 특징, 그리고 주는 분위기와 감정. 예: '이 향수들은 시트러스 계열의 상쾌한 향과 따뜻한 우디 노트가 어우러져 세련된 느낌을 줍니다.']\n\n"
            "추천:\n"
            "추천:\n"
            "1. [향수명]: [구체적인 추천 이유와 어울리는 상황]\n"
            "2. [향수명]: [구체적인 추천 이유와 어울리는 상황]\n"
            "3. [향수명]: [구체적인 추천 이유와 어울리는 상황]\n\n"
            "설명:\n{caption}\n\n"
            "사용 가능한 향수:\n{perfume_list}"
        )
    )

    try:
        # Generate prompt and get response from OpenAI
        formatted_prompt = prompt_template.format(caption=caption, perfume_list=perfume_list)
        response = llm.invoke(formatted_prompt)
        common_feeling = ""
        recommendations = []

        def clean_name(name):
            return re.sub(r"^\d+\s+|\s*\([^)]*\)|:$|^\d+\.\s", "", name).strip()

        for line in response.split("\n"):
            if line.startswith("공통 느낌:"):
                common_feeling = line.replace("공통 느낌:", "").strip()
            elif line.startswith(("1.", "2.", "3.")):
                parts = line.split(": ", 1)
                if len(parts) == 2:
                    raw_name = parts[0].split(".", 1)[1].strip()
                    reason = parts[1].strip()
                    
                    matching_perfume = next(
                        (p for p in sampled_perfumes if clean_name(p["name"]) in clean_name(raw_name)),
                        None
                    )
                    
                    if matching_perfume:
                        recommendations.append({
                            "id": matching_perfume["id"],
                            "name": matching_perfume["name"],
                            "brand": matching_perfume["brand"],
                            "grade": matching_perfume["grade"],
                            "description": matching_perfume["description"],
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
    perfume_data_path = "perfume.json"
    perfume_data = load_perfume_data(perfume_data_path)
    
    if not perfume_data:
        print("No perfume data available.")
        exit()

    # 이미지 분석
    image_path = "test_fs.jpg"
    caption = analyze_image(image_path)
    if caption:
        print(f"\nImage Caption:\n{caption}")
        result = recommend_perfumes(caption, perfume_data, max_perfumes=5)
        print("\nResult:\n")
        print(json.dumps(result, ensure_ascii=False, indent=4))
    else:
        print("Failed to generate caption")
