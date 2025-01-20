# 향료 데이터 한국어로 번역
import os
import json
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_file_name(index):
    if 1 <= index <= 400:
        return 'spices_1_400.json'
    elif 801 <= index <= 1200:
        return 'spices_801_1200.json'
    elif index > 1200:
        return 'spices_1200_end.json'
    else:
        return 'spices_rest.json'

def save_json(spices, index):
    filename = get_file_name(index)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(spices, f, ensure_ascii=False, indent=2)

async def translate_spices():
    # Read JSON file
    with open('./spices.json', 'r', encoding='utf-8') as f:
        spices = json.load(f)
    
    # Create separate lists for each range
    spices_1_400 = spices[:400]
    spices_801_1200 = spices[800:1200]
    spices_rest = spices[400:800] + spices[1200:]
    
    # Save initial split files
    save_json(spices_1_400, 1)
    save_json(spices_801_1200, 801)
    save_json(spices_rest, 401)
    
    total = len(spices)
    for index, spy in enumerate(spices, 1):
        try:
            updated = False
            print(f"Processing {index}/{total}: {spy['name_en']}")

            # Translate name if empty
            if not spy['name_kr']:
                name_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{
                        "role": "system",
                        "content": "You are a Korean translator. Translate the given English fragrance name to Korean naturally."
                    }, {
                        "role": "user",
                        "content": f"Translate: {spy['name_en']}"
                    }]
                )
                spy['name_kr'] = name_response.choices[0].message.content.strip()
                updated = True
            
            # Generate content_en if empty
            if not spy['content_en']:
                desc_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{
                        "role": "system",
                        "content": "You are a fragrance expert. Provide a brief description of this fragrance note."
                    }, {
                        "role": "user",
                        "content": f"Describe the fragrance note: {spy['name_en']}"
                    }]
                )
                spy['content_en'] = desc_response.choices[0].message.content.strip()
                updated = True

            # Translate content if empty
            if not spy['content_kr'] and spy['content_en']:
                content_response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[{
                        "role": "system",
                        "content": "You are a Korean translator. Translate the given English fragrance description to Korean naturally."
                    }, {
                        "role": "user",
                        "content": f"Translate: {spy['content_en']}"
                    }]
                )
                spy['content_kr'] = content_response.choices[0].message.content.strip()
                updated = True

            if updated:
                # Update the appropriate file based on index
                if 1 <= index <= 400:
                    spices_1_400[index-1] = spy
                    save_json(spices_1_400, index)
                elif 801 <= index <= 1200:
                    spices_801_1200[index-801] = spy
                    save_json(spices_801_1200, index)
                else:
                    if index <= 800:
                        spices_rest[index-401] = spy
                    else:
                        spices_rest[index-1201+400] = spy
                    save_json(spices_rest, index)

        except Exception as e:
            print(f"Error processing {spy['name_en']}: {e}")

if __name__ == "__main__":
    asyncio.run(translate_spices())

# 아래 방법은 성능 不好
# import json
# from googletrans import Translator

# translator = Translator()

# input_path = "spices.json"
# with open(input_path, 'r', encoding='utf-8') as file:
#     spice_data = json.load(file)

# for spice in spice_data:
#     if spice.get("content"):
#         translated_content = translator.translate(spice["content"], src="en", dest="ko").text
#         spice["content_kr"] = translated_content
#     else:
#         spice["content_kr"] = ""

#     if spice.get("name"):
#         translated_name = translator.translate(spice["name"], src="en", dest="ko").text
#         spice["name_kr"] = translated_name
#     else:
#         spice["name_kr"] = ""

# output_path = "spices_trans.json"
# with open(output_path, 'w', encoding='utf-8') as output_file:
#     json.dump(spice_data, output_file, ensure_ascii=False, indent=4)

# print(f"Translation completed. Saved to {output_path}")