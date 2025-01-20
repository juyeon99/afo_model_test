from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
model = AutoModel.from_pretrained('bert-base-multilingual-cased')

def extract_tags(review):
    tag_categories = {
        '향특성': ['플로럴', '머스크', '달콤한', '시트러스'],
        '성능': ['지속력', '잔향', '투사력'],
        '계절': ['봄', '여름', '가을', '겨울']
    }
    
    inputs = tokenizer(review, return_tensors='pt', truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        
    attention_mask = inputs['attention_mask']
    mean_embedding = (embeddings * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(1).unsqueeze(-1)
    
    extracted_tags = []
    for category, tags in tag_categories.items():
        for tag in tags:
            tag_inputs = tokenizer(tag, return_tensors='pt')
            with torch.no_grad():
                tag_outputs = model(**tag_inputs)
                tag_embedding = tag_outputs.last_hidden_state.mean(1)
            
            similarity = F.cosine_similarity(mean_embedding, tag_embedding)
            if similarity > 0.3:
                extracted_tags.append(tag)
    
    return extracted_tags

sample_review = """이 향수는 첫 스프레이에서 강렬한 재스민과 로즈 향이 느껴집니다. 
지속력이 6시간 정도로 양호하고, 잔향이 특히 매력적입니다. 플로럴하면서도 
머스크가 은은하게 깔려있어 중성적인 매력이 있네요. 시간이 지날수록 달콤한 
바닐라 노트가 드러나는데, 투사력이 조금 아쉬워요. 봄, 가을에 데일리로 사용하기 좋은 향수입니다."""

tags = extract_tags(sample_review)
print("추출된 태그:", tags)