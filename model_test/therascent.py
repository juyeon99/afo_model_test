import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np

class TheraScent:
    def __init__(self):
        # Initialize the SBERT model
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Create scent therapy database
        self.scent_db = pd.DataFrame({
            'effect': [
                '휴식이 필요해요',
                '잠이 잘 안와요',
                '집중력이 필요해요',
                '에너지가 필요해요',
                '스트레스 해소가 필요해요'
            ],
            'scent': [
                '라벤더',
                '캐모마일',
                '페퍼민트',
                '레몬',
                '샌달우드'
            ],
            'description': [
                '진정 효과가 있는 달콤한 허브향으로 휴식과 수면에 도움을 줍니다.',
                '부드럽고 달콤한 허브향으로 수면의 질을 높여줍니다.',
                '시원하고 상쾌한 향으로 집중력과 기억력 향상에 도움을 줍니다.',
                '상큼한 시트러스향으로 기분 전환과 활력 증진에 효과적입니다.',
                '깊이 있는 우디향으로 마음의 안정과 스트레스 해소에 좋습니다.'
            ],
            'products': [
                {'디퓨저': '29,000원', '에센셜오일': '15,000원', '향수': '45,000원'},
                {'디퓨저': '27,000원', '에센셜오일': '14,000원', '향수': '42,000원'},
                {'디퓨저': '25,000원', '에센셜오일': '13,000원', '향수': '40,000원'},
                {'디퓨저': '26,000원', '에센셜오일': '14,000원', '향수': '41,000원'},
                {'디퓨저': '32,000원', '에센셜오일': '17,000원', '향수': '48,000원'}
            ]
        })
        
        # Pre-compute embeddings for all effects
        self.effect_embeddings = self.model.encode(self.scent_db['effect'].tolist())
        
    def get_recommendation(self, user_input: str) -> dict:
        """
        Get scent recommendation based on user input
        """
        # Encode user input
        input_embedding = self.model.encode([user_input])
        
        # Calculate similarities
        similarities = cosine_similarity(input_embedding, self.effect_embeddings)[0]
        
        # Get most similar effect index
        best_match_idx = np.argmax(similarities)
        
        # Get recommendation details
        recommendation = {
            'scent': self.scent_db.iloc[best_match_idx]['scent'],
            'description': self.scent_db.iloc[best_match_idx]['description'],
            'products': self.scent_db.iloc[best_match_idx]['products'],
            'similarity_score': similarities[best_match_idx]
        }
        
        return recommendation
    
    def format_response(self, recommendation: dict) -> str:
        """
        Format the recommendation into a natural response
        """
        products_text = '\n'.join([f"- {k}: {v}" for k, v in recommendation['products'].items()])
        
        response = f"""추천 향기: {recommendation['scent']}

{recommendation['description']}

제품 정보:
{products_text}"""
        
        return response

def chat_interface():
    """
    Simple chat interface for testing
    """
    bot = TheraScent()
    print("TheraScent 향기 테라피 봇입니다. 어떤 도움이 필요하신가요? (종료하려면 '끝'을 입력하세요)")
    
    while True:
        user_input = input("\n사용자: ")
        if user_input.lower() == '끝':
            break
            
        recommendation = bot.get_recommendation(user_input)
        response = bot.format_response(recommendation)
        print(f"\nTheraScent: {response}")

if __name__ == "__main__":
    chat_interface()