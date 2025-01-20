import openai
from typing import List, Dict
import numpy as np
from datetime import datetime

class ReviewSummarizer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        openai.api_key = api_key
        self.summary_cache = {}
        
    def _chunk_reviews(self, reviews: List[str], chunk_size: int = 10) -> List[List[str]]:
        return [reviews[i:i + chunk_size] for i in range(0, len(reviews), chunk_size)]
    
    def summarize_chunk(self, reviews: List[str]) -> str:
        combined_reviews = "\n".join(reviews)
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Summarize the key points from these reviews concisely."},
                {"role": "user", "content": combined_reviews}
            ]
        )
        
        return response.choices[0].message['content']
    
    def update_summary(self, product_id: str, new_reviews: List[str]) -> str:
        existing_summary = self.summary_cache.get(product_id, "")
        
        if existing_summary:
            prompt = f"Previous summary:\n{existing_summary}\n\nNew reviews:\n{' '.join(new_reviews)}\nUpdate the summary incorporating these new reviews."
        else:
            prompt = "\n".join(new_reviews)
            
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Create a comprehensive summary of product reviews."},
                {"role": "user", "content": prompt}
            ]
        )
        
        new_summary = response.choices[0].message['content']
        self.summary_cache[product_id] = new_summary
        return new_summary
    
    def process_reviews(self, product_id: str, reviews: List[str], chunk_size: int = 10) -> str:
        # Split into manageable chunks
        chunks = self._chunk_reviews(reviews, chunk_size)
        
        # Get summaries for each chunk
        chunk_summaries = []
        for chunk in chunks:
            summary = self.summarize_chunk(chunk)
            chunk_summaries.append(summary)
            
        # Combine chunk summaries
        if len(chunk_summaries) > 1:
            final_summary = self.summarize_chunk(chunk_summaries)
        else:
            final_summary = chunk_summaries[0]
            
        self.summary_cache[product_id] = final_summary
        return final_summary

if __name__ == "__main__":
    summarizer = ReviewSummarizer("openai-api-key")
    
    # Initial batch of reviews
    reviews = [
        "Great product, very durable",
        "Decent quality but expensive",
        # ... more reviews ...
    ]
    
    # Process initial batch
    summary = summarizer.process_reviews("lelabo_thematcha26", reviews)
    print("Initial Summary:", summary)
    
    # New reviews come in
    new_reviews = [
        "Amazing customer service",
        "Product broke after 2 months",
    ]
    
    # Update summary with new reviews
    updated_summary = summarizer.update_summary("product123", new_reviews)
    print("Updated Summary:", updated_summary)