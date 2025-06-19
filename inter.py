from sentence_transformers import SentenceTransformer
from fastapi import FastAPI
import numpy as np

app = FastAPI()
model = SentenceTransformer('all-MiniLM-L6-v2')

products =[
{"id": 1, "name": "Wireless Noise-Canceling Headphones", "description": "Immerse yourself in pure audio with these comfortable and powerful headphones. Perfect for travel and focused work."},
{"id": 2, "name": "Ergonomic Office Chair", "description": "Designed for all-day comfort and support, this chair helps improve posture and reduce back pain."},
{"id": 3, "name": "Smart Coffee Maker with Voice Control", "description": "Start your day right with a freshly brewed cup of coffee, controlled by your voice or a mobile app."},
{"id": 4, "name": "Lightweight Running Shoes", "description": "Engineered for speed and agility, these shoes provide excellent cushioning for your daily runs."},

{"id": 5, "name": "Bamboo Cutting Board Set", "description": "A durable and eco-friendly set of cutting boards for all your food preparation needs."},
{"id": 6, "name": "Portable Bluetooth Speaker", "description": "Enjoy your favorite music anywhere with this compact and water-resistant Bluetooth speaker."},
]  # Your product list from the question

# Precompute product embeddings
product_descriptions = [p["description"] for p in products]
product_embeddings = model.encode(product_descriptions)


@app.post("/search")
async def semantic_search(query: str):
    # Generate query embedding
    query_embedding = model.encode([query])[0]
    
    # Calculate cosine similarity
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    product_norms = product_embeddings / np.linalg.norm(product_embeddings, axis=1, keepdims=True)
    similarities = np.dot(product_norms, query_norm)
    
    # Format results
    results = []
    for i, product in enumerate(products):
        result = {**product, "similarity_score": float(similarities[i])}
        results.append(result)
    
    # Sort by score descending
    return sorted(results, key=lambda x: x["similarity_score"], reverse=True)
