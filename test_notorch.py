from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

products = [
    {"id": 1, "name": "Wireless Noise-Canceling Headphones", "description": "Immerse yourself in pure audio with these comfortable and powerful headphones. Perfect for travel and focused work."},
    {"id": 2, "name": "Ergonomic Office Chair", "description": "Designed for all-day comfort and support, this chair helps improve posture and reduce back pain."},
    {"id": 3, "name": "Smart Coffee Maker with Voice Control", "description": "Start your day right with a freshly brewed cup of coffee, controlled by your voice or a mobile app."},
    {"id": 4, "name": "Lightweight Running Shoes", "description": "Engineered for speed and agility, these shoes provide excellent cushioning for your daily runs."},
    {"id": 5, "name": "Bamboo Cutting Board Set", "description": "A durable and eco-friendly set of cutting boards for all your food preparation needs."},
    {"id": 6, "name": "Portable Bluetooth Speaker", "description": "Enjoy your favorite music anywhere with this compact and water-resistant Bluetooth speaker."},
]

# Combine product name and description for better semantic coverage
product_texts = [f"{product['name']} {product['description']}" for product in products]

# Initialize and fit the TF-IDF vectorizer on the product texts
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(product_texts)

def search_products(query):
    # Transform the query into a TF-IDF vector
    query_vector = vectorizer.transform([query])
    # Calculate cosine similarity between query and all products
    similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # Prepare results with similarity scores
    results = []
    for index, product in enumerate(products):
        product_with_score = product.copy()
        product_with_score['similarity_score'] = round(similarity_scores[index], 2)
        results.append(product_with_score)
    
    # Sort products by similarity score in descending order
    results.sort(key=lambda x: x['similarity_score'], reverse=True)
    return results

# Example usage
if __name__ == "__main__":
    query = "always enjoy immersing your self in your own world undisturbed"
    search_results = search_products(query)
    # Display the top result
    print(search_results[0])