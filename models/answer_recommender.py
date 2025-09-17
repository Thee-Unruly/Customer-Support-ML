import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

def get_recommendation_system():
    """
    Builds and returns a function to recommend answers for new support tickets.
    """
    # Step 1: Data Preparation
    csv_file_path = "C:\\Users\\ibrahim.fadhili\\OneDrive - Agile Business Solutions\\Desktop\\Customer Support\\data\\processed\\processed_tickets.csv"
    if not os.path.exists(csv_file_path):
        print(f"Error: CSV file not found at '{csv_file_path}'")
        return None

    df = pd.read_csv(csv_file_path)
    
    # Create full_body column and handle missing values
    df["full_body"] = df["subject"].fillna("") + " " + df["body"].fillna("")
    
    # Remove any rows with empty full_body
    df = df[df["full_body"].str.strip().astype(bool)]
    
    # Check if we have any data left
    if len(df) == 0:
        print("Error: No valid text data found after cleaning")
        return None

    # Step 2: Generate Embeddings
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Generating embeddings for historical tickets...")
    
    # Convert to list and ensure all items are strings
    texts = df["full_body"].tolist()
    ticket_embeddings = model.encode(texts)

    # Step 3: Build the Retrieval System
    def get_recommended_answer(new_ticket_text):
        """
        Embeds a new ticket and finds the most similar historical answer.
        """
        # Embed the new ticket
        new_ticket_embedding = model.encode([new_ticket_text])

        # Calculate cosine similarity with all historical tickets
        similarities = cosine_similarity(new_ticket_embedding, ticket_embeddings)

        # Find the index of the most similar ticket
        most_similar_index = np.argmax(similarities)

        # Get the recommended answer and its similarity score
        recommended_ticket = df.iloc[most_similar_index]
        recommended_answer = recommended_ticket["answer"]
        similarity_score = similarities[0][most_similar_index]

        return recommended_answer, similarity_score, recommended_ticket

    return get_recommended_answer

# --- Main script execution ---
if __name__ == "__main__":
    recommender = get_recommendation_system()

    if recommender:
        # Define a new, incoming ticket
        new_ticket_body = "I am having trouble logging into my account. It says the server is down. Is there an outage?"

        # Get the recommendation
        recommended_answer, score, source_ticket = recommender(new_ticket_body)

        # Print the results
        print("\n" + "="*50)
        print("🔍 New Ticket Received:")
        print(new_ticket_body)
        print("="*50)
        print("🤖 Recommended Answer:")
        print(recommended_answer)
        print("="*50)
        print(f"Similarity Score (Cosine Similarity): {score:.4f}")
        print("Source of Recommendation (Most Similar Historical Ticket):")
        print(f"  Subject: {source_ticket['subject']}")
        print(f"  Body: {source_ticket['body'][:100]}...")
        print("="*50)