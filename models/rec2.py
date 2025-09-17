import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path

class TicketResponseRecommender:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize the recommendation system with Sentence-BERT model
        
        Args:
            model_name: Pre-trained Sentence-BERT model name
        """
        self.model = SentenceTransformer(model_name)
        self.historical_tickets = None
        self.ticket_embeddings = None
        self.answer_embeddings = None
        self.ticket_texts = None
        
    def load_data(self, file_path):
        """
        Load historical ticket data from CSV file
        
        Args:
            file_path: Path to CSV file containing ticket data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.historical_tickets = pd.read_csv(file_path)
        print(f"Loaded {len(self.historical_tickets)} historical tickets from {file_path}")
        
        # Display basic info about the data
        print(f"Columns available: {list(self.historical_tickets.columns)}")
        if not self.historical_tickets.empty:
            print(f"Sample data:\n{self.historical_tickets.head(2)}")
        
    def preprocess_data(self):
        """
        Preprocess the ticket data and generate embeddings
        """
        if self.historical_tickets is None:
            raise ValueError("Please load data first using load_data()")
        
        # Check which columns are available
        available_columns = set(self.historical_tickets.columns)
        
        # Create ticket texts based on available columns
        ticket_texts = []
        for _, row in self.historical_tickets.iterrows():
            parts = []
            
            if 'subject' in available_columns and pd.notna(row.get('subject')):
                parts.append(str(row['subject']))
            
            if 'body' in available_columns and pd.notna(row.get('body')):
                parts.append(str(row['body']))
            
            if 'description' in available_columns and pd.notna(row.get('description')):
                parts.append(str(row['description']))
            
            ticket_text = ". ".join(parts)
            ticket_texts.append(ticket_text)
        
        self.ticket_texts = ticket_texts
        
        # Generate embeddings for ticket texts
        print("Generating embeddings for historical tickets...")
        self.ticket_embeddings = self.model.encode(ticket_texts)
        
        # Generate embeddings for answers if available
        if 'answer' in available_columns:
            print("Generating embeddings for answers...")
            self.answer_embeddings = self.model.encode(self.historical_tickets['answer'].tolist())
        else:
            print("Warning: 'answer' column not found. Only ticket similarity will be available.")
            self.answer_embeddings = None
        
        print("Embeddings generated successfully!")
    
    def find_similar_ticket(self, new_ticket_text, top_k=3):
        """
        Find the most similar historical tickets to a new ticket
        
        Args:
            new_ticket_text: Text of the new ticket
            top_k: Number of similar tickets to return
            
        Returns:
            List of similar tickets with similarity scores
        """
        if self.ticket_embeddings is None:
            raise ValueError("Please preprocess data first using preprocess_data()")
        
        # Generate embedding for new ticket
        new_ticket_embedding = self.model.encode([new_ticket_text])
        
        # Calculate similarity scores
        similarities = cosine_similarity(new_ticket_embedding, self.ticket_embeddings)[0]
        
        # Get top-k similar tickets
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            result = {
                'similarity_score': float(similarities[idx]),
                'ticket_text': self.ticket_texts[idx][:200] + "..." if len(self.ticket_texts[idx]) > 200 else self.ticket_texts[idx]
            }
            
            # Add available columns to results
            available_columns = set(self.historical_tickets.columns)
            for col in available_columns:
                if col in self.historical_tickets.columns:
                    result[col] = self.historical_tickets.iloc[idx][col]
            
            results.append(result)
        
        return results
    
    def recommend_response(self, new_ticket_text, threshold=0.7):
        """
        Recommend a canned response for a new ticket
        
        Args:
            new_ticket_text: Text of the new ticket
            threshold: Minimum similarity threshold for recommendation
            
        Returns:
            Recommended response or None if no good match found
        """
        if self.answer_embeddings is None:
            raise ValueError("Answer embeddings not available. Cannot recommend responses.")
        
        similar_tickets = self.find_similar_ticket(new_ticket_text, top_k=1)
        
        if similar_tickets and similar_tickets[0]['similarity_score'] >= threshold:
            return similar_tickets[0].get('answer', 'No answer available')
        else:
            return None
    
    def save_embeddings(self, file_path):
        """
        Save embeddings to file for future use
        """
        if self.ticket_embeddings is None:
            raise ValueError("No embeddings to save")
        
        # Save additional metadata
        metadata = {
            'ticket_texts': self.ticket_texts,
            'columns': list(self.historical_tickets.columns) if self.historical_tickets is not None else []
        }
        
        np.savez(file_path, 
                ticket_embeddings=self.ticket_embeddings,
                answer_embeddings=self.answer_embeddings if self.answer_embeddings is not None else np.array([]),
                metadata=metadata)
        print(f"Embeddings saved to {file_path}")
    
    def load_embeddings(self, file_path, data_file_path=None):
        """
        Load precomputed embeddings
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Embeddings file not found: {file_path}")
        
        data = np.load(file_path, allow_pickle=True)
        self.ticket_embeddings = data['ticket_embeddings']
        
        # Handle answer embeddings (might be empty array)
        answer_embeds = data['answer_embeddings']
        self.answer_embeddings = answer_embeds if answer_embeds.size > 0 else None
        
        # Load metadata
        metadata = data['metadata'].item()
        self.ticket_texts = metadata.get('ticket_texts', [])
        
        # Load data if provided
        if data_file_path and os.path.exists(data_file_path):
            self.load_data(data_file_path)
        
        print("Embeddings loaded successfully!")

def test_with_sample_data():
    """Test the system with your actual data files"""
    
    # Define paths
    data_dir = Path("data/processed")
    train_data_path = data_dir / "processed_tickets.csv"
    test_data_path = data_dir / "test_tickets.csv"  # Assuming you have a test file
    
    # Initialize the recommender system
    recommender = TicketResponseRecommender()
    
    # Check if embeddings already exist
    embeddings_path = data_dir / "ticket_embeddings.npz"
    
    if embeddings_path.exists():
        print("Loading precomputed embeddings...")
        recommender.load_embeddings(embeddings_path, train_data_path)
    else:
        print("Loading training data and computing embeddings...")
        recommender.load_data(train_data_path)
        recommender.preprocess_data()
        recommender.save_embeddings(embeddings_path)
    
    # Test with some sample queries
    test_queries = [
        "I can't access my account. The login page is not working.",
        "How do I reset my password? I forgot my login credentials.",
        "The system is down and I cannot complete my work.",
        "I need help with integration and API access.",
        "My order hasn't arrived yet, where is my package?"
    ]
    
    print("\n" + "=" * 80)
    print("TICKET RESPONSE RECOMMENDATION SYSTEM - TEST RESULTS")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🧪 TEST {i}:")
        print(f"📝 Query: {query}")
        print("-" * 60)
        
        try:
            # Find similar tickets
            similar_tickets = recommender.find_similar_ticket(query, top_k=2)
            
            for j, similar_ticket in enumerate(similar_tickets, 1):
                print(f"\n📋 Similar Ticket {j} (Score: {similar_ticket['similarity_score']:.3f}):")
                if 'subject' in similar_ticket:
                    print(f"   Subject: {similar_ticket['subject']}")
                print(f"   Text: {similar_ticket['ticket_text']}")
                if 'answer' in similar_ticket:
                    print(f"   Answer: {similar_ticket['answer'][:100]}...")
            
            # Get recommended response if answers are available
            if recommender.answer_embeddings is not None:
                recommended_response = recommender.recommend_response(query)
                if recommended_response:
                    print(f"\n✅ RECOMMENDED RESPONSE:")
                    print(recommended_response)
                else:
                    print(f"\n❌ No suitable canned response found (similarity below threshold)")
            else:
                print(f"\nℹ️  Answer recommendations not available in this dataset")
                
        except Exception as e:
            print(f"❌ Error processing query: {e}")
        
        print("=" * 80)

def test_with_actual_test_file():
    """Test with your actual test data file"""
    
    data_dir = Path("data/processed")
    train_data_path = data_dir / "processed_tickets.csv"
    test_data_path = data_dir / "test_tickets.csv"
    
    if not test_data_path.exists():
        print(f"Test file not found: {test_data_path}")
        return
    
    # Initialize recommender
    recommender = TicketResponseRecommender()
    embeddings_path = data_dir / "ticket_embeddings.npz"
    
    if embeddings_path.exists():
        recommender.load_embeddings(embeddings_path, train_data_path)
    else:
        recommender.load_data(train_data_path)
        recommender.preprocess_data()
        recommender.save_embeddings(embeddings_path)
    
    # Load test data
    test_data = pd.read_csv(test_data_path)
    print(f"Loaded {len(test_data)} test tickets")
    
    print("\n" + "=" * 80)
    print("TESTING WITH ACTUAL TEST DATA")
    print("=" * 80)
    
    for idx, row in test_data.iterrows():
        # Create test query from available columns
        test_query_parts = []
        if 'subject' in row and pd.notna(row['subject']):
            test_query_parts.append(str(row['subject']))
        if 'body' in row and pd.notna(row['body']):
            test_query_parts.append(str(row['body']))
        if 'description' in row and pd.notna(row['description']):
            test_query_parts.append(str(row['description']))
        
        test_query = ". ".join(test_query_parts)
        
        if not test_query.strip():
            continue
            
        print(f"\n🧪 TEST TICKET {idx + 1}:")
        print(f"📝 Original: {test_query[:150]}...")
        print("-" * 60)
        
        try:
            similar_tickets = recommender.find_similar_ticket(test_query, top_k=1)
            
            if similar_tickets:
                best_match = similar_tickets[0]
                print(f"📊 Best match similarity: {best_match['similarity_score']:.3f}")
                
                if 'answer' in best_match and recommender.answer_embeddings is not None:
                    print(f"\n✅ RECOMMENDED RESPONSE:")
                    print(best_match['answer'])
                else:
                    print(f"\nℹ️  No answer available for recommendation")
            else:
                print("❌ No similar tickets found")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("=" * 80)

if __name__ == "__main__":
    # Test with sample queries
    print("Testing with sample queries...")
    test_with_sample_data()
    
    # Test with actual test file if available
    print("\n\nTesting with actual test file...")
    test_with_actual_test_file()