"""
RAG pipeline module for generating answers using retrieved products.
"""

import os
from openai import OpenAI


class RAGPipeline:
    """Handles RAG-based answer generation."""
    
    def __init__(self, retriever, api_key=None):
        self.retriever = retriever
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY must be set in environment or passed as argument")
        self.client = OpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"  # Using gpt-4o-mini as per notebook
    
    def format_rag_context(self, products, max_items=5):
        """Format retrieved products into context string."""
        rows = []
        for i, (_, row) in enumerate(products.head(max_items).iterrows(), start=1):
            name = row.get("Product Name", "")
            brand = row.get("Brand Name", "")
            category = row.get("Category", "")
            price = row.get("Selling Price", "")
            desc = row.get("product_text", "")
            
            snippet = desc[:600] + ("..." if len(desc) > 600 else "")
            
            rows.append(
                f"Product {i}:\n"
                f"  Name: {name}\n"
                f"  Brand: {brand}\n"
                f"  Category: {category}\n"
                f"  Price: {price}\n"
                f"  Details: {snippet}\n"
            )
        return "\n".join(rows)
    
    def build_rag_prompt(self, user_query, products):
        """Build RAG prompt with retrieved product context."""
        context_block = self.format_rag_context(products)
        
        prompt = f"""
You are a helpful e-commerce assistant working over a fixed product catalog.
You can ONLY use the retrieved product information below.

Instructions:
- First, check if any retrieved product clearly matches what the user is asking about.
- If none of the products seem to match well, say that the catalog does not contain
  that exact product and that you are only suggesting similar alternatives.
- Do NOT invent brand or product names that are not present in the retrieved context.
- When recommending products, mention their names and key features explicitly.
- Be concise, factual, and cautious when you are unsure.

Customer query:
{user_query}

Retrieved products:
{context_block}

Now answer the customer's question using ONLY the products above.
"""
        return prompt
    
    def generate_answer(self, prompt: str) -> str:
        """Generate answer using OpenAI API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def answer_text_query_rag(self, user_query, k=5):
        """
        Full RAG pipeline for text queries.
        
        Returns:
            tuple: (answer, retrieved_products)
        """
        retrieved = self.retriever.retrieve_by_text_query(user_query, k=k)
        prompt = self.build_rag_prompt(user_query, retrieved)
        answer = self.generate_answer(prompt)
        return answer, retrieved
    
    def answer_image_query_rag(self, image_path_or_pil, user_query=None, k=5):
        """
        Full RAG pipeline for image queries.
        
        Args:
            image_path_or_pil: Path to image or PIL Image object
            user_query: Optional text query to accompany the image
            k: Number of products to retrieve
            
        Returns:
            tuple: (answer, retrieved_products)
        """
        if user_query is None or user_query.strip() == "":
            user_query = "Identify and describe this product and how it is used."
        
        retrieved = self.retriever.retrieve_by_image_query(image_path_or_pil, k=k)
        prompt = self.build_rag_prompt(user_query, retrieved)
        answer = self.generate_answer(prompt)
        return answer, retrieved

