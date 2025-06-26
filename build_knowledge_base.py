import os
import chromadb
import requests
from bs4 import BeautifulSoup
from create_embedding import get_image_embedding

def get_knowledge_from_url(url):
    try:
        print(f"Attempting to fetch knowledge from: {url}")
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()
        
        text = soup.get_text(separator='\n', strip=True)
        if text:
            print(f"Successfully fetched {len(text)} characters of knowledge.")
            return text
        else:
            print("Warning: Fetched content but found no text.")
            return None
    except Exception as e:
        print(f"ERROR: Could not fetch or process URL. {e}")
        return None

print("Initializing knowledge base...")
client = chromadb.PersistentClient(path="db")
collection = client.get_or_create_collection(name="curious_curator_collection")
IMAGE_DIR = "object_images"

processed_objects_in_this_run = []

for root, _, files in os.walk(IMAGE_DIR):
    for filename in files:
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(root, filename)
            object_name = os.path.basename(root)

            if object_name not in processed_objects_in_this_run:
                
                existing_entry = collection.get(where={"object_name": object_name}, limit=1)
                
                if existing_entry['ids']:
                    print(f"Knowledge for '{object_name}' already exists in the database. Skipping knowledge gathering.")
                    processed_objects_in_this_run.append(object_name)
                else:
                    print(f"\n--- New Object Type Found: '{object_name}' ---")
                    url = input(f"Please provide a URL for a detailed description of '{object_name}' (e.g., a Wikipedia page): ")
                    
                    text_knowledge = get_knowledge_from_url(url)
                    
                    if text_knowledge:
                        collection.add(
                            embeddings=[[0.0] * 512], 
                            metadatas=[{"object_name": object_name, "knowledge": text_knowledge}],
                            ids=[f"knowledge_for_{object_name}"]
                        )
                        print(f"--- Knowledge for '{object_name}' has been ADDED to the database. ---")
                    else:
                        print(f"--- FAILED to add knowledge for '{object_name}'. ---")

                    processed_objects_in_this_run.append(object_name)

            if not collection.get(ids=[image_path])['ids']:
                print(f"Processing image: {image_path}")
                embedding = get_image_embedding(image_path)
                if embedding:
                    knowledge_entry = collection.get(ids=[f"knowledge_for_{object_name}"])
                    knowledge_text = knowledge_entry['metadatas'][0].get('knowledge', 'Default knowledge not found.')

                    collection.add(
                        embeddings=[embedding],
                        metadatas=[{"image_path": image_path, "object_name": object_name, "knowledge": knowledge_text}],
                        ids=[image_path]
                    )
                    print(f"Image embedding for '{filename}' added to knowledge base.")

print("\nKnowledge base build process complete!")