# query_knowledge_base.py (High-Performance GGUF Version)
import cv2
import chromadb
from PIL import Image
from llama_cpp import Llama 

from create_embedding import get_image_embedding

print("Initializing components...")
client = chromadb.PersistentClient(path="db")
collection = client.get_collection(name="curious_curator_collection")

print("Loading GGUF LLM... This may take a moment.")
try:
    llm = Llama.from_pretrained(
	repo_id="google/gemma-2b-it-GGUF",
    n_ctx=4096,  # set context size to 4096 tokens
	filename="gemma-2b-it.gguf",
)
    print("GGUF LLM loaded successfully.")
except Exception as e:
    print(f"Error loading GGUF model: {e}")
    exit()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nWebcam opened. Show me an object you've taught me about.")
print("Press 's' to scan the object, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('Webcam - Press "s" to scan, "q" to quit', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('s'):
        print("\nScanning object...")
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        query_embedding = get_image_embedding(pil_image)

        if query_embedding:
            results = collection.query(query_embeddings=[query_embedding], n_results=1)

            if results and results['metadatas'] and results['metadatas'][0]:
                best_match = results['metadatas'][0][0]
                object_name = best_match['object_name']
                knowledge = best_match.get('knowledge', None)

                print(f"I think this is a: {object_name}")

                if not knowledge:
                    print("I recognize this object, but I don't have detailed knowledge about it yet.")
                else:
                    # --- THE CONVERSATIONAL LOOP ---
                    print(f"I have information about '{object_name}'. What would you like to know?")
                    user_question = input("Your question: ")

                    knowledge = knowledge[:3000] # Truncate for context

                    prompt = f"""<start_of_turn>user
You are an expert AI assistant. Your task is to use the provided 'CONTEXT' to answer the user's 'QUESTION'. Base your entire answer ONLY on the context provided. If the answer is not in the context, say "I do not have information about that in my knowledge base."

CONTEXT:
---
{knowledge}
---

QUESTION:
{user_question}
<end_of_turn>
<start_of_turn>model
"""
                    print("Asking the curator...")
                    output = llm(
                        prompt,
                        max_tokens=250,
                        stop=["<end_of_turn>"],
                        echo=False
                    )
                    
                    print("\n--- Curator's Response ---")
                    print(output['choices'][0]['text'].strip())
                    print("--------------------------\n")
            else:
                print("I don't recognize this object.")
        else:
            print("Could not create embedding.")
        
        print("Ready for next scan. Press 's' to scan again or 'q' to quit.")

cap.release()
cv2.destroyAllWindows()