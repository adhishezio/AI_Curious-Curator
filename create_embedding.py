# create_embedding.py
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
print("Model loaded successfully.")


def get_image_embedding(image_input):
    """
    Takes either a file path (string) or a PIL Image object
    and returns its vector embedding using CLIP.
    """
    try:
        if isinstance(image_input, str):
            image = Image.open(image_input)
        else:
            image = image_input

        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_features = model.get_image_features(**inputs)
        return image_features.detach().numpy().flatten().tolist()
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

if __name__ == '__main__':
    test_image_path = 'object_images/anime figure/anime figure_1.jpg'

    print(f"\nAttempting to create embedding for: {test_image_path}")

    try:
        embedding = get_image_embedding(test_image_path)
        if embedding:
            print(f"Success! Created an embedding of size: {len(embedding)}")
            print("This list of numbers is the AI's 'fingerprint' for your image.")
        else:
            print("Failed to create an embedding.")
    except FileNotFoundError:
        print(f"Error: The file '{test_image_path}' was not found.")
        print("Please make sure you have captured an image and updated the 'test_image_path' variable in the script.")