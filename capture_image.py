import cv2
import os

IMAGE_DIR = "object_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\nPoint the camera at an object you want to teach the AI.")
print("Press the 's' key to save a picture. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame. Exiting ...")
        break

    cv2.imshow('Webcam Feed - Press "s" to save, "q" to quit', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        object_name = input("Enter a simple name for this object (e.g., 'blue_mug', 'watch'): ")
        if object_name:
            object_path = os.path.join(IMAGE_DIR, object_name)
            os.makedirs(object_path, exist_ok=True)

            count = len(os.listdir(object_path))
            image_name = f"{object_name}_{count + 1}.jpg"
            image_path = os.path.join(object_path, image_name)

            cv2.imwrite(image_path, frame)
            print(f"Image saved as: {image_path}")
        else:
            print("Invalid name. Image not saved.")

    elif key == ord('q'):
        print("Quitting.")
        break

cap.release()
cv2.destroyAllWindows()
