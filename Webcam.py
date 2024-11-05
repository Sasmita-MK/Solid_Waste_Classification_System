import cv2
import torch
from torchvision import transforms
from PIL import Image

# Define class names corresponding to your dataset labels
class_names = ["Glass", "Metal", "Paper", "Plastic", "Cardboard", "Organic", "Other"]  # Adjust this list as per your dataset

def capture_and_classify(model, device):
    # Set model to evaluation mode
    model.eval()

    # Define preprocessing for webcam input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with torch.no_grad():
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # Convert to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Preprocess the image
            image = transform(image).unsqueeze(0).to(device)

            # Classify the image
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

            # Get the class label from the predicted index
            label = class_names[predicted.item()]
            label_text = f"Predicted Class: {label}"
            print(label_text)

            # Display the frame with prediction
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam - Solid Waste Classification", frame)

            # Check for 'q' key press to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting webcam...")
                break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()