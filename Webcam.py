import cv2
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

class_names = ["Glass", "Metal", "Paper", "Plastic", "Cardboard", "Organic"]

def capture_and_classify(models, device, thresholds={"Paper": 0.7, "Cardboard": 0.7}):
    for model in models.values():
        model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = transform(image).unsqueeze(0).to(device)
            outputs = [model(image) for model in models.values()]
            avg_output = torch.mean(torch.stack(outputs), dim=0)
            probs = F.softmax(avg_output, dim=1)
            confidence, predicted = torch.max(probs, 1)

            label = class_names[predicted.item()]
            threshold = thresholds.get(label, 0.6)
            if confidence.item() > threshold:
                label_text = f"Predicted: {label} ({confidence.item():.2f})"
            else:
                label_text = "Not recognized as waste"

            print(label_text)
            cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Webcam - Solid Waste Classification", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Exiting webcam...")
                break

    cap.release()
    cv2.destroyAllWindows()