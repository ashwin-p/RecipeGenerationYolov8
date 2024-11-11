from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

def display_yolo_detections(image_path, model_path='best.pt'):
    """
    Loads a YOLO model, runs inference on an image, and displays the image with bounding boxes and class names.

    Parameters:
    - image_path (str): The path to the input image.
    - model_path (str): The path to the YOLO weights file (e.g., 'best.pt').
    """
    # Load the YOLO model with the specified weights
    model = YOLO(model_path)

    # Run inference on the image
    results = model(image_path)

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB for Matplotlib

    # Loop through each detection and draw bounding boxes and class names
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box coordinates
        class_name = model.names[int(box.cls)]          # Class name
        confidence = box.conf[0]                        # Confidence score

        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # Put the class name and confidence on the image
        label = f"{class_name} ({confidence:.2f})"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Display the image with bounding boxes and labels
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.imshow(image)
    plt.show()

# Example usage
image_path = "fruit1.jpg"
model_path = "./runs/detect/train/weights/best.pt"
display_yolo_detections(image_path, model_path)

