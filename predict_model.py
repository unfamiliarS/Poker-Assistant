import os
import random
from ultralytics import YOLO
import cv2

def predict_and_visualize(model_path, test_images_dir, output_dir):
    model = YOLO(model_path)
    all_images = [os.path.join(test_images_dir, img) for img in os.listdir(test_images_dir) if img.endswith(('.jpg'))]
    test_images = random.sample(all_images, 4)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for img_path in test_images:

        results = model(img_path)

        img = cv2.imread(img_path)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, img)

        cv2.imshow('Prediction', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = 'trained_model.pt'
    test_images_dir = 'datasets/test/images'
    output_dir = 'predictions'
    predict_and_visualize(model_path, test_images_dir, output_dir)
