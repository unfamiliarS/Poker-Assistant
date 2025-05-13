from ultralytics import YOLO
import cv2

def real_time_card_detection(model_path):
    # Загрузка модели
    model = YOLO(model_path)

    # Инициализация захвата видео с веб-камеры
    cap = cv2.VideoCapture(0)  # 0 для встроенной камеры

    while cap.isOpened():
        # Захват кадра
        ret, frame = cap.read()
        if not ret:
            break

        # Предсказание на кадре
        results = model(frame)

        # Список для хранения обнаруженных карт
        detected_cards = set()

        # Визуализация результатов
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Добавление карты в список
                detected_cards.add(model.names[int(cls)])

        # Отображение списка карт в правом нижнем углу
        card_list_text = ", ".join(detected_cards)
        cv2.putText(frame, f"Cards in hand: {card_list_text}", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Показать кадр
        cv2.imshow('Real-time Card Detection', frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = 'trained_model.pt'  # путь к вашей обученной модели
    real_time_card_detection(model_path)
