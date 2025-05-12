from ultralytics import YOLO
import cv2

def real_time_card_detection(model_path, video_path, output_video_path):
    # Загрузка модели
    model = YOLO(model_path)

    # Инициализация захвата видео из файла
    cap = cv2.VideoCapture(video_path)

    # Получение параметров видео для записи
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Инициализация VideoWriter для сохранения видео
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для MP4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        # Захват кадра
        ret, frame = cap.read()
        if not ret:
            break

        # Предсказание на кадре
        results = model(frame)

        # Визуализация результатов
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Запись кадра в выходной видеофайл
        out.write(frame)

        # Показать кадр
        cv2.imshow('Real-time Card Detection', frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    out.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    model_path = 'trained_model.pt'  # путь к вашей обученной модели
    video_path = 'video_2025-05-12_19-54-09.mp4'  # путь к вашему видеофайлу
    output_video_path = 'predictions/output_video.mp4'  # путь для сохранения обработанного видео
    real_time_card_detection(model_path, video_path, output_video_path)