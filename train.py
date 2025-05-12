from ultralytics import YOLO
import multiprocessing

def main():
    # 1. Загрузка модели (выберите подходящий размер)
    model = YOLO('best.pt') 
    # model = YOLO('yolov8n.pt')  # nano (самая быстрая, меньше точность)
    # model = YOLO('yolov8s.pt')  # small
    # model = YOLO('yolov8m.pt')  # medium
    # model = YOLO('yolov8l.pt')  # large
    # model = YOLO('yolov8x.pt')  # xlarge (самая точная, но медленная)

    # 2. Обучение модели
    results = model.train(
        resume=True,
        # data='datasets/data.yaml',  # путь к вашему конфигурационному файлу
        # epochs=100,        # количество эпох
        # imgsz=640,         # размер изображения
        # batch=8,           # размер батча (уменьшите, если не хватает памяти)
        # patience=10,       # ранняя остановка если нет улучшений
        # device='0',        # '0' для GPU, 'cpu' для CPU
        # name='playing_cards_v1',   # имя эксперимента
        # pretrained=True,   # использовать предобученные веса
        # optimizer='auto',  # автоматический выбор оптимизатора
        # lr0=0.01,          # начальная скорость обучения
        # augment=True       # аугментация данных
    )

    # 3. Валидация модели
    metrics = model.val()  # оценка на валидационном наборе
    print(metrics.box.map)  # mAP50-95

    # 4. Экспорт модели (опционально)
    model.export(format='onnx')

if __name__ == '__main__':
    # Фикс для multiprocessing в Windows/Jupyter
    multiprocessing.set_start_method('spawn', force=True)
    main()