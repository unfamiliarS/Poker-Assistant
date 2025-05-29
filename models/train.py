from ultralytics import YOLO
import multiprocessing

def main():
    model = YOLO('best.pt') 
    model = YOLO('yolov8n.pt')

    # 2. Обучение модели
    results = model.train(
        data='datasets/data.yaml',  # путь к вашему конфигурационному файлу
        epochs=100,        # количество эпох
        imgsz=640,         # размер изображения
        batch=8,           # размер батча (уменьшите, если не хватает памяти)
        patience=10,       # ранняя остановка если нет улучшений
        device='0',        # '0' для GPU, 'cpu' для CPU
        name='playing_cards_v1',   # имя эксперимента
        pretrained=True,   # использовать предобученные веса
        optimizer='auto',  # автоматический выбор оптимизатора
        lr0=0.01,          # начальная скорость обучения
        augment=True       # аугментация данных
    )

    metrics = model.val()
    print(metrics.box.map)

    model.export(format='onnx')

if __name__ == '__main__':
    # Фикс для multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    main()