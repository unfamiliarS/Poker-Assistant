from ultralytics import YOLO
import cv2
from utils.card_sorter import CardSorter
from utils.poker_hand_evaluator import PokerHandEvaluator
from utils.card_history_manager import CardHistoryManager
from utils.card_image_loader import CardImageLoader

class PokerAssistant:
    """Основной класс покерного ассистента"""
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.card_history = CardHistoryManager()
        self.card_loader = CardImageLoader()
        self.video_width = 960
        self.video_height = 720
    
    def run(self):
        """Запуск основного цикла обработки видео"""
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = self._process_frame(frame)
            cv2.imshow('Poker Assistant', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    
    def _process_frame(self, frame):
        """Обработка одного кадра видео"""
        frame = cv2.resize(frame, (self.video_width, self.video_height))
        
        current_cards = self._detect_cards(frame)
        
        self.card_history.update_stats(current_cards)
        
        stable_cards = self.card_history.get_most_common_cards(5)
        stable_cards_sorted = CardSorter.sort_cards(stable_cards)
        
        if len(stable_cards_sorted) >= 5:
            self._display_combination_info(frame, stable_cards_sorted)
        
        self._display_card_thumbs(frame, stable_cards_sorted)
        
        return frame
    
    def _detect_cards(self, frame):
        """Детекция карт на кадре"""
        current_cards = []
        results = self.model(frame)
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cls = box.cls[0]
                card_name = self.model.names[int(cls)]
                label = f"{card_name} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                current_cards.append(card_name)
        
        return current_cards
    
    def _display_combination_info(self, frame, cards):
        """Отображение информации о комбинации"""
        combination, rarity = PokerHandEvaluator.evaluate_hand(cards[:5])
        
        # Вывод информации
        cv2.putText(frame, f"Combination: {combination}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f"Rarity: {rarity}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    def _display_card_thumbs(self, frame, cards):
        """Отображение миниатюр карт"""
        for i, card in enumerate(cards[:5]):
            card_img = self.card_loader.get_card_image(card)
            if card_img is not None:
                card_img = cv2.resize(card_img, (100, 140))
                x_offset = 10 + i * 110
                y_offset = frame.shape[0] - 160
                frame[y_offset:y_offset+140, x_offset:x_offset+100] = card_img