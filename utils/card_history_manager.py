from collections import Counter, deque
import time

class CardHistoryManager:
    """Класс для управления историей карт"""
    def __init__(self, maxlen=50):
        self.card_stats = Counter()
        self.last_detection_time = time.time()
        self.card_history = deque(maxlen=maxlen)
    
    def update_stats(self, cards):
        """Обновление статистики карт"""
        current_time = time.time()
        
        if current_time - self.last_detection_time > 1.0:
            self.card_stats.clear()
            self.card_history.clear()

        for card in cards:
            self.card_history.append(card)
        self.last_detection_time = current_time

        self.card_stats = Counter(self.card_history)
    
    def get_most_common_cards(self, n=5):
        """Получение n самых частых карт"""
        return [card for card, _ in self.card_stats.most_common(n)]