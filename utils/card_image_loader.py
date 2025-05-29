import cv2
import os

class CardImageLoader:
    """Класс для загрузки изображений карт"""
    def __init__(self, cards_dir='images/cards_pic'):
        self.card_images = self._load_card_images(cards_dir)
    
    def _load_card_images(self, cards_dir):
        """Загрузка изображений карт из директории"""
        card_images = {}
        for card_file in os.listdir(cards_dir):
            card_name = card_file.split('.')[0]
            card_images[card_name] = cv2.imread(os.path.join(cards_dir, card_file))
        return card_images
    
    def get_card_image(self, card_name):
        """Получение изображения карты по названию"""
        return self.card_images.get(card_name)