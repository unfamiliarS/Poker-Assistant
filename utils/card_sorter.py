class CardSorter:
    """Класс для сортировки карт"""
    SORT_ORDER = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
        '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12,
        'H': 0, 'D': 1, 'C': 2, 'S': 3
    }

    @classmethod
    def sort_cards(cls, cards):
        """Сортировка карт по рангу и масти"""
        return sorted(cards, key=lambda x: (
            cls.SORT_ORDER.get(x[:-1].upper(), 99),
            cls.SORT_ORDER.get(x[-1].upper(), 99)
        ))