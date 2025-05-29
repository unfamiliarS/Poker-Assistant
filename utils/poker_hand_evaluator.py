from collections import Counter

class PokerHandEvaluator:
    """Класс для оценки покерных комбинаций"""
    RANK_ORDER = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
        '9': 9, '10': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    
    @classmethod
    def evaluate_hand(cls, cards):
        """Оценка комбинации карт"""
        ranks, suits = cls._extract_ranks_and_suits(cards)
        
        if len(ranks) < 5:
            return "Waiting...", 0.0

        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        is_flush = len(suit_counts) == 1
        is_straight = cls._check_straight(ranks, rank_counts)
        is_straight_flush = is_flush and is_straight

        return cls._determine_combination(
            is_straight_flush, ranks, rank_counts, is_flush, is_straight
        )
    
    @classmethod
    def _extract_ranks_and_suits(cls, cards):
        """Извлечение рангов и мастей из карт"""
        ranks = []
        suits = []
        
        for card in cards:
            if len(card) >= 2:
                rank = card[:-1].upper()
                suit = card[-1].upper()
                if rank in cls.RANK_ORDER and suit in ['H', 'D', 'C', 'S']:
                    ranks.append(cls.RANK_ORDER[rank])
                    suits.append(suit)
        return ranks, suits
    
    @classmethod
    def _check_straight(cls, ranks, rank_counts):
        """Проверка на стрит"""
        return (max(ranks) - min(ranks) == 4) and (len(rank_counts) == 5)
    
    @classmethod
    def _determine_combination(cls, is_straight_flush, ranks, rank_counts, is_flush, is_straight):
        """Определение конкретной комбинации"""
        if is_straight_flush and max(ranks) == 14:
            return "ROYAL FLUSH", 0.0001
        elif is_straight_flush:
            return "STRAIGHT FLUSH", 0.001
        elif 4 in rank_counts.values():
            return "FOUR OF A KIND", 0.02
        elif sorted(rank_counts.values()) == [2, 3]:
            return "FULL HOUSE", 0.14
        elif is_flush:
            return "FLUSH", 0.19
        elif is_straight:
            return "STRAIGHT", 0.39
        elif 3 in rank_counts.values():
            return "THREE OF A KIND", 2.11
        elif list(rank_counts.values()).count(2) == 2:
            return "TWO PAIR", 4.75
        elif 2 in rank_counts.values():
            return "ONE PAIR", 42.2
        else:
            return "HIGH CARD", 50