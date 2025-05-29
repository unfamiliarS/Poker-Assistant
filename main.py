from models.poker_assistant import PokerAssistant

if __name__ == '__main__':
    model_path = 'trained_model.pt'
    assistant = PokerAssistant(model_path)
    assistant.run()