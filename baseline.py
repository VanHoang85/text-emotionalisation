import random
from lexicons import patterns


class NaiveStyleTransfer:
    def __init__(self):
        self.emotion_exclamations = self.load_emotion_exclamations()
        self.emotion_phrases = self.load_emotion_phrases()

    def load_emotion_exclamations(self):
        return {'anger': patterns.anger_exclamations,
                'happiness': patterns.happiness_exclamations,
                'sadness': patterns.sadness_exclamations}

    def load_emotion_phrases(self):
        anger_phrases, happiness_phrases, sadness_phrases = [], [], []
        for phrases in patterns.anger_patterns.values():
            anger_phrases.extend(phrases['phrases'])
        for phrases in patterns.happiness_patterns.values():
            happiness_phrases.extend(phrases['phrases'])
        for phrases in patterns.sadness_patterns.values():
            sadness_phrases.extend(phrases['phrases'])
        return {'anger': anger_phrases,
                'happiness': happiness_phrases,
                'sadness': sadness_phrases}

    def generate_sent(self, sentence: str, emotion: str) -> str:
        phrase = random.choice(self.emotion_phrases.get(emotion)) + '!'
        return random.choice([' '.join([sentence, '!', phrase]), ' '.join([phrase, sentence])])

    def generate_batch(self, sentence: str, emotion: str, num: int) -> list:
        return [self.generate_sent(sentence, emotion) for i in range(num)]

    def generate_sent_with_exclamations(self, sentence: str, emotion: str) -> str:
        phrase = random.choice(self.emotion_exclamations.get(emotion)) + '!'
        return random.choice([' '.join([sentence, '!', phrase]), ' '.join([phrase, sentence])])

    def generate_batch_with_exclamations(self, sentence: str, emotion: str, num: int) -> list:
        return [self.generate_sent(sentence, emotion) for i in range(num)]
