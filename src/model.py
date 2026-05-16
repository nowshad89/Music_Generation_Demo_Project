import random
import pickle
from collections import defaultdict

class MarkovMusicModel:
    def __init__(self, order):
        self.order = order
        self.transitions = defaultdict(list)

    def train(self, sequences):
        for seq in sequences:
            for i in range(len(seq) - self.order):
                state = tuple(seq[i : i + self.order])
                next_note = seq[i + self.order]
                self.transitions[state].append(next_note)
        print(f"Model trained with {len(self.transitions)} unique states.")

    def generate(self, length=30):
        if not self.transitions:
            raise ValueError("Model must be trained before generation.")
            
        current_state = random.choice(list(self.transitions.keys()))
        output = list(current_state)

        for _ in range(length):
            possible_next_notes = self.transitions.get(tuple(current_state))
            if not possible_next_notes:
                current_state = random.choice(list(self.transitions.keys()))
                continue
                
            next_note = random.choice(possible_next_notes)
            output.append(next_note)
            current_state = output[-self.order:]
        return output

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)