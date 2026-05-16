from music21 import stream, note
from src.data_loader import MusicDataLoader
from src.model import MarkovMusicModel
import pygame
import time

class MusicPipeline:
    def __init__(self, model_path="artifacts/bach_model.pkl", order=3):
        self.model_path = model_path
        self.loader = MusicDataLoader()
        self.model = MarkovMusicModel(order)

    def run_training_flow(self):
        data = self.loader.get_bach_sequences()
        self.model.train(data)
        self.model.save(self.model_path)

    def run_generation_flow(self, output_path, length=50):
        # Load trained model
        self.model = MarkovMusicModel.load(self.model_path)
        tokens = self.model.generate(length)
        
        # Convert tokens to music21 stream
        output_stream = stream.Stream()
        for token in tokens:
            try:
                p_name, d_val = token.split('_')
                new_note = note.Note(p_name)
                new_note.quarterLength = float(d_val)
                output_stream.append(new_note)
            except:
                continue
        
        output_stream.write('midi', fp=output_path)
        print(f"File saved to {output_path}")

    def play_output(self, file_path):
        """Utility to play the generated MIDI immediately."""
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(1)
        pygame.mixer.quit()