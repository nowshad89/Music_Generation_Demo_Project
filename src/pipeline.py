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
        """Utility to play the generated MIDI immediately with an interactive stop."""
        import pygame
        import threading

        try:
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            print("🔊 Music is playing...")
            print("👉 Press [ENTER] in this terminal at any time to STOP the music.")

            # Create a background thread that waits for the user to hit Enter
            stop_event = threading.Event()
            def wait_for_input():
                input()  # Waits silently for the user to press Enter
                stop_event.set()

            input_thread = threading.Thread(target=wait_for_input, daemon=True)
            input_thread.start()

            # Keep checking if the music finished naturally OR if the user hit Enter
            while pygame.mixer.music.get_busy():
                if stop_event.is_set():
                    print("\n🛑 Playback stopped early by user.")
                    break
                pygame.time.Clock().tick(10) # Efficient low-CPU sleep loop

        except Exception as e:
            print(f"An error occurred during playback: {e}")
        finally:
            pygame.mixer.music.stop()
            pygame.mixer.quit()