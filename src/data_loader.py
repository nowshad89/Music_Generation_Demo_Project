from music21 import corpus, note

class MusicDataLoader:
    def __init__(self, limit=300):
        self.limit = limit

    def get_bach_sequences(self):
        sequences = []
        # Search for Bach bundle
        bach_bundle = corpus.search('bach', 'composer')[:self.limit]
        
        if len(bach_bundle) == 0:
            bach_bundle = corpus.corpora.CoreCorpus().search('bach/bwv')[:self.limit]

        for entry in bach_bundle:
            try:
                score = entry.parse()
                if len(score.parts) > 0:
                    soprano = score.parts[0].flatten().notes
                    # Combined Feature Extraction: Pitch_Duration
                    notes_with_timing = [
                        f"{n.pitch.nameWithOctave}_{n.duration.quarterLength}" 
                        for n in soprano if isinstance(n, note.Note)
                    ]
                    if notes_with_timing:
                        sequences.append(notes_with_timing)
            except Exception as e:
                print(f"Skipping a file due to error: {e}")
        
        print(f"Extracted {len(sequences)} sequences.")
        return sequences