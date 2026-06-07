import time
import pipeline.models as models


class ASRAgent:
    """Transcribes audio file to raw text using Whisper."""

    def run(self, audio_file_path: str) -> str:
        print(f"[ASRAgent] Transcribing {audio_file_path}")
        t0 = time.time()
        result = models.whisper_pipe(audio_file_path)['text']
        elapsed = round(time.time() - t0, 2)
        print(f"[ASRAgent] Done in {elapsed}s | {len(result)} chars")
        print(f"[ASRAgent] Preview: {result[:200]}...")
        return result
