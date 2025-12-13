import torch
import torchaudio
import numpy as np
import soundfile as sf
import io

print(f"Torchaudio version: {torchaudio.__version__}")
print(f"Soundfile version: {sf.__version__}")

# Create a dummy in-memory wav file
sr = 16000
data = np.random.uniform(-1, 1, size=(sr,)).astype(np.float32)
buffer = io.BytesIO()
sf.write(buffer, data, sr, format='WAV')
buffer.seek(0)

print("\nAttempting to load with torchaudio (default)...")
try:
    y, sr_out = torchaudio.load(buffer)
    print("Success with default backend!")
except Exception as e:
    print(f"Failed with default: {e}")
    
buffer.seek(0)
print("\nAttempting to load with backend='soundfile'...")
try:
    y, sr_out = torchaudio.load(buffer, backend="soundfile")
    print("Success with soundfile backend!")
except Exception as e:
    print(f"Failed with backend='soundfile': {e}")
