import tempfile
import urllib.request
import os
import sys

url = "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
dest_dir = r"c:\Users\welcome\Downloads\Gojan AI\Gojan-AI-Voice-Assistant\models\gguf"
dest_file = os.path.join(dest_dir, "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf")

os.makedirs(dest_dir, exist_ok=True)

# Expected size roughly 4.37 GB (4370000000 bytes). Checking precisely via HEAD.
try:
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req) as resp:
        expected_size = int(resp.headers.get('Content-Length', 0))
    print(f"Target size: {expected_size / 1024 / 1024:.2f} MB")
except Exception as e:
    print("Could not get HEAD:", e)
    sys.exit(1)

current_size = 0
if os.path.exists(dest_file):
    current_size = os.path.getsize(dest_file)
    if current_size == expected_size:
        print("Model already fully downloaded.")
        sys.exit(0)
    elif current_size > expected_size:
        print("Existing file larger than expected. Restarting.")
        os.remove(dest_file)
        current_size = 0
    else:
        print(f"Resuming download from {current_size / 1024 / 1024:.2f} MB...")

headers = {}
if current_size > 0:
    headers['Range'] = f'bytes={current_size}-'

req = urllib.request.Request(url, headers=headers)
try:
    with urllib.request.urlopen(req) as response:
        with open(dest_file, 'ab' if current_size > 0 else 'wb') as f:
            chunk_size = 8 * 1024 * 1024
            downloaded = current_size
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (100 * 1024 * 1024) < chunk_size:
                    print(f"Progress: {downloaded / 1024 / 1024:.2f} / {expected_size / 1024 / 1024:.2f} MB")
    print("\nDownload complete!")
except Exception as e:
    print("Failed during download:", e)
    sys.exit(1)
