import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from predict import predict_flare

class NewImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if event.src_path.endswith(".png"):
            # 1. Wait for the file to finish writing
            # Give it 2 seconds to settle
            time.sleep(2) 
            
            # 2. Check if file is still empty or missing
            if os.path.exists(event.src_path) and os.path.getsize(event.src_path) > 0:
                print(f"🔭 Processing: {os.path.basename(event.src_path)}")
                try:
                    prob = predict_flare(event.src_path, [1.0, 0.0, 0.0, 960.0])
                    print(f"Probability: {prob*100:.4f}%")
                except Exception as e:
                    print(f"❌ Error reading image: {e}")
            else:
                print(f"⚠️ File {os.path.basename(event.src_path)} not ready yet.")

if __name__ == "__main__":
    path = "/home/aniruddha0516/aditya_l1_project/processed_images"
    print(f"🛰️  Aditya-L1 Watcher Active on: {path}")
    event_handler = NewImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
