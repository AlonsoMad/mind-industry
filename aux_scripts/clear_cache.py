import joblib
import os
import shutil
import time

CACHE_DIR = "cache"  # Change this to your cache directory
MAX_AGE = (3600 *2) # Delete files older than 1 hour (3600 seconds)

def clear_old_cache(cache_dir=CACHE_DIR, max_age=MAX_AGE):
    """Clears only old joblib cache files, keeping recent ones."""
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return

    now = time.time()
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.isfile(file_path) and now - os.path.getmtime(file_path) > max_age:
                os.remove(file_path)
                print(f"Deleted: {file_path}")

if __name__ == "__main__":
    clear_old_cache()
