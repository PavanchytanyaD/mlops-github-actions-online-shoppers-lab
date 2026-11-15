"""Simulates new labelled data arriving: moves the next file from data/incoming/
into data/batches/. Committing and pushing it starts the retraining workflow."""
import shutil

from data import BATCH_DIR, INCOMING_DIR, batch_files

if __name__ == "__main__":
    waiting = batch_files(INCOMING_DIR)
    if not waiting:
        print("No new batches left in data/incoming/")
    else:
        shutil.move(str(waiting[0]), BATCH_DIR / waiting[0].name)
        print(f"Moved {waiting[0].name} to data/batches/ "
              f"({len(batch_files())} batches now, {len(waiting) - 1} still incoming)")
