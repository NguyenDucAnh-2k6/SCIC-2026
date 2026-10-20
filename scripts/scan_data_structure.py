import os
from pathlib import Path
import pandas as pd
import zipfile

data_dir = Path("data")

print("=== SCANNING DATA DIRECTORY ===")
for item in sorted(data_dir.iterdir()):
    if item.is_file():
        print(f"File: {item.name} ({item.stat().st_size / (1024*1024):.2f} MB)")
    elif item.is_dir():
        print(f"Directory: {item.name}")
        for sub in sorted(item.iterdir()):
            if sub.is_file():
                print(f"  - {sub.name} ({sub.stat().st_size / (1024*1024):.2f} MB)")
            elif sub.is_dir():
                print(f"  + {sub.name}/")
