import urllib.request
import gzip
import shutil
from pathlib import Path

out_dir = Path("data/sociopatterns")
out_dir.mkdir(parents=True, exist_ok=True)

files = [
    # 1. Primary school (Lyon - 242 học sinh, 10 giáo viên)
    ("primaryschool.csv.gz", "https://www.sociopatterns.org/assets/data/primaryschool.csv.gz"),
    # 2. High school (Marseille 2013 - mạng tiếp xúc temporal proximity giữa 9 lớp)
    ("HighSchool2013_proximity_net.csv.gz", "https://www.sociopatterns.org/assets/data/HighSchool2013_proximity_net.csv.gz"),
    # 3. High school (Mạng quan hệ bạn bè tự khai báo)
    ("HighSchool2013_friendship_network.csv.gz", "https://www.sociopatterns.org/assets/data/HighSchool2013_friendship_network.csv.gz")
]

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

for fname, url in files:
    gz_path = out_dir / fname
    csv_path = out_dir / fname.replace(".gz", "")
    print(f"Downloading {fname} from {url}...")
    
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req) as resp, open(gz_path, 'wb') as f_out:
        shutil.copyfileobj(resp, f_out)
        
    print(f"-> Saved {fname} ({gz_path.stat().st_size / 1024:.1f} KB). Extracting...")
    with gzip.open(gz_path, 'rb') as f_in, open(csv_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
    print(f"-> Extracted to {csv_path.name} ({csv_path.stat().st_size / 1024:.1f} KB).")

print("\nSuccess! SocioPatterns school datasets are ready at data/sociopatterns/")

