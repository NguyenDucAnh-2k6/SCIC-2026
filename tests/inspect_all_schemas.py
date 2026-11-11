import pandas as pd
import zipfile
from pathlib import Path
import json

def inspect_csv(path, sep=',', nrows=3):
    try:
        df = pd.read_csv(path, sep=sep, nrows=nrows, low_memory=False)
        print(f"\n--- FILE: {path} (cols: {len(df.columns)}) ---")
        for col in df.columns:
            val = df[col].iloc[0] if len(df) > 0 else None
            dtype = str(df[col].dtype)
            print(f"  {col}: {dtype} (ex: {repr(val)})")
    except Exception as e:
        print(f"Error reading {path}: {e}")

print("=== 1. ASSISTMENTS ===")
inspect_csv("data/assistments/ASSISTments_skill_builder_data.csv")

print("\n=== 2. ASSISTMENTS AFFECT ===")
inspect_csv("data/assistments_affect/2012-2013-data-with-predictions-4-final.csv")

print("\n=== 3. BRIDGE TO ALGEBRA (COGNITIVE TUTOR) ===")
inspect_csv("data/bridge_to_algebra_2006_2007/bridge_to_algebra_2006_2007_train.txt", sep='\t')

print("\n=== 4. OULAD ===")
inspect_csv("data/oulad/studentInfo.csv")
inspect_csv("data/oulad/studentAssessment.csv")
inspect_csv("data/oulad/studentVle.csv")
inspect_csv("data/oulad/assessments.csv")
inspect_csv("data/oulad/courses.csv")
inspect_csv("data/oulad/vle.csv")
inspect_csv("data/oulad/studentRegistration.csv")

print("\n=== 5. SOCIOPATTERNS ===")
try:
    with open("data/sociopatterns/primaryschool.csv") as f:
        print("primaryschool.csv first 3 lines:")
        for _ in range(3):
            print(" ", f.readline().strip())
except Exception as e:
    print(e)

try:
    with open("data/sociopatterns/HighSchool2013_proximity_net.csv") as f:
        print("HighSchool2013_proximity_net.csv first 3 lines:")
        for _ in range(3):
            print(" ", f.readline().strip())
except Exception as e:
    print(e)

print("\n=== 6. IN-GAUGE & EN-GAGE ===")
in_gauge_dir = Path("data/in_gauge_engage")
for p in in_gauge_dir.rglob("*.csv"):
    inspect_csv(p, nrows=2)
    break

print("\n=== 7. EDNET ZIP HEADERS ===")
for kt in [1, 2, 3, 4]:
    zpath = f"data/EdNet-KT{kt}.zip"
    if Path(zpath).exists():
        with zipfile.ZipFile(zpath, 'r') as z:
            names = z.namelist()
            csvs = [n for n in names if n.endswith('.csv')]
            if csvs:
                sample_csv = csvs[0]
                with z.open(sample_csv) as f:
                    df = pd.read_csv(f, nrows=2)
                    print(f"EdNet-KT{kt} ({sample_csv}): cols={list(df.columns)}")
