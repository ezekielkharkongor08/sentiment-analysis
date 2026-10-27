import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = str(BASE_DIR / "data" / "raw")

def id_generator(start=1, prefix=None):
    current = start
    while True:
        if prefix:
            yield f"{prefix}_{current:04d}"
        else:
            yield current
        current += 1

def resolve_overlapping_ids(directory=None, file_list=None, create_merged=True):
    if directory is None:
        directory = DEFAULT_DATA_DIR
    if file_list is None:
        file_list = [
            "mygov_scraped_dataset.json",
            "pgportal_scraped_dataset.json"
        ]
        
    print("=" * 65)
    print("[*] RESOLVING OVERLAPPING IDs ACROSS DATASETS")
    print(f"[*] Target Directory: {directory}")
    print("=" * 65 + "\n")
    
    gen = id_generator(start=1)
    
    all_combined_records = []
    
    for filename in file_list:
        file_path = os.path.join(directory, filename)
        if not os.path.exists(file_path):
            print(f"[-] File not found: {file_path}, skipping.")
            continue
            
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)
            
        old_start_id = records[0]["id"] if records else 0
        old_end_id = records[-1]["id"] if records else 0
        
        for record in records:
            record["id"] = next(gen)
            all_combined_records.append(record)
            
        new_start_id = records[0]["id"] if records else 0
        new_end_id = records[-1]["id"] if records else 0

    # nono overlapping
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
            
        print(f"[OK] {filename}:")
        print(f"     - Total records: {len(records)}")
        print(f"     - Old IDs range: {old_start_id} to {old_end_id}")
        print(f"     - New IDs range: {new_start_id} to {new_end_id} (Resolved!)\n")

#save in master file
    if create_merged and all_combined_records:
        merged_file_path = os.path.join(directory, "master_combined_dataset.json")
        with open(merged_file_path, "w", encoding="utf-8") as f:
            json.dump(all_combined_records, f, indent=2, ensure_ascii=False)
        print("=" * 65)
        print("[OK] Created Combined Master File: master_combined_dataset.json")
        print(f"[OK] Total Unified Records: {len(all_combined_records)} (IDs: 1 to {len(all_combined_records)})")
        print("=" * 65)

if __name__ == "__main__":
    resolve_overlapping_ids()
