import os
import glob

folder = "ton_path"

files = glob.glob(f"{folder}/*.parquet")

merged_files = [
    f for f in files
    if "-merged" in os.path.basename(f)
]

print(f"Nombre de fichiers avec '-merged' : {len(merged_files)}")

for f in merged_files:
    base = os.path.basename(f)
    new_base = base.replace("-merged", "")

    os.rename(
        f,
        os.path.join(folder, new_base)
    )

    print(f"{base} -> {new_base}")

print("Renommage terminé")