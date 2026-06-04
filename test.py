import os
import glob

folder = "chemin/vers/tes/parquets"

for f in glob.glob(f"{folder}/*.parquet"):
    base = os.path.basename(f)

    if "-merged" in base:
        new_base = base.replace("-merged", "")
        old_path = f
        new_path = os.path.join(folder, new_base)

        os.rename(old_path, new_path)

        print(f"{base} -> {new_base}")