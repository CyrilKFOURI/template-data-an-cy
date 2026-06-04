import os
import glob

folder = "ton_path"

files = glob.glob(f"{folder}/*.parquet")

print(f"{len(files)} fichiers trouvés")

for f in files:
    base = os.path.basename(f)
    print(base)

    if "-merged" in base:
        new_base = base.replace("-merged", "")
        print(f"RENAME: {base} -> {new_base}")

        os.rename(
            f,
            os.path.join(folder, new_base)
        )