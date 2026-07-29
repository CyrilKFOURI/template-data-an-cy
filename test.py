import pandas as pd
from profiler_generator import DataProfiler, DataReconstructor

# 1. Charger votre dataset d'origine
df = pd.read_csv("vos_donnees.csv")

# 2. Profiler le dataset
profiler = DataProfiler()
profiler.fit(df)

# 3. Sauvegarder les métadonnées (le modèle statistique), le rapport Markdown et les CSV
output_dir = "mon_analyse"
profiler.to_json(f"{output_dir}/metadata.json")
profiler.save_report_and_matrices(df, output_dir)

# -------------------------------------------------------------
# 4. Reconstruction (ne nécessite PAS le dataset d'origine !)
# -------------------------------------------------------------
reconstructor = DataReconstructor()
reconstructor.load_metadata(f"{output_dir}/metadata.json")

# Générer 5000 nouvelles lignes statistiquement cohérentes
df_synth = reconstructor.generate(num_rows=5000, seed=42)

# Sauvegarder le dataset reconstruit
df_synth.to_csv(f"{output_dir}/donnees_reconstruites.csv", index=False)
