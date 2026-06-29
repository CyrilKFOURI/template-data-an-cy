def on_export(b):
    out_exp.clear_output()
    with out_exp:
        # Vérification de l'existence des données
        if 'df_clean' not in globals() or df_clean is None:
            print('Charger les données d\'abord (Cellule 2).')
            return
        
        # Vérification que les méthodes ont été calculées
        missing = [n for n, det in [("M1", d1), ("M2", d2), ("M3", d3)] if det is None]
        if missing:
            print(f"Méthode(s) non encore exécutée(s) : {', '.join(missing)}")
            print('Lance les boutons correspondants avant d\'exporter.')
            return

        code = country_dropdown.value
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print('Export des parquets...')

        for det, name in [(d1, f"reuse_{code}_method1"), 
                          (d2, f"reuse_{code}_method2"), 
                          (d3, f"reuse_{code}_method3")]:
            
            if len(det) > 0:
                # Création d'une copie pour éviter de modifier le DataFrame original
                det = det.copy()
                
                # 1. Suppression des colonnes en double
                if not det.columns.is_unique:
                    det = det.loc[:, ~det.columns.duplicated()]
                
                # 2. Conversion forcée de ID_CONTRAT en string pour éviter les erreurs de type
                if 'ID_CONTRAT' in det.columns:
                    det['ID_CONTRAT'] = det['ID_CONTRAT'].astype(str)
                
                # Assignation des variables de contexte
                det[PERIOD_LABEL] = period_label
                det[COUNTRY_CODE] = code
                
                # Sauvegarde au format Parquet
                p = os.path.join(OUTPUT_DIR, f"{name}.parquet")
                det.to_parquet(p, index=False)
                print(f"{name}.parquet - {len(det):,} lignes")
            else:
                print(f"{name} vide - parquet non généré")

        # Export Excel
        print('\nExport Excel...')
        excel_path = os.path.join(OUTPUT_DIR, f"reuse_{code}.xlsx")
        
        build_excel(
            d1 if d1 is not None else pd.DataFrame(),
            d2 if d2 is not None else pd.DataFrame(),
            d3 if d3 is not None else pd.DataFrame(),
            code, period_label, excel_path
        )
        
        print(f"\nTous les fichiers sont dans : {os.path.abspath(OUTPUT_DIR)}")
        print('Filtres Excel sur la ligne 5 de chaque onglet.')
