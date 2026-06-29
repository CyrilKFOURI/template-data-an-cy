def on_export(b):
    out_exp.clear_output()
    with out_exp:
        if 'df_clean' not in globals() or df_clean is None:
            print('Charger les données d\'abord (Cellule 2).')
            return
        
        # Vérification des méthodes exécutées
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
                # --- CORRECTION : Gestion des colonnes dupliquées ---
                if not det.columns.is_unique:
                    print(f"Attention : Doublons détectés dans {name}. Nettoyage en cours...")
                    # Renomme les colonnes dupliquées avec un suffixe _1, _2, etc.
                    det = det.loc[:, ~det.columns.duplicated()].copy()
                
                # Assignation des colonnes
                det[PERIOD_LABEL] = period_label
                det[COUNTRY_CODE] = code
                
                p = os.path.join(OUTPUT_DIR, f"{name}.parquet")
                det.to_parquet(p, index=False)
                print(f"{name}.parquet - {len(det):,} lignes")
            else:
                print(f"{name} vide - parquet non généré")

        print('\nExport Excel...')
        excel_path = os.path.join(OUTPUT_DIR, f"reuse_{code}.xlsx")
        
        # Préparation sécurisée des DataFrames pour l'Excel
        build_excel(
            d1 if d1 is not None else pd.DataFrame(),
            d2 if d2 is not None else pd.DataFrame(),
            d3 if d3 is not None else pd.DataFrame(),
            code, period_label, excel_path
        )
        
        print(f"\nTous les fichiers dans : {os.path.abspath(OUTPUT_DIR)}")
        print('Filtres Excel sur la ligne 5 de chaque onglet.')

# Liaison du bouton (à conserver tel quel si déjà fait)
# btn_exp.on_click(on_export)
