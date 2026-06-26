import pandas as pd
import xlsxwriter

def exporter_excel_debug(df, nom_fichier):
    print("--- Début du debug ---")
    
    # 1. Vérification du type
    print(f"Type de df reçu: {type(df)}")
    
    # 2. Vérification si vide (utilisant .empty qui est la méthode safe)
    if df.empty:
        print("Erreur: Le DataFrame est vide !")
        return
    else:
        print("Le DataFrame contient des données.")

    # 3. Vérification des colonnes
    print(f"Colonnes détectées: {list(df.columns)}")
    
    try:
        print("Tentative de création du writer...")
        writer = pd.ExcelWriter(f"{nom_fichier}.xlsx", engine='xlsxwriter')
        
        print("Tentative d'écriture du dataframe...")
        df.to_excel(writer, sheet_name='Données', index=True)
        
        workbook  = writer.book
        worksheet = writer.sheets['Données']
        
        # Formats simples
        header_fmt = workbook.add_format({'bold': True, 'fg_color': '#2C3E50', 'font_color': 'white'})
        
        print("Application des styles...")
        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num + 1, value, header_fmt)
            
        writer.close()
        print(f"Succès ! Fichier '{nom_fichier}.xlsx' généré.")
        
    except Exception as e:
        print(f"--- ERREUR CRITIQUE DÉTECTÉE ---")
        print(f"Type d'erreur: {type(e).__name__}")
        print(f"Message: {e}")
        print("--------------------------------")

# --- APPEL ---
# Si le code plante ici, regardez bien la console, 
# le print affichera la ligne coupable.
exporter_excel_debug(df, "test_debug")
