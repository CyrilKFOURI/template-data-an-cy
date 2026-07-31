# vérifie l'hypothèse : MORE_30D devrait ≈ somme des tranches au-dessus de 30j
check = df["ARRS_MORE_30D"] - (df["ARRS_BTWN_31_60D"] + df["ARRS_BTWN_61_90D"]
                                + df["ARRS_BTWN_91_180D"] + df["ARRS_BTWN_181_270D"]
                                + df["ARRS_MORE_270D"])
print(check.abs().describe())  
