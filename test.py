cls_map = {"1":"PV","2":"Station wagon","3":"Vans","4":"LCV","5":"Medium-duty CV","6":"HCV","7":"Minibus / People Mover","8":"CV with truck registration"}

def _map(v):
    v = str(v).strip()
    if v in cls_map:
        return cls_map[v]
    try:
        return cls_map.get(str(int(float(v))), "Not Identified")
    except:
        return v 

df["CLS_VEHICLE_TYPE"] = df["CLS_VEHICLE_TYPE"].apply(_map)
print(df["CLS_VEHICLE_TYPE"].value_counts())
