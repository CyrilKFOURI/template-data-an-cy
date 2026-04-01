import plotly.express as px

def skew_label(sk):
    a=abs(sk)
    if a<0.5:return "quasi symetrique"
    if a<1:return "asymetrie moderee"
    return "forte asymetrie"

def plot_distribution(df,col,nbins=60,show_plot=True):
    s=df[col].dropna()
    if show_plot:
        fig=px.histogram(s,x=col,nbins=nbins,histnorm="probability density",marginal="box",title=f"{col} - Distribution")
        fig.update_layout(template="plotly_white",xaxis_title="Valeur",yaxis_title="Densite")
        fig.show()
    stats=s.agg(mean="mean",median="median",std="std",skew="skew")
    print(f"\n{col}")
    print(f"- mean={stats['mean']:.3f}, median={stats['median']:.3f}, std={stats['std']:.3f}, skew={stats['skew']:.3f} ({skew_label(stats['skew'])})")
    return stats






from xlsx2csv import Xlsx2csv
import pyarrow.csv as pv
import pyarrow.parquet as pq
import os
from datetime import datetime

def process_files(country, start_yyyymm, end_yyyymm):
    
    os.makedirs("NewDataCSV", exist_ok=True)
    os.makedirs("NewDataParquet", exist_ok=True)

    start = datetime.strptime(start_yyyymm, "%Y%m")
    end = datetime.strptime(end_yyyymm, "%Y%m")

    current = start

    while current <= end:
        date_str = current.strftime("%Y%m")

        excel_path = f"NewData/NOVA {country} {date_str}.xlsx"
        csv_path = f"NewDataCSV/NOVA {country} {date_str}.csv"
        parquet_path = f"NewDataParquet/NOVA {country} {date_str}.parquet"

        print(f"Processing {date_str}...")

        try:
            # 1) Excel → CSV
            Xlsx2csv(excel_path).convert(csv_path)

            # 2) CSV → Parquet (rapide avec pyarrow)
            table = pv.read_csv(csv_path)
            pq.write_table(table, parquet_path)

        except Exception as e:
            print(f"Erreur sur {date_str} : {e}")

        # mois suivant
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    print("Done ✅")



process_files(
    country="SPAIN",
    start_yyyymm="202602",
    end_yyyymm="202604"
)
