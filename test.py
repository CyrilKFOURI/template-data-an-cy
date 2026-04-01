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
