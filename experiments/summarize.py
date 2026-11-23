import argparse,pandas as pd
from pathlib import Path
if __name__=="__main__":
 ap=argparse.ArgumentParser(); ap.add_argument("--results-dir",default="../results"); a=ap.parse_args(); p=Path(a.results_dir); df=pd.read_csv(p/"fold_metrics.csv"); ms=["auc","macro_f1","accuracy","macro_precision","macro_recall"]; agg=df.groupby(["dataset","method"])[ms].agg(["mean","std"]).reset_index(); agg.to_csv(p/"summary.csv",index=False); print(agg.to_string(index=False))
