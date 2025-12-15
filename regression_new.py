import pandas as pd
from pathlib import Path
import numpy as np
import sys
_main_dir = Path(__file__).resolve().parent 
if str(_main_dir) not in sys.path:
    sys.path.insert(0, str(_main_dir))

from reg_package.process import train_eval_tree, drop_correlated

ROOT = Path(__file__).resolve().parents[0]
PATH_DATA = ROOT / "data"
PATH_OUTPUT = ROOT / "output"

data_frames ={}
for file_path in PATH_DATA.glob("*.csv"):
    key_name = file_path.stem 
    df = pd.read_csv(file_path)
    data_frames[key_name] = df
    print(f"Geladen: {key_name} -> Shape: {df.shape}")


if __name__ == "__main__":

    # Possum Dataset
    possum_tree, results_possum = train_eval_tree(
        name="Possum",
        df=data_frames["possum"],
        target_col="footlgth",
        cat_cols=["site", "sex", "Pop"],
        )

    filename = f"possum_tree.csv"
    file_path = PATH_OUTPUT / filename
    results_possum.to_csv(file_path, index=False)

    # Student Dataset
    student_tree, results_student = train_eval_tree(
        name="Student",
        df=data_frames["Student_Performance"],
        target_col="Performance Index",
        cat_cols=["Hours Studied", "Extracurricular Activities", "Sample Question Papers Practiced"]
        )

    filename = f"student_tree.csv"
    file_path = PATH_OUTPUT / filename
    results_possum.to_csv(file_path, index=False)


    # Cancer Dataset
    cancer_data = data_frames["cancer_reg"].drop(columns= ["pctsomecol18_24", "binnedinc", "geography"])
    
    features_only = cancer_data.drop(columns=['target_deathrate'])
    columns_to_drop = drop_correlated(features_only, 0.80)
    cancer_reduced = cancer_data.drop(columns=columns_to_drop)

    cancer_reduced["log_deathrate"] = np.log1p(cancer_reduced['target_deathrate'])
    
    cancer_tree, results_cancer = train_eval_tree(
        name="Cancer",
        df=cancer_reduced.drop(columns = ["target_deathrate"]), 
        target_col="log_deathrate",
        cat_cols=[])
    
    filename = f"cancer_tree.csv"
    file_path = PATH_OUTPUT / filename
    results_cancer.to_csv(file_path, index=False)


