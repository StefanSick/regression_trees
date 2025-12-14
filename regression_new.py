import pandas as pd
from pathlib import Path
import numpy as np
import sys
_main_dir = Path(__file__).resolve().parent 
if str(_main_dir) not in sys.path:
    sys.path.insert(0, str(_main_dir))

from reg_package.process import train_eval_tree

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
    possum_tree, results_possum = train_eval_tree(
        name="Possum",
        df=data_frames["possum"],
        target_col="footlgth",
        cat_cols=["site", "sex", "Pop"],
        )

    filename = f"possum_tree.csv"
    file_path = PATH_OUTPUT / filename
    results_possum.to_csv(file_path, index=False)



    household_tree, results_household = train_eval_tree(
        name="Household_Income",
        df=data_frames["Household_Income"], 
        target_col="Income",
        cat_cols=["Education_Level", "Occupation", "Number_of_Dependents", "Location", "Marital_Status", "Employment_Status", "Household_Size", "Homeownership_Status", "Type_of_Housing", "Gender",  "Primary_Mode_of_Transportation"])

    filename = f"household_tree.csv"
    file_path = PATH_OUTPUT / filename
    results_household.to_csv(file_path, index=False)


    cancer_tree, results_cancer = train_eval_tree(
        name="Cancer",
        df=data_frames["cancer_reg"].drop(columns= ["pctsomecol18_24", "binnedinc", "geography"]), 
        target_col="target_deathrate",
        cat_cols=[])
    
    filename = f"cancer_tree.csv"
    file_path = PATH_OUTPUT / filename
    results_cancer.to_csv(file_path, index=False)


