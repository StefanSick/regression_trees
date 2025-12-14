import pandas as pd
from pathlib import Path
import numpy as np
import sys
_main_dir = Path(__file__).resolve().parent 
if str(_main_dir) not in sys.path:
    sys.path.insert(0, str(_main_dir))

from reg_package.process import train_eval_rf, drop_correlated

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
    rf_possum, results_possum= train_eval_rf(
    name="Possum",
    df=data_frames["possum"].drop(columns=["case"]), 
    target_col="footlgth",
    cat_cols=["site", "sex", "Pop"])

    filename = f"possum_results.csv"
    file_path = PATH_OUTPUT / filename
    results_possum.to_csv(file_path, index=False)
    
    # Household dataset
    household_data = data_frames["Household_Income"]
    household_data["log_income"] = np.log1p(household_data['Income']).drop(columns = ["Income"])
   
    rf_household, results_household = train_eval_rf(
    name="Household_Income",
    df=data_frames["Household_Income"].drop(columns = ["Income"]), 
    target_col="log_income",
    cat_cols=["Education_Level", "Occupation", "Number_of_Dependents", "Location", "Marital_Status", "Employment_Status", "Household_Size", "Homeownership_Status", "Type_of_Housing", "Gender",  "Primary_Mode_of_Transportation"])
    
    filename = f"household_results.csv"
    file_path = PATH_OUTPUT / filename
    results_household.to_csv(file_path, index=False)

    # Cancer dataset
    cancer_data = data_frames["cancer_reg"].drop(columns= ["pctsomecol18_24", "binnedinc", "geography"])
    
    features_only = cancer_data.drop(columns=['target_deathrate'])
    columns_to_drop = drop_correlated(features_only, 0.80)
    cancer_reduced = cancer_data.drop(columns=columns_to_drop)

    cancer_reduced["log_deathrate"] = np.log1p(cancer_reduced['target_deathrate'])
    
    rf_cancer, results_cancer = train_eval_rf(
    name="Cancer",
    df=cancer_reduced.drop(columns= ["target_deathrate"]), 
    target_col="log_deathrate",
    cat_cols=[])
    
    filename = f"cancer_results.csv"
    file_path = PATH_OUTPUT / filename
    results_cancer.to_csv(file_path, index=False)
    data_frames["cancer_reg"]