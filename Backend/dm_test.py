import numpy as np
import pandas as pd 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))
from dieboldmariano import dm_test

#from diebold_mariano_test import diebold_mariano_test 
#H0: both models perform the same H1: bridge model is better
def dm_test(df): #takes input from RMSFE function (error for each obs)
    comparisons = [
    ("RMSFE_model_AR_h1", "RMSFE_model_ADL_bridge_m3"),
    ("RMSFE_model_AR_h2", "RMSFE_model_ADL_bridge_m6"),
    ("RMSFE_model_RF_h1", "RMSFE_model_RF_bridge_m3"),
    ("RMSFE_model_RF_h2", "RMSFE_model_RF_bridge_m6")]

    for model1, model2 in comparisons:
        se_diff = df[model1]**2 - df[model2]**2  #benchmark model - bridge model
