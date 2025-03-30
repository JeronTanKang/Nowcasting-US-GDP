import numpy as np
import pandas as pd 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend')))


#from diebold_mariano_test import diebold_mariano_test 

"""
    # Perform Diebold-Mariano test
    dm_results = diebold_mariano_test(
        actual_values=forecast_results["Actual_GDP"],
        bridge_forecast=forecast_results["Bridge_Forecast"],
        ar_forecast=forecast_results["AR_Forecast"]
    )

    print("\n Forecast Results:")
    print(forecast_results)

    print("\n Diebold-Mariano Test Results:")
    print(dm_results)
"""
