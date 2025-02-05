import numpy as np
import pandas as pd
import utils

# Read data
# Should contain colums
# - target from NGS
# - rapid test result
# - EAGLE score
df = pd.read_csv('example_data.csv')

# Rapid test historical data
rapid_npv = 0.954
rapid_npv_95ci_inf = 0.942
rapid_npv_95ci_sup = 0.965
rapid_ppv = 0.988
rapid_ppv_95ci_inf = 0.979
rapid_ppv_95ci_sup = 0.988

# Simulate deployment
utils.plot_simulation(
    df,
    metric='reduction',
    rapid_performance=(rapid_npv, rapid_ppv),
    rapid_confint=(rapid_npv_95ci_inf, rapid_npv_95ci_sup, rapid_ppv_95ci_inf, rapid_ppv_95ci_sup),
    se_max=0.5,
    sp_min=0.5,
    N=201,
    target_col='target',
    rapid_col='rapid',
    eagle_col='score'
)

# Assisted performance metrics
threshold_npv = 0.023
threshold_ppv = 0.997
utils.get_performance_assisted_bootstrapped(df, th0=[threshold_npv], th1=[threshold_ppv], n=1000, target_col='target', rapid_col='rapid', eagle_col='score')
