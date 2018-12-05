import pandas as pd
import math
from numpy.random import normal


for year in ["2011", "2012", "2013"]:
    df = pd.read_csv("q_matrix_"+year+".csv").set_index("Team_ID")
    df = df.applymap(lambda x: math.log(x/(1-x)))
    df = df.applymap(lambda x: x+normal(0,2))
    df = df.applymap(lambda x: math.exp(x)/(1+math.exp(x)))
    df.to_csv("q_matrix_noise"+year+".csv")