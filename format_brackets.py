import pandas as pd
import os

for file in os.listdir("brackets"):
    if file == "__init__.py" or file == ".DS_Store":
        continue
    try:
        year = file.split("_")[0]
        overlap = file.split("_")[1]
        bracket_num = file.split("_")[2]
        df = pd.read_csv("brackets/" + file)
    except:
        year = file[:4]
        overlap = file[4:6]
        bracket_num = file.split("_")[1]
        df = pd.read_csv("brackets/" + file)
        file = "_".join([year, overlap, bracket_num])+'.csv'

    if "Round 1" in df.columns:
        continue

    q_matrix = pd.read_csv("q_matrix_"+year+".csv").set_index("Team_ID")

    df.index = q_matrix.index
    df.columns = ["Round "+str(i) for i in range(1, 7)]

    df.to_csv("brackets/"+file)