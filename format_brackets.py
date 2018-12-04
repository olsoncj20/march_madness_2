import pandas as pd
import os

for file in os.listdir("brackets"):
    if file == "__init__.py":
        continue
    year = file.split("_")[0]
    overlap = file.split("_")[1]
    bracket_num = file.split("_")[2]

    df = pd.read_csv("brackets/"+file)

    if "Round 1" in df.columns:
        continue

    q_matrix = pd.read_csv("q_matrix_"+year+".csv")
    q_matrix["Team_ID"] = q_matrix["Unnamed: 0"]
    q_matrix = q_matrix.set_index("Team_ID")

    df.index = q_matrix.index
    df.columns = ["Round "+str(i) for i in range(1, 7)]

    df.to_csv("brackets/"+file)