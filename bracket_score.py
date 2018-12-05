import pandas as pd

scoring = {
    1: 10,
    2: 20,
    3: 40,
    4: 80,
    5: 160,
    6: 320
}


def bracket_score(file_path):
    if "noise" in file_path:
        year = file_path.split("_")[0]
        overlap_coef = file_path.split("_")[2]
        lineup_number = file_path.split("_")[3]
    else:
        year = file_path.split("_")[0]
        overlap_coef = file_path.split("_")[1]
        lineup_number = file_path.split("_")[2]

    df = pd.read_csv("brackets/" + file_path).set_index("Team_ID")
    master = pd.read_csv(year + "_outcome.csv").set_index("Team_ID")

    score = 0
    for i in range(1,7):
        bracket_col = df["Round "+str(i)]
        master_col = master["Round "+str(i)]
        score_round = scoring[i]
        score += score_round*bracket_col.dot(master_col)
    return score


# EXAMPLE:
print(bracket_score("2011_30_1.csv"))