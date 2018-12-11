import pandas as pd
import numpy as np

def create_and_reformat_probs(file):
    """
    Creates a pandas DataFrame from a probability csv and reformats it such that the
    team numbers are on both the x and y axis

    Inputs:
        file (csv):  matrix of probabilities that y axis team beats x axis team

    Returns:
        probs (pandas DataFrame)
    """
    probs = pd.read_csv(file)
    ind = np.array(list(probs))
    probs.set_index(ind, inplace=True)
    return probs


def get_prob_win(team_a, team_b, prob_matrix):
    """
    Inputs:
        team_a: team identifier (str or int)
        team_b: team identifier (str or int)
        prob_matrix (pandas DataFrame): matrix of probabilities
        - each entry represents the prob that the row team beats the column team

    Returns:
        prob (float): probability that team a beats team b
    """

    return prob_matrix[str(team_b)][str(team_a)]


def tournament_results_for_specific_year(file, year):
    """
    Creates a pandas DataFrame with all the detailed results of who won which games in a specific year
    Used to figure out who won play in games

    Inputs:
        file (csv): tourney detailed results dating back to 2003
        year (int): tournament year of interest

    Returns:
        results_for_year (pandas DataFrame)
    """

    overall_tourney_results = pd.read_csv(file)
    results_for_year = overall_tourney_results[overall_tourney_results['Season'] == year]
    return results_for_year


def get_tournament_seeds_for_specific_year(file, year):
    """
    Creates a pandas DataFrame with the seeds from a specific year

    Inputs:
        file (csv): tourney seeds dating back to 1985
        year (int): tournament year of interest

    Returns:
        seeds_for_year (pandas DataFrame): seeds for specific year
    """

    seeds = pd.read_csv(file)
    seeds_for_year = seeds[seeds['Season'] == year]
    return seeds_for_year


def get_play_in_teams(seeds):
    """
    Returns a list of team numbers representing teams that played in play-in games

    Inputs:
        seeds (pandas DataFrame): seeds for a specific year

    Returns:
        play_in_teams (list): list of play in teams for that year
    """

    play_in_teams = []
    for index, row in seeds.iterrows():
        if 'a' in row['Seed'] or 'b' in row['Seed']:
            play_in_teams.append(row['Team'])
    return play_in_teams

def get_play_in_losers(play_in_teams, results_for_year):
    """
    Returns a list containing losing teams IDs

    Inputs:
        play_in_teams (list): list of play in teams for that year
        results_for_year (Pandas DataFrame): tourney detailed results for specified year

    Returns:
        losers (list): losing play-in team IDs
    """

    losers = []
    for index, row in results_for_year.iterrows():
        if row['Wteam'] in play_in_teams and row['Lteam'] in play_in_teams:
            losers.append(row['Lteam'])
    return losers

def filter_and_reformat_based_on_play_in_games(seeds_for_year, losers):
    """
    1. Filters out losing play-in game teams
    2. Reformats play in game winners to have a seed without a letter
        - e.g. changes seed 'W16a' to 'W16'

    Inputs:
        seeds_for_year (pandas DataFrame): seeds for specific year
    Returns:
        new_seeds_for_year (pandas DataFrame): new seeds for year reformatted

    """

    new_seeds_for_year = seeds_for_year[seeds_for_year['Team'].isin(losers) == False]
    brack_copy = new_seeds_for_year.copy()
    for index, row in brack_copy.iterrows():
        if 'a' in row['Seed'] or 'b' in row['Seed']:
            new_seeds_for_year.set_value(index, 'Seed', new_seeds_for_year.loc[index]['Seed'][0:3])
    return new_seeds_for_year

def get_teams(seeds_for_year):
    """
    Returns a list of all the team IDs for that year

    Inputs:
        seeds_for_year (pandas DataFrame): seeds for specific year

    Returns:
        teams (list): list of all the team IDs for that year
    """

    teams = [row['Team'] for index, row in seeds_for_year.iterrows()]
    return teams

def get_team_to_seed(seeds_for_year):
    """
    Inputs:
        seeds_for_year (pandas DataFrame): seeds for specific year

    Returns:
        team_to_seed (dict): dictionary mapping each team ID to its seed (int) in the bracket
    """

    team_to_seed = {}
    for index, row in seeds_for_year.iterrows():
        team_to_seed[row['Team']] = int(row['Seed'][1:3])
    return team_to_seed

def get_team_to_region(seeds_for_year):
    """
    Inputs:
        seeds_for_year (pandas DataFrame): seeds for specific year

    Returns:
        team_to_region (dict): dictionary mapping each team ID to its region (str), which could be 'W', 'X', 'Y', or 'Z'
    """

    team_to_region = {}
    for index, row in seeds_for_year.iterrows():
        team_to_region[row['Team']] = row['Seed'][0]
    return team_to_region


class Bracket:
    """
    Bracket class is a recursive data type that allows the user to represent anything from a 2 team bracket to an entire 64 team bracket
    """

    def __init__(self, teams, team_to_region, team_to_seed, round=6):
        self.teams = teams
        self.team_to_region = team_to_region
        self.team_to_seed = team_to_seed
        self.round = round
        self.matchup = None, None

        top_teams = []
        bottom_teams = []

        if round == 6:
            top_teams = [i for i in teams if self.team_to_region[i] in ['W', 'X']]
            bottom_teams = [i for i in teams if self.team_to_region[i] in ['Y', 'Z']]

        elif round == 5:
            top_teams = [i for i in teams if self.team_to_region[i] in ['W', 'Z']]
            bottom_teams = [i for i in teams if self.team_to_region[i] in ['X', 'Y']]

        elif round == 4:
            top_teams = [i for i in teams if self.team_to_seed[i] in [1, 16, 8, 9, 5, 12, 4, 13]]
            bottom_teams = [i for i in teams if self.team_to_seed[i] in [6, 11, 3, 14, 7, 10, 2, 15]]

        elif round == 3:
            top_teams = [i for i in teams if team_to_seed[i] in [1, 16, 8, 9, 6, 11, 3, 14]]
            bottom_teams = [i for i in teams if team_to_seed[i] in [5, 12, 4, 13, 7, 10, 2, 15]]

        elif round == 2:
            top_teams = [i for i in teams if team_to_seed[i] in [1, 16, 5, 12, 6, 11, 7, 10]]
            bottom_teams = [i for i in teams if team_to_seed[i] in [8, 9, 4, 13, 3, 14, 2, 15]]

        if len(top_teams) != 0:
            self.matchup = Bracket(top_teams, self.team_to_region, self.team_to_seed, round-1), Bracket(bottom_teams, self.team_to_region, self.team_to_seed, round-1)

        else:
            self.matchup = teams[0], teams[1]

    def get_possible_matchups(self, team, goal_round):
        """
        Inputs:
            team (int): team ID of interest
            goal_round (int): round of interest

        Returns:
            matchups (list): list of team IDs that input team could play in goal_round
        """

        top, bottom = self.matchup[0], self.matchup[1]
        if isinstance(top, Bracket):
            tops = top.teams
        else:
            tops = [top]

        if isinstance(bottom, Bracket):
            bottoms = bottom.teams
        else:
            bottoms = [bottom]

        matchups = []

        # check if team is on top or bottom side, and if we're at the round we want
        if team in tops and goal_round < self.round:
            matchups = top.get_possible_matchups(team, goal_round)
        elif team in bottoms and goal_round < self.round:
            matchups = bottom.get_possible_matchups(team, goal_round)
        elif team in tops:
            matchups = [opponent for opponent in bottoms]
        elif team in bottoms:
            matchups = [opponent for opponent in tops]

        return matchups


def calculate_q_values(teams, seeds_for_year, prob_matrix):
    """
    Inputs:
        teams (list): list of team ids for specific year
        seeds_for_year (pandas DataFrame): seeds for specific year
        prob_matrix (pandas DataFrame):

    Returns:
        team_win_in_round_prob (dictionary): dictionary that maps each team to another dictionary, where each key in the inner
        dictionary is the round number, mapped to the probability that the team wins in that round
    """

    team_to_region = get_team_to_region(seeds_for_year)
    team_to_seed = get_team_to_seed(seeds_for_year)
    team_win_in_round_prob = {t: {} for t in teams}

    bracket = Bracket(teams, team_to_region, team_to_seed)
    for r in range(1, 7):
        for t in teams:
            opponents = bracket.get_possible_matchups(t, r)
            if r == 1:
                prob = get_prob_win(t, opponents[0], prob_matrix)
                team_win_in_round_prob[t][r] = prob
            else:
                prob_advanced = team_win_in_round_prob[t][r-1]
                opponents_prob = sum(team_win_in_round_prob[o][r-1]*get_prob_win(t, o, prob_matrix) for o in opponents)
                team_win_in_round_prob[t][r] = prob_advanced * opponents_prob
    return team_win_in_round_prob


def create_q_matrix(prob_file, seeds_file, tourney_results_file, year, save_df=False):
    """
    Inputs:
        prob_file (csv): matrix of probabilities that y axis team beats x axis team
        seeds_file (csv): tourney seeds dating back to 1985
        tourney_results_file (csv): detailed tournament results data dating back to 2003
        year (int): year of interest
        save_df (bool): optional argument determining whether to save df to a csv titled new_q_matrix_year

    Returns:
        df (Pandas DataFrame): q matrix for specified year
    """

    probs = create_and_reformat_probs(prob_file)
    seeds = get_tournament_seeds_for_specific_year(seeds_file, year)
    results_for_year = tournament_results_for_specific_year(tourney_results_file, year)
    play_in_teams = get_play_in_teams(seeds)
    losers = get_play_in_losers(play_in_teams, results_for_year)

    # filter seeds to get rid of losers
    seeds = filter_and_reformat_based_on_play_in_games(seeds, losers)
    teams = get_teams(seeds)

    # map each team to its seed
    seed_to_team = {}
    for index, row in seeds.iterrows():
        seed_to_team[row['Seed']] = row['Team']

    regions = ['W', 'X', 'Y', 'Z']
    r1_matchups = ['01', '16', '08', '09', '05', '12', '04', '13', '06', '11', '03', '14', '07', '10', '02', '15']
    ind = []  # list of team ids in correct order for matrix optimization
    for r in regions:
        for m in r1_matchups:
            seed = r + m
            ind.append(seed_to_team[seed])
    q_vals = calculate_q_values(teams, seeds, probs)
    df_list = []
    for team in ind:
        df_list.append(q_vals[team])
    df = pd.DataFrame(df_list, columns=[1, 2, 3, 4, 5, 6], index=np.array(ind))

    if save_df:
        df.to_csv("new_q_matrix_" + str(year) + ".csv")
    return df


# uncomment line below to generate q matrix for 2011
# print(create_q_matrix("Probabilities2011.csv", "TourneySeeds.csv", "AllTourneyDetailedResults.csv", 2011))

# uncomment line below to generate q matrix for 2012
# print(create_q_matrix("Probabilities2012.csv", "TourneySeeds.csv", "AllTourneyDetailedResults.csv", 2012))

# uncomment line below to generate q matrix for 2013
# print(create_q_matrix("Probabilities2013.csv", "TourneySeeds.csv", "AllTourneyDetailedResults.csv", 2013))
