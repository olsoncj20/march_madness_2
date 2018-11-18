from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical
from keras.regularizers import l2, activity_l2
from keras.optimizers import Adam
import numpy as np
import numbers
import pandas as pd
from sklearn.preprocessing import normalize


def calc_season_stats(season_results, teams, seeds, years):
    """
    Aggregates game data into season averages and aggregates it
    :param season_results: season stats across many years
    :param teams: teams to inspect
    :param seeds: NCAA Tournament seed data
    :param years: years to inspect
    :return: season statistics for each team in teams and each year
    """

    year_team_data = []
    # We iterate through each year and consider each season
    # independently (i.e. we assume that a teams performance
    # in the past year is not indicative of its performance
    # in the next)
    for year in years:
        print "Calculating Season Statistics for", year
        teams_data = []
        # Games from the year being explored
        year_data = season_results[season_results["Season"] == year]

        # Seeds from the year being explored
        year_seeds = seeds[seeds["Season"] == year]

        # We now are concerned with teams that have a seed
        # (made the tournament)
        for team in year_seeds["Team"]:
            # Aggregate team id and its seed into one vector
            team_year_vector = teams[teams["Team_Id"] == team]
            team_year_vector = team_year_vector.set_index("Team_Id")\
                .join(year_seeds.set_index("Team"))

            team_id = team_year_vector.index.values[0]

            # Convert seed to an integer
            # (we don't care about the region)
            seed = team_year_vector["Seed"].values[0]
            team_year_vector["Seed"] = int(seed[1:3])

            # We now look at season averages by
            # aggregating data from games won and lost
            wins = year_data[year_data["Wteam"] == team_id]
            losses = year_data[year_data["Lteam"] == team_id]
            games = pd.concat([wins, losses])

            num_wins = wins.shape[0]  # Number of games won
            num_losses = losses.shape[0]  # Number of games lost
            total_games = float(num_losses + num_wins)

            team_year_vector["Wins"] = num_wins
            team_year_vector["Losses"] = num_losses

            def get_total_stat(tag1, tag2, name):
                stat = (wins[tag1].sum() + losses[tag2].sum()) \
                       / total_games
                team_year_vector[name] = stat
                return stat

            # PPG (Points Per Game)
            ppg = get_total_stat("Wscore", "Lscore", "ppg")

            # FGM (Field Goals Made)
            fgm = get_total_stat("Wfgm", "Lfgm", "fgm")

            # FGA (Field Goals Attempted)
            fga = get_total_stat("Wfga", "Lfga", "fga")

            # FGM3 (3-Point Field Goals Made)
            fgm_three = get_total_stat("Wfgm3", "Lfgm3", "fgm3")

            # FGA3 (3-Point Field Goals Attempted)
            fga_three = get_total_stat("Wfga3", "Lfga3", "fga3")

            # FTM (Free Throws Made)
            ftm = get_total_stat("Wftm", "Lftm", "ftm")

            # FTA (Free Throws Attempted)
            fta = get_total_stat("Wfta", "Lfta", "fta")

            # OR (Offensive Rebounds)
            off_reb = get_total_stat("Wor", "Lor", "or")

            # DR (Defensive Rebounds)
            def_reb = get_total_stat("Wdr", "Ldr", "dr")

            # APG (Assists Per Game)
            apg = get_total_stat("Wast", "Last", "ast")

            # TPG (Turnovers Per Game)
            tpg = get_total_stat("Wto", "Lto", "to")

            # FPG (Fouls Per Game)
            fpg = get_total_stat("Wpf", "Lpf", "foul")

            # SPG (Steals Per Game)
            spg = get_total_stat("Wstl", "Lstl", "stl")

            # BPG (Blocks Per Game)
            bpg = get_total_stat("Wblk", "Lblk", "blk")

            # EFG (Effective Field Goal Percentage)
            efg = (fgm + .5 * fgm_three) / fga
            team_year_vector["efg"] = efg

            # PPGA (Points Per Game Against)
            ppga = get_total_stat("Lscore", "Wscore", "ppga")

            # FGMA (Field Goals Made Against)
            fgma = get_total_stat("Lfgm", "Wfgm", "fgma")

            # FGAA (Field Goals Attempted Against)
            fgaa = get_total_stat("Lfga", "Wfga", "fgaa")

            # FGMA3 (3-Point Field Goals Made Against)
            fgma_three = get_total_stat("Lfgm3", "Wfgm3", "fgma3")

            # FGAA3 (3-Point Field Goals Attempted Against)
            fgaa_three = get_total_stat("Lfga3", "Wfga3", "fgaa3")

            # FTMA (Free Throws Made Against)
            ftma = get_total_stat("Lftm", "Wftm", "ftma")

            # FTAA (Free Throws Attempted Against)
            ftaa = get_total_stat("Lfta", "Wfta", "ftaa")

            # ORA (Offensive Rebounds Against)
            off_reb_a = get_total_stat("Lor", "Wor", "ora")

            # DRA (Defensive Rebounds Against)
            def_reb_a = get_total_stat("Ldr", "Wdr", "dra")

            # APGA (Assists Per Game Against)
            apga = get_total_stat("Last", "Wast", "asta")

            # TPGA (Turnovers Per Game Against)
            tpga = get_total_stat("Lto", "Wto", "toa")

            # FPGA (Fouls Per Game Against)
            fpga = get_total_stat("Lpf", "Wpf", "foula")

            # SPGA (Steals Per Game Against)
            spga = get_total_stat("Lstl", "Wstl", "stla")

            # BPGA (Blocks Per Game Against)
            bpga = get_total_stat("Lblk", "Wblk", "blka")

            # EFGA (Effective Field Goal Percentage Against)
            efga = (fgma + .5 * fgma_three) / fgaa
            team_year_vector["efga"] = efga

            # OE (Offensive Efficiency)
            possessions = .96 * (fga - off_reb + tpg + (.44 * fta))
            oe = ppg * 100 / possessions
            team_year_vector["oe"] = oe

            # DE (Defensive Efficiency)
            possessions_against = .96 * (fgaa - off_reb_a + tpga
                                         + (.44 * ftaa))
            de = ppg * 100 / possessions_against
            team_year_vector["de"] = de

            # Create unique identifier for team
            # to quickly search in the future
            team_year_vector["ID"] = team_year_vector["Team_Name"] \
                                     + str(year)

            teams_data.append(team_year_vector)
        teams_data = pd.concat(teams_data)
        year_team_data.append(teams_data)
    year_team_data = pd.concat(year_team_data)
    return year_team_data


def filter_nonnumerics(array):
    """
    Filters nonnumerical values from a 1xN array
    :param array:  numpy array
    :return: numpy array trimmed
    """
    new_array = list(list(array)[0])
    for i in new_array:
        if not isinstance(i, numbers.Number):
            new_array.remove(i)
    new_array.pop(0)  # We don't need the year as a feature
    return np.array([new_array])


def format_training_data(statistics, results, years):
    """
    Creates labels with "1" denoting the winner
    and "0" denoting the loser. Also creates an
    associated feature vector for the winning
    and losing teams in the game.
    :param statistics: data used to create feature vector
    :param results: historic game data
    :param years: years to explore
    :return: tuple of game vectors and labels
    """
    training_data = []
    labels = []
    for year in years:
        result = results[results["Season"] == year]
        statistic = statistics[statistics["Season"] == year]
        matchups = result[["Wteam", "Lteam"]]
        for _, matchup in matchups.iterrows():
            winner_id = matchup["Wteam"]
            loser_id = matchup["Lteam"]
            winner_stats = statistic.loc[winner_id].to_frame()\
                .transpose().values
            loser_stats = statistic.loc[loser_id].to_frame()\
                .transpose().values
            # Now that we have the statistics, we want to
            # create a feature vector. Therefore, we trim
            # all nonnumerical fields
            winner_vector = filter_nonnumerics(winner_stats)
            loser_vector = filter_nonnumerics(loser_stats)
            winning_vector = np.append(winner_vector, loser_vector)
            losing_vector = np.append(loser_vector, winner_vector)
            training_data.append(winning_vector)
            training_data.append(losing_vector)
            # winning_vector is a vector with winning team first
            labels.append(1)
            # losing_vector is a vector with losing team first
            labels.append(0)
    return training_data, labels


def create_feature_vector(stats1, stats2):
    """
    Creates a feature vector for a matchup
    :param stats1: season stats of one team
    :param stats2: season stats of another team
    :return: feature vector of the teams
    """
    values1 = stats1.to_frame().transpose().values
    values2 = stats2.to_frame().transpose().values
    vector1 = filter_nonnumerics(values1)
    vector2 = filter_nonnumerics(values2)
    vector = np.append(vector1, vector2)
    return np.array([vector])


def accuracy(y_hat, y):
    """
    Determines accuracy of predictions
    :param y_hat: predictions
    :param y: realized data
    :return: accuracy of model
    """
    correct_predictions = sum([1 if y_hat[i] == y[i]
                               else 0 for i in range(len(y))])
    accuracy = float(correct_predictions) / len(y)
    return 100 * accuracy


def predict_matchups(years, seeds, model, stats):
    """
    Creates a CSV with predictions for all matchups in the tournament.
    :param years: years to predict matchups for
    :param seeds: seeds for the tournament that year
    :param model: neural net
    :param stats: season statistics
    """

    for year in years:
        year_stats = stats[stats["Season"] == year]
        year_seeds = seeds[seeds["Season"] == year]
        year_teams = sorted(year_seeds["Team"].values)
        df = pd.DataFrame(index=year_teams, columns=year_teams).fillna(0.)
        for i in range(len(year_teams)):
            for j in range(i, len(year_teams)):
                if i != j:
                    team1_stats = year_stats.loc[year_teams[i]]
                    team2_stats = year_stats.loc[year_teams[j]]
                    vector = normalize(create_feature_vector(
                        team1_stats, team2_stats)).reshape(1, -1)
                    prediction = model.predict(
                        vector, verbose=0)
                    prediction = prediction[0,1]
                    df.loc[year_teams[i], year_teams[j]] = prediction
                    df.loc[year_teams[j], year_teams[i]] = 1-prediction
        with open("Predictions" + str(year) + ".csv", "w") as pred:
            df.to_csv(pred, index=False)


def define_model(input, p):
    """
    Define the Neural Network we are going to use for predictions
    :param input: feature vectors
    :return: the neural net object
    """
    a = 0.1
    model = Sequential([
        # We randomly drop 20% of the input
        # nodes to avoid over fitting
        Dropout(p, input_shape=(len(input[0]),)),

        # Normalize inputs
        BatchNormalization(epsilon=0.001, mode=0,
                           axis=-1, momentum=0.99,
                           weights=None, beta_init='zero',
                           gamma_init='one',
                           gamma_regularizer=None,
                           beta_regularizer=None),

        # Creates the model structure
        Dense(512, W_regularizer=l2(0.01),
              activity_regularizer=activity_l2(0.01), ),
        LeakyReLU(alpha=a),
        Dense(256, W_regularizer=l2(0.01),
              activity_regularizer=activity_l2(0.01), ),
        LeakyReLU(alpha=a),
        Dense(128, W_regularizer=l2(0.01),
              activity_regularizer=activity_l2(0.01), ),
        LeakyReLU(alpha=a),
        Dense(64),
        LeakyReLU(alpha=a),
        Dense(32),
        LeakyReLU(alpha=a),
        Dense(32),
        LeakyReLU(alpha=a),
        Dense(32),
        LeakyReLU(alpha=a),
        Dense(2),
        Activation("softmax"),
    ])

    # Train the model
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9,
                                 beta_2=0.999, epsilon=1e-08,
                                 decay=0.0),
                  metrics=['accuracy'])
    return model


def predict_march_madness_matchups(new_model=False):
    """
    Predicts March Madness matchups for
    2011, 2012, and 2013 tournaments
    """
    file_names = ["RegularSeasonDetailedResults.csv",
                  "teams.csv", "TourneyDetailedResults.csv",
                  "TourneySeeds.csv"]
    data_frames = []
    for file_name in file_names:
        data_frame = pd.read_csv(file_name)
        data_frames.append(data_frame)

    reg_season_results = data_frames[0]
    teams = data_frames[1]
    tourney_results = data_frames[2]
    seeds = data_frames[3]

    # Determine which years to use for training
    # the model (training data) and which years
    # to use as validation data
    training_years = [2003, 2006, 2007,
                      2010, 2015, 2016]
    validation_years = [2005, 2008, 2017]
    test_years = [2004, 2009, 2014]

    # Get season data for teams which we will
    # use to create feature vectors
    training_statistics = calc_season_stats(
        reg_season_results, teams, seeds, training_years)
    validation_statistics = calc_season_stats(
        reg_season_results, teams, seeds, validation_years)
    test_statistics = calc_season_stats(
        reg_season_results, teams, seeds, test_years)

    # Create feature vectors and labels
    data, labels = format_training_data(
        training_statistics, tourney_results, training_years)
    validation_data, validation_labels = format_training_data(
        validation_statistics, tourney_results, validation_years)
    test_data, test_labels = format_training_data(
        test_statistics, tourney_results, test_years)

    # Normalize feature vectors
    data = normalize(data)
    validation_data = normalize(validation_data)
    test_data = normalize(test_data)

    # Modify labels to fit model's expectations
    labels = to_categorical(labels)
    # Instantiate the model
    model = define_model(data, .2)

    # Train the model with the training data
    if new_model:
        model.fit(data, labels,
                  nb_epoch=1000, batch_size=64)
        model.save('model.h5')
    else:
        model = load_model('model.h5')

    # Test the model on the validation data
    model_output = model.predict_classes(
        validation_data, batch_size=64)
    print "\nValidation Accuracy:"
    print accuracy(model_output, validation_labels)

    model_output = model.predict_classes(
        test_data, batch_size=64)
    print "\nTest Accuracy:"
    print accuracy(model_output, test_labels)

    # Calculate the season data for
    # out prediction years (test data)
    prediction_years = [2011, 2012, 2013]
    prediction_statistics = calc_season_stats(
        reg_season_results, teams, seeds, prediction_years)

    # Output every possible match up and
    # its outcome using our model to CSV
    predict_matchups(
        prediction_years, seeds, model, prediction_statistics)


if __name__ == "__main__":
    predict_march_madness_matchups()
