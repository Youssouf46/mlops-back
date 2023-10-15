#import the needed librairies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime as dt

"""
This fonction serves to clean the data for trainining and it will split it into train and test needed for further steps
"""




def create_db(df):
    def result_finder(home, away):
        if home > away:
          return pd.Series([0, 3, 0])
        if home < away:
          return pd.Series([1, 0, 3])
        else:
          return pd.Series([2, 1, 1])

    results = df.apply(lambda x: result_finder(x["home_score"], x["away_score"]), axis=1)

    df[["result", "home_team_points", "away_team_points"]] = results
    df["rank_dif"] = df["rank_home"] - df["rank_away"]
    df["sg"] = df["home_score"] - df["away_score"]
    df["points_home_by_rank"] = df["home_team_points"]/df["rank_away"]
    df["points_away_by_rank"] = df["away_team_points"]/df["rank_home"]
    
    home_team = df[["date", "home_team", "home_score", "away_score", "rank_home", "rank_away","rank_change_home", "total_points_home", "result", "rank_dif", "points_home_by_rank", "home_team_points"]]

    away_team = df[["date", "away_team", "away_score", "home_score", "rank_away", "rank_home","rank_change_away", "total_points_away", "result", "rank_dif", "points_away_by_rank", "away_team_points"]]
    
    home_team.columns = [h.replace("home_", "").replace("_home", "").replace("away_", "suf_").replace("_away", "_suf") for h in home_team.columns]

    away_team.columns = [a.replace("away_", "").replace("_away", "").replace("home_", "suf_").replace("_home", "_suf") for a in away_team.columns]
    
    team_stats = home_team.append(away_team)#.sort_values("date")

    team_stats_raw = team_stats.copy()



    # Créez une liste vide pour stocker les statistiques calculées
    stats_val = []

    # Parcourez chaque ligne du DataFrame team_stats
    for index, row in team_stats.iterrows():
      # Récupérez le nom de l'équipe et la date du match
      team = row["team"]
      date = row["date"]

     # Sélectionnez tous les matchs passés de l'équipe jusqu'à la date actuelle, triés par date décroissante
      past_games = team_stats.loc[(team_stats["team"] == team) & (team_stats["date"] < date)].sort_values(by=['date'], ascending=False)

    # Sélectionnez les 5 derniers matchs de l'équipe
      last5 = past_games.head(5)

    # Calculez la moyenne des buts marqués par l'équipe pour tous les matchs passés et les 5 derniers matchs
      goals = past_games["score"].mean()
      goals_l5 = last5["score"].mean()

    # Calculez la moyenne des buts encaissés par l'équipe pour tous les matchs passés et les 5 derniers matchs
      goals_suf = past_games["suf_score"].mean()
      goals_suf_l5 = last5["suf_score"].mean()

    # Calculez la moyenne du classement FIFA de l'adversaire pour tous les matchs passés et les 5 derniers matchs
      rank = past_games["rank_suf"].mean()
      rank_l5 = last5["rank_suf"].mean()

    # Calculez les points FIFA gagnés par l'équipe pour tous les matchs passés et les 5 derniers matchs
      if len(last5) > 0:
        points = past_games["total_points"].values[0] - past_games["total_points"].values[-1]  # quantité de points gagnés
        points_l5 = last5["total_points"].values[0] - last5["total_points"].values[-1]
      else:
        points = 0
        points_l5 = 0

    # Calculez la moyenne des points de jeu de l'équipe pour tous les matchs passés et les 5 derniers matchs
      gp = past_games["team_points"].mean()
      gp_l5 = last5["team_points"].mean()

    # Calculez la moyenne des points de jeu en fonction du classement de l'adversaire pour tous les matchs passés et les 5 derniers matchs
      gp_rank = past_games["points_by_rank"].mean()
      gp_rank_l5 = last5["points_by_rank"].mean()

    # Ajoutez les statistiques calculées sous forme de liste à la liste stats_val
      stats_val.append([goals, goals_l5, goals_suf, goals_suf_l5, rank, rank_l5, points, points_l5, gp, gp_l5, gp_rank, gp_rank_l5])




    # Créez une liste des noms de colonnes pour les statistiques
    stats_cols = ["goals_mean", "goals_mean_l5", "goals_suf_mean", "goals_suf_mean_l5", "rank_mean", "rank_mean_l5", "points_mean", "points_mean_l5", "game_points_mean", "game_points_mean_l5", "game_points_rank_mean", "game_points_rank_mean_l5"]

    # Créez un DataFrame à partir de la liste de statistiques et des valeurs de statistiques calculées
    stats_df = pd.DataFrame(stats_val, columns=stats_cols)

   # Réinitialisez l'index du DataFrame team_stats et concaténez-le avec le DataFrame des statistiques
    full_df = pd.concat([team_stats.reset_index(drop=True), stats_df], axis=1, ignore_index=False)


    home_team_stats = full_df.iloc[:int(full_df.shape[0]/2),:]
    away_team_stats = full_df.iloc[int(full_df.shape[0]/2):,:] 

    home_team_stats = home_team_stats[home_team_stats.columns[-12:]]
    away_team_stats = away_team_stats[away_team_stats.columns[-12:]]

    home_team_stats.columns = ['home_'+str(col) for col in home_team_stats.columns]
    away_team_stats.columns = ['away_'+str(col) for col in away_team_stats.columns]

    match_stats = pd.concat([home_team_stats, away_team_stats.reset_index(drop=True)], axis=1, ignore_index=False)

    full_df = pd.concat([df, match_stats.reset_index(drop=True)], axis=1, ignore_index=False)

    def find_friendly(x):
        if x == "Friendly":
          return 1
        else: 
         return 0

    full_df["is_friendly"] = full_df["tournament"].apply(lambda x: find_friendly(x))

    full_df = pd.get_dummies(full_df, columns=["is_friendly"])

    base_df = full_df[["date", "home_team", "away_team", "rank_home", "rank_away","home_score", "away_score","result", "rank_dif", "rank_change_home", "rank_change_away", 'home_goals_mean',
       'home_goals_mean_l5', 'home_goals_suf_mean', 'home_goals_suf_mean_l5',
       'home_rank_mean', 'home_rank_mean_l5', 'home_points_mean',
       'home_points_mean_l5', 'away_goals_mean', 'away_goals_mean_l5',
       'away_goals_suf_mean', 'away_goals_suf_mean_l5', 'away_rank_mean',
       'away_rank_mean_l5', 'away_points_mean', 'away_points_mean_l5','home_game_points_mean', 'home_game_points_mean_l5',
       'home_game_points_rank_mean', 'home_game_points_rank_mean_l5','away_game_points_mean',
       'away_game_points_mean_l5', 'away_game_points_rank_mean',
       'away_game_points_rank_mean_l5',
       'is_friendly_0', 'is_friendly_1']]
    
    base_df_no_fg = base_df.dropna()

    df = base_df_no_fg
    
    
    def no_draw(x):
       if x == 2:
        return 1
       else:
        return x

    df["target"] = df["result"].apply(lambda x: no_draw(x))

    # Liste des colonnes que nous voulons inclure dans la base de données résultante.
    columns = ["home_team", "away_team", "target", "rank_dif", "home_goals_mean", "home_rank_mean", "away_goals_mean", "away_rank_mean", "home_rank_mean_l5", "away_rank_mean_l5", "home_goals_suf_mean", "away_goals_suf_mean", "home_goals_mean_l5", "away_goals_mean_l5", "home_goals_suf_mean_l5", "away_goals_suf_mean_l5", "home_game_points_rank_mean", "home_game_points_rank_mean_l5", "away_game_points_rank_mean", "away_game_points_rank_mean_l5","is_friendly_0", "is_friendly_1"]

    # Crée un DataFrame de base en sélectionnant uniquement les colonnes spécifiées.
    base = df.loc[:, columns]

    # Calcule la différence de buts entre l'équipe à domicile et l'équipe à l'extérieur.
    base.loc[:, "goals_dif"] = base["home_goals_mean"] - base["away_goals_mean"]
    base.loc[:, "goals_dif_l5"] = base["home_goals_mean_l5"] - base["away_goals_mean_l5"]

    # Calcule la différence de buts encaissés entre l'équipe à domicile et l'équipe à l'extérieur.
    base.loc[:, "goals_suf_dif"] = base["home_goals_suf_mean"] - base["away_goals_suf_mean"]
    base.loc[:, "goals_suf_dif_l5"] = base["home_goals_suf_mean_l5"] - base["away_goals_suf_mean_l5"]

    # Calcule la différence de buts par rapport au classement FIFA.
    base.loc[:, "goals_per_ranking_dif"] = (base["home_goals_mean"] / base["home_rank_mean"]) - (base["away_goals_mean"] / base["away_rank_mean"])

    # Calcule la différence de classement entre l'équipe à domicile et l'équipe à l'extérieur.
    base.loc[:, "dif_rank_agst"] = base["home_rank_mean"] - base["away_rank_mean"]
    base.loc[:, "dif_rank_agst_l5"] = base["home_rank_mean_l5"] - base["away_rank_mean_l5"]

    # Calcule la différence de points FIFA entre l'équipe à domicile et l'équipe à l'extérieur.
    base.loc[:, "dif_points_rank"] = base["home_game_points_rank_mean"] - base["away_game_points_rank_mean"]
    base.loc[:, "dif_points_rank_l5"] = base["home_game_points_rank_mean_l5"] - base["away_game_points_rank_mean_l5"]

    # Sélectionne les colonnes finales à inclure dans le modèle.
    model_df = base[["home_team", "away_team", "target", "rank_dif", "goals_dif", "goals_dif_l5", "goals_suf_dif", "goals_suf_dif_l5", "goals_per_ranking_dif", "dif_rank_agst", "dif_rank_agst_l5", "dif_points_rank", "dif_points_rank_l5", "is_friendly_0", "is_friendly_1"]]
   
    X = model_df.iloc[:, 3:]
    y = model_df[["target"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)  # Split data 80/20
    #scaler = MinMaxScaler()  # Normalize train & test features
    #X_train[num_features] = scaler.fit_transform(X_train[num_features])
    #X_test[num_features] = scaler.transform(X_test[num_features])
    return X_train,X_test,y_train,y_test