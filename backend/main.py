from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import sklearn
from fastapi import FastAPI, File, UploadFile
import uvicorn
import sys  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
from models.transaction_info.TransactionModel import TransactionModel
import os
from src.clean_data_csv import clean_data
from src.clean_data_csv import find_stats
from src.clean_data_csv import find_features
from src.clean_data_json import clean_data_json
from operator import itemgetter
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout



os.environ['MLFLOW_TRACKING_USERNAME']= "Youssouf"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "2685You@"

#setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/Youssouf/mlops_project.mlflow') #your mlfow tracking uri

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#let's call the model from the model registry ( in production stage)
all_experiments = [exp.experiment_id for exp in mlflow.search_experiments()]
df_mlflow = mlflow.search_runs(experiment_ids=all_experiments,filter_string="metrics.F1_score_test <1")
run_id = df_mlflow.loc[df_mlflow['metrics.F1_score_test'].idxmax()]['run_id']

logged_model = f'runs:/{run_id}/ML_models'

#model = mlflow.pyfunc.load_model(logged_model)
model=mlflow.sklearn.load_model(logged_model)

@app.get("/")
def read_root():
    return {"Hello": "to foot app"}

# this endpoint receives data in the form of csv file (histotical transactions data)
"""@app.post("/predict/csv")
def return_predictions(file: UploadFile = File(...)):
    data = pd.read_csv(file.file)
    preprocessed_data = clean_data(data)
    predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()}
"""

# this endpoint receives data in the form of json (informations about one transaction)
''' @app.post("/predict")
def predict(data : TransactionModel):
    received = data.dict()
    df =  pd.DataFrame(received,index=[0])
    preprocessed_data = clean_data_json(df)
    predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()}
     '''
''' 
@app.post("/predict/json")
def predict(data: TransactionModel):
    received = data.dict()
    print(received) 
    df = pd.DataFrame([received])  # Créez un DataFrame à partir des données reçues
    preprocessed_data = clean_data_json(df)
    # Assurez-vous d'avoir le modèle `model` approprié pour faire des prédictions
    predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()} '''






@app.post("/predict/csv")
def return_predictions(file_predict: UploadFile = File(...),data_file: UploadFile = File(...)):
    data_predict = pd.read_csv(file_predict.file)
    print('hi')
    data = pd.read_csv(data_file.file)
    #preprocessed_data = clean_data(data)
    table={}
    for letter in 'ABCDEF':
        table[letter]=[]
        for name in data_predict[letter]:
            table[letter].append([name, 0, []])
    ''' 
    table={'A': [['Ivory Coast', 0, []],['Guinea-Bissau', 0, []],['Nigeria', 0, []],['Equatorial Guinea', 0, []]],
     'B': [['Egypt', 0, []],['Mozambique', 0, []],['Ghana', 0, []],['Cape Verde', 0, []]],
     'C': [['Senegal', 0, []],['Gambia', 0, []],['Cameroon', 0, []],['Guinea', 0, []]],
     'D': [['Mauritania', 0, []],['Burkina Faso', 0, []],['Algeria', 0, []],['Angola', 0, []]],
     'E': [['Tunisia', 0, []],['Namibia', 0, []],['Mali', 0, []],['South Africa', 0, []]],
     'F': [['Congo', 0, []],['Zambia', 0, []],['Morocco', 0, []],['Tanzania', 0, []]]
      } '''
    from itertools import combinations
    matches=[]
    for letter in 'ABCDEF':
        for match in list(combinations(set(data_predict[letter]), 2)):
            matches.append((letter,)+match)
           

    ''' matches = [('A', 'Ivory Coast', 'Guinea-Bissau'),('A', 'Nigeria', 'Equatorial Guinea'),('A', 'Equatorial Guinea', 'Guinea-Bissau'),('A', 'Nigeria', 'Ivory Coast'),('A', 'Guinea-Bissau', 'Nigeria'),('A', 'Ivory Coast', 'Equatorial Guinea'),
    ('B', 'Egypt', 'Mozambique'),('B', 'Ghana', 'Cape Verde'),('B', 'Cape Verde', 'Mozambique'),('B', 'Egypt', 'Ghana'),('B', 'Mozambique', 'Ghana'),('B', 'Cape Verde', 'Egypt'),
    ('C', 'Senegal', 'Gambia'),('C', 'Cameroon', 'Guinea'),('C', 'Senegal', 'Cameroon'),('C', 'Guinea', 'Gambia'),('C', 'Guinea', 'Senegal'),('C', 'Gambia', 'Cameroon'),
    ('D', 'Algeria', 'Angola'),('D', 'Mauritania', 'Burkina Faso'),('D', 'Algeria', 'Burkina Faso'),('D', 'Mauritania', 'Angola'),('D', 'Angola', 'Burkina Faso'),('D', 'Mauritania', 'Algeria'),
    ('E', 'Tunisia', 'Namibia'),('E', 'Mali', 'South Africa'),('E', 'Tunisia', 'Mali'),('E', 'South Africa', 'Namibia'),('E', 'South Africa', 'Tunisia'),('E', 'Namibia', 'Mali'),
    ('F', 'Morocco', 'Tanzania'),('F', 'Congo', 'Zambia'),('F', 'Morocco', 'Congo'),('F', 'Zambia', 'Tanzania'),('F', 'Tanzania', 'Congo'),('F', 'Zambia', 'Morocco')]
 '''
     
    
    advanced_group = []
    last_group = ""

    for k in table.keys():
        for t in table[k]:
            t[1] = 0
            t[2] = []
    final_tables=[]
    for teams in matches:
        draw = False
        team_1 = find_stats(teams[1],data)
        team_2 = find_stats(teams[2],data)

        

        features_g1 = find_features(team_1, team_2)
        features_g2 = find_features(team_2, team_1)

        probs_g1 = model.predict_proba([features_g1])
        probs_g2 = model.predict_proba([features_g2])
        
        team_1_prob_g1 = probs_g1[0][0]
        team_1_prob_g2 = probs_g2[0][1]
        team_2_prob_g1 = probs_g1[0][1]
        team_2_prob_g2 = probs_g2[0][0]

        team_1_prob = (probs_g1[0][0] + probs_g2[0][1])/2
        team_2_prob = (probs_g2[0][0] + probs_g1[0][1])/2
        
        if ((team_1_prob_g1 > team_2_prob_g1) & (team_2_prob_g2 > team_1_prob_g2)) | ((team_1_prob_g1 < team_2_prob_g1) & (team_2_prob_g2 < team_1_prob_g2)):
            draw=True
            for i in table[teams[0]]:
                if i[0] == teams[1] or i[0] == teams[2]:
                    i[1] += 1
                    
        elif team_1_prob > team_2_prob:
            winner = teams[1]
            winner_proba = team_1_prob
            for i in table[teams[0]]:
                if i[0] == teams[1]:
                    i[1] += 3
                    
        elif team_2_prob > team_1_prob:  
            winner = teams[2]
            winner_proba = team_2_prob
            for i in table[teams[0]]:
                if i[0] == teams[2]:
                    i[1] += 3
        
        for i in table[teams[0]]: #adding criterio de desempate (probs por jogo)
                if i[0] == teams[1]:
                    i[2].append(team_1_prob)
                if i[0] == teams[2]:
                    i[2].append(team_2_prob)
        
        if last_group != teams[0]:
            if last_group != "":
                #print("\n")
                #print("Group %s advanced: "%(last_group))
                
                for i in table[last_group]: #adding crieterio de desempate
                    i[2] = np.mean(i[2])
                
                final_points = table[last_group]
                final_table = sorted(final_points, key=itemgetter(1, 2), reverse = True)
                advanced_group.append([final_table[0][0], final_table[1][0],final_table[2][::]])
                ''' for i in final_table: '''
                '''     print("%s -------- %d"%(i[0], i[1])) '''
                final_tables+=[final_table]
            #print("\n")
            #print("-"*10+" Starting Analysis for Group %s "%(teams[0])+"-"*10)
            
            
        ''' if draw == False: '''
        '''     print("Group %s - %s vs. %s: Winner %s with %.2f probability"%(t eams[0], teams[1], teams[2], winner, winner_proba))'''
        ''' else: '''
        '''    print("Group %s - %s vs. %s: Draw"%(teams[0], teams[1], teams[2])) '''
        last_group =  teams[0]

    #print("\n")
    #print("Group %s advanced: "%(last_group))

    for i in table[last_group]: #adding crieterio de desempate
        i[2] = np.mean(i[2])
                
    final_points = table[last_group]
    final_table = sorted(final_points, key=itemgetter(1, 2), reverse = True)
    advanced_group.append([final_table[0][0], final_table[1][0],final_table[2][::]])
    '''     for i in final_table:
        print("%s -------- %d"%(i[0], i[1])) '''
    final_tables+=[final_table]


    

    Advanced_Third_place={}
    selected_third_place=sorted([['ABCDEF'[i]]+advanced_group[i][2] for i in range(len(advanced_group))   ],key=itemgetter(2,3),reverse=True)[:4:]
    [Advanced_Third_place.update({item[0]:item[1]}) for item in selected_third_place ]
    #print(Advanced_Third_place.items())
    def choose(a,b,c):
        nomT=''
        if a in Advanced_Third_place.keys():
            nomT=Advanced_Third_place[a]
            Advanced_Third_place.pop(a)   
        elif b in Advanced_Third_place.keys():
            nomT=Advanced_Third_place[b]
            Advanced_Third_place.pop(b) 
        else:
            nomT=Advanced_Third_place[c]
            Advanced_Third_place.pop(c)        
        return nomT
    advanced=[[advanced_group[0][1], advanced_group[2][1]],
    [advanced_group[3][0], choose('B','E','F')],
    [advanced_group[1][0], choose('A','C','D')],
    [advanced_group[5][0], advanced_group[4][1]],
    [advanced_group[1][1], advanced_group[5][1]],
    [advanced_group[0][0], choose('C','D','E')],
    [advanced_group[4][0], advanced_group[3][1]],
    [advanced_group[2][0], choose('A','B','F')]
    ]
    
     
    playoffs = {"Round of 16": [], "Quarter-Final": [], "Semi-Final": [], "Final": []}


    for p in playoffs.keys():
        playoffs[p] = []

    actual_round = ""
    next_rounds = []
    labels = [] 
    for p in playoffs.keys():
    
        if p == "Round of 16":
            playoffs[p] = advanced
            
            for i in range(0, len(playoffs[p]), 1):
                game = playoffs[p][i]
                
                home = game[0]
                away = game[1]
                team_1 = find_stats(home,data)
                team_2 = find_stats(away,data)

                features_g1 = find_features(team_1, team_2)
                features_g2 = find_features(team_2, team_1)
                
                probs_g1 = model.predict_proba([features_g1])
                probs_g2 = model.predict_proba([features_g2])
                
                team_1_prob = (probs_g1[0][0] + probs_g2[0][1])/2
                team_2_prob = (probs_g2[0][0] + probs_g1[0][1])/2
                
                if actual_round != p:
                  #  print("-"*10)
                    print("Starting simulation of %s"%(p))
                    #print("-"*10)
                    #print("\n")
                
                if team_1_prob < team_2_prob:
                    #print("%s vs. %s: %s advances with prob %.2f"%(home, away, away, team_2_prob))
                    next_rounds.append(away)
                else:
                    #print("%s vs. %s: %s advances with prob %.2f"%(home, away, home, team_1_prob))
                    next_rounds.append(home)
                
                game.append([team_1_prob, team_2_prob])
                playoffs[p][i] = game
                actual_round = p
            
        else:
            playoffs[p] = [[next_rounds[c], next_rounds[c+1]] for c in range(0, len(next_rounds)-1, 1) if c%2 == 0]
            next_rounds = []
            for i in range(0, len(playoffs[p])):
                game = playoffs[p][i]
                home = game[0]
                away = game[1]
                team_1 = find_stats(home,data)
                team_2 = find_stats(away,data)
                
                features_g1 = find_features(team_1, team_2)
                features_g2 = find_features(team_2, team_1)
                
                probs_g1 = model.predict_proba([features_g1])
                probs_g2 = model.predict_proba([features_g2])
                
                team_1_prob = (probs_g1[0][0] + probs_g2[0][1])/2
                team_2_prob = (probs_g2[0][0] + probs_g1[0][1])/2
                
                if actual_round != p:
                    print("-"*10)
                    print("Starting simulation of %s"%(p))
                    print("-"*10)
                    print("\n")
                
                if team_1_prob < team_2_prob:
                    #print("%s vs. %s: %s advances with prob %.2f"%(home, away, away, team_2_prob))
                    next_rounds.append(away)
                else:
                    #print("%s vs. %s: %s advances with prob %.2f"%(home, away, home, team_1_prob))
                    next_rounds.append(home)
                game.append([team_1_prob, team_2_prob])
                playoffs[p][i] = game
                actual_round = p

                
                
        
        
        
          
        for game in playoffs[p]:
                label = f"{game[0]}({round(game[2][0], 2)}) \n {game[1]}({round(game[2][1], 2)})"
                label = f"{game[0]}({round(game[2][0], 2)}) \n {game[1]}({round(game[2][1], 2)})"
                labels.append(label)
    plt.figure(figsize=(15, 10))
    G = nx.balanced_tree(2, 3)
    
    labels_dict = {}
    labels_rev = list(reversed(labels))
    for l in range(len(list(G.nodes))):
            labels_dict[l] = labels_rev[l]
    pos = graphviz_layout(G, prog='twopi')
    labels_pos = {n: (k[0], k[1]-0.08*k[1]) for n,k in pos.items()}
    center  = pd.DataFrame(pos).mean(axis=1).mean()

    nx.draw(G, pos = pos, with_labels=False, node_color=range(15), edge_color="#bbf5bb", width=10, font_weight='bold',cmap=plt.cm.Greens, node_size=5000)
        
    nx.draw_networkx_labels(G, pos = labels_pos, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=.5, alpha=1),labels=labels_dict)
       
    texts = ["Round \n of 16", "Quarter \n Final", "Semi \n Final", "Final\n"]
    pos_y = pos[0][1] + 55
    for text in reversed(texts):
            pos_x = center
            pos_y -= 75 
            plt.text(pos_y, pos_x, text, fontsize = 18)
    plt.axis('equal')
    plt.savefig("graph_image.png")
    
    file_bytes = open("graph_image.png", "rb").read()
    
    
    # Côté serveur (FastAPI)
    import base64

    # Convertir l'image en base64
    image_base64 = base64.b64encode(file_bytes).decode("utf-8")

    # Renvoyer l'image en tant que chaîne base64
    return {"final_table": final_tables,"img": image_base64,}

    #return {file_bytes}

    """predictions = model.predict(preprocessed_data)
    return {"predictions": predictions.tolist()}"""


 





if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080)
