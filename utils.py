import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def getStates(df):
    states = list(df["State"].unique())
    return states

def getLabels(df):
    labels = list(df["Label"].unique())
    return labels

def getRecreations(df):
    temp = df["Recreation"].unique()
    a=[]
    for rec in temp:
        b= rec.split(" ")
        a.extend(b)
    a=list(set(a))
    return a

def getWeathers(df):
    weathers = list(df["Weather"].unique())
    return weathers

def createRecreation(df):
    df["Recreation"] = np.where(df['Religious Sentiment']== 'y',"Religious","Party")
    df["Recreation"]+=" "
    df["Recreation"]+=np.where(df['Bustle']== 'y',"Bustle","Relaxing")
    return df
def combine_features(row):
	try:
		return row['State']+" "+row["Label"]+" "+row["Weather"] +" "+row["Recreation"]
	except:
		print("Error:", row)
def getSpots(df,state,label,recreation,weather,budget):

    user={}
    spotindex = max(df["index"])+1
    user["index"] = [spotindex]
    user["State"] = [state]
    user["Label"] = [label]
    user["Recreation"] = [recreation]
    user["Weather"] = [weather]
    df2 = pd.DataFrame(user)
    df=df.append(df2)
    df=df.set_index("index")
    df["combined_features"] = df.apply(combine_features,axis=1)
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])
    cosine_sim = cosine_similarity(count_matrix) 
    
    similar_spots =  list(enumerate(cosine_sim[spotindex]))

    ## Step 7: Get a list of similar spots in descending order of similarity score
    sorted_similar_spots = sorted(similar_spots,key=lambda x:x[1],reverse=True)
    sorted_similar_spots = sorted_similar_spots[1:]
    ## Step 8: Print names of first 10 Tourist Spots
    spots = []
    i=0
    n=11
    for element in sorted_similar_spots:
		#print(df.iloc[[element[0]]])
        spot=[]
        spot.append(str(df.iloc[element[0]]["Tourist Spots"]))
        spot.append(str(df.iloc[element[0]]["City"]))
        spot.append(str(df.iloc[element[0]]["State"]))
        spot.append(str(df.iloc[element[0]]["Label"]))
        spot.append(str(df.iloc[element[0]]["Recreation"]))
        spot.append(str(df.iloc[element[0]]["Distance From Airport(KM)"]))
        spots.append(spot)
        print(str(df.iloc[element[0]]["Tourist Spots"])+" "+str(df.iloc[element[0]]["State"])+" "+str(df.iloc[element[0]]["Label"]))
        i=i+1
        if i>=n:
            break
    return spots