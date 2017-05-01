import pandas as pd
from sklearn.decomposition import PCA

def generate_features(df):
    df["date_time"] = pd.to_datetime(df["date_time"]);
    df["srch_ci"] = pd.to_datetime(df["srch_ci"], format='%Y-%m-%d', errors="coerce");
    df["srch_co"] = pd.to_datetime(df["srch_co"], format='%Y-%m-%d', errors="coerce");
    
    features = {};
    for prop in ["year","month", "day"]:
        features["srch_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
    #for index in ["year","month", "day"]:
     #   features[index] = getattr(df["date_time"].dt, index);
    
    carryover = [p for p in df.columns if p not in ["date_time", "srch_ci", "srch_co"]]
    for prop in carryover:
        features[prop] = df[prop]
    
    date_features = ["year","month", "day", "dayofweek"]
    for prop in date_features:
        features["ci_{0}".format(prop)] = getattr(df["srch_ci"].dt, prop)
        features["co_{0}".format(prop)] = getattr(df["srch_co"].dt, prop)
    features["stay_time"] = (df["srch_co"] - df["srch_ci"]).astype('timedelta64[h]')
        
    cleaned = pd.DataFrame(features)
    
    cleaned = cleaned.join(projection, on="srch_destination_id", how='left', rsuffix="dest")
    cleaned = cleaned.drop("srch_destination_iddest", axis=1)
    return cleaned;

destinations = pd.read_csv("./data/destinations.csv")
train = pd.read_csv("../data/10cluster_new.csv")

# do PCA and project 149 columns into 3 columns
pca = PCA(n_components=3)
projection = pca.fit_transform(destinations[["d{0}".format(i + 1) for i in range(149)]])
projection = pd.DataFrame(projection)
projection.columns = ['latent_1', 'latent_2','latent_3']

projection["srch_destination_id"] = destinations["srch_destination_id"]
cleaned = generate_features(train);
#print cleaned.head();
cleaned.fillna(0, inplace=True)
cleaned.to_csv("../data/data_cleaned.csv", sep=',',index=False);


