import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc, classification_report
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.seasonal import seasonal_decompose
from lifelines import KaplanMeierFitter
import networkx as nx
from sklearn.cluster import DBSCAN
import folium
from scipy import stats
# Load the data from the Tripsit files
file_paths = [
    '/mnt/disks/data/analysis/combined_tripsit.csv',
    '/mnt/disks/data/analysis/open_tripsit-2_page_1.csv',
    '/mnt/disks/data/analysis/web-tripsit-2_page_1.csv',
    '/mnt/disks/data/analysis/web-tripsit-1-page_1.csv'
]
# Full list of substances grouped by categories with their variations
substances = {
    "Cannabis": ["cannabis", "marijuana", "hashish", "weed", "pot", "bud", "ganja", "grass", "dope", "reefer"],
    "Cocaine": ["cocaine", "powder", "crack", "coke", "snow", "rock", "freebase", "blow", "nose candy", "flake"],
    "Opioids": ["heroin", "morphine", "codeine", "fentanyl", "oxycodone", "hydrocodone", "oxymorphone", "tramadol", "methadone", "buprenorphine"],
    "Amphetamine-type stimulants": ["methamphetamine", "mdma", "ecstasy", "amphetamine", "speed", "crystal meth", "ice", "dexedrine", "adderall", "ritalin"],
    "Hallucinogens": ["lsd", "psilocybin", "dmt", "mescaline", "ayahuasca", "ibogaine", "salvia divinorum", "2c-b", "2c-i", "dox"],
    "Dissociatives": ["ketamine", "pcp", "dxm", "nitrous oxide", "mxe", "dck", "3-meo-pcp", "o-pce", "deschloroketamine", "diphenidine"],
    "Synthetic cannabinoids": ["k2", "spice", "jwh-018", "jwh-073", "am-2201", "ab-fubinaca", "ab-chminaca", "5f-adb", "5f-akb48", "mdmb-fubinaca"],
    "Synthetic cathinones": ["bath salts", "mephedrone", "mdpv", "alpha-pvp", "4-mmc", "3-mmc", "4-mec", "3-cmc", "pentedrone", "ethylone"],
    "Phenethylamines": ["2c-b", "2c-i", "dox", "dom", "dob", "doi", "2c-t-7", "2c-e", "2c-c", "2c-d"],
    "Tryptamines": ["5-meo-dmt", "dpt", "dmt", "psilocin", "psilocybin", "amt", "5-meo-mipt", "5-meo-dipt", "5-meo-dalt", "5-meo-dipt"],
    "Benzodiazepines": ["diazepam", "alprazolam", "xanax", "benzos", "clonazepam", "lorazepam", "bars", "flunitrazepam", "temazepam", "oxazepam", "nitrazepam", "bromazepam", "chlordiazepoxide"],
    "Barbiturates": ["phenobarbital", "secobarbital", "amobarbital", "butalbital", "pentobarbital", "thiopental", "thiamylal", "methohexital", "aprobarbital", "hexobarbital"],
    "Z-drugs": ["zolpidem", "zaleplon", "zopiclone", "eszopiclone", "zolimidine", "zoplicone", "indiplon", "zaleplon", "zopiclone", "eszopiclone"],
    "GHB": ["ghb", "gbl", "bd", "1,4-butanediol", "gamma-hydroxybutyric acid", "gamma-butyrolactone", "1,4-butanediol", "gamma-hydroxybutyrate", "gamma-hydroxybutanoic acid"], 
    "Prescription stimulants": ["amphetamine", "methylphenidate", "modafinil", "armodafinil", "dextroamphetamine", "lisdexamfetamine", "phentermine", "benzphetamine", "diethylpropion", "phendimetrazine"],
    "Inhalants": ["solvents", "aerosols", "gases", "nitrites", "amyl nitrite", "butyl nitrite", "cyclohexyl nitrite", "isobutyl nitrite", "isopropyl nitrite", "nitrous oxide"],
    # Note: 'Others' category can include any additional substances not listed above
    "Others": ["dextromethorphan", "dxm", "loperamide", "pseudoephedrine", "diphenhydramine", "acetaminophen", "ibuprofen", "aspirin", "codeine"]

}
# Flatten the list of substances for pattern matching
substance_list = [item for sublist in substances.values() for item in sublist]
# Data Loading Function
def load_data(file_paths):
    data_frames = []
    for file_path in file_paths:
        df = pd.read_csv(file_path, error_bad_lines=False, warn_bad_lines=True)
        data_frames.append(df)
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df
# Text Preprocessing and Substance Mention Counting Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\\s]', '', text)
    return text
def count_substance_mentions(message, substances):
    message = clean_text(message)
    return Counter(substance for substance in substances if substance in message)
# Descriptive Statistics Function
def descriptive_statistics(data):
    desc_stats = data.describe(include='all')
    return desc_stats
# Longitudinal Analysis Function
def longitudinal_analysis(data, time_var, target_var, freq='M', save_csv=False):
    data[time_var] = pd.to_datetime(data[time_var])
    data.set_index(time_var, inplace=True)
    data_resampled = data[target_var].resample(freq).mean()
    
    result = seasonal_decompose(data_resampled, model='additive')
    result.plot()
    plt.show()
    if save_csv:
        data_resampled.to_csv(f'{target_var}_trends.csv')
    return data_resampled
# Predictive Modeling Function
def predictive_modeling(data, features, target, model, param_grid, scoring='roc_auc'):
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3, random_state=42)
    
    grid = GridSearchCV(model, param_grid, scoring=scoring, cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_
    predictions = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC = {roc_auc:.2f})'))
    fig.update_layout(title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    fig.show()
    return best_model, grid.best_params_, roc_auc
# Network Analysis Function
def network_analysis(nodes, edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    pos = nx.spring_layout(G)
    centrality = nx.degree_centrality(G)
    communities = nx.community.greedy_modularity_communities(G)
    # Visualization code to be added
    return centrality, communities
# Spatial Clustering Function
def spatial_clustering(data, spatial_cols, eps=0.01, min_samples=5):
    coords = data[spatial_cols].values
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
    data['cluster'] = db.labels_
    
    # Visualization code to be added
    return data
# Sentiment Analysis
def analyze_sentiment(messages):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = [sia.polarity_scores(message) for message in messages]
    return sentiment_scores
"""
0 comments on commit 3cd4636
