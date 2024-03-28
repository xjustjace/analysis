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
import spacy

# Load spaCy for text processing
nlp = spacy.load("en_core_web_sm")

# Specify the file paths
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
    "Others": ["dextromethorphan", "dxm", "loperamide", "pseudoephedrine", "diphenhydramine", "acetaminophen", "ibuprofen", "aspirin", "codeine"]
}

# Flatten the list of substances for pattern matching
substance_list = [item for sublist in substances.values() for item in sublist]

# Data Loading Function
def load_data(file_paths, encoding='utf-8'):
    """
    Load data from a list of CSV file paths and concatenate them into a single DataFrame.
    """
    data_frames = []
    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            data_frames.append(df)
        except FileNotFoundError:
            print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    if not data_frames:
        raise ValueError("No objects to concatenate")
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

# Load the data
data = load_data(file_paths)

# Text Preprocessing and Substance Mention Counting Function
def clean_text(text):
    """
    Normalize text data by converting to lowercase and removing special characters.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def count_substance_mentions(message, substance_list):
    """
    Count mentions of substances in a given message.
    """
    message = clean_text(message)
    return Counter(substance for substance in substance_list if substance in message)

# Descriptive Statistics Function
def descriptive_statistics(data):
    """
    Compute and return descriptive statistics for a given DataFrame.
    """
    desc_stats = data.describe(include='all')
    return desc_stats

# Longitudinal Analysis Function
def longitudinal_analysis(data, time_var, target_var, freq='M'):
    """
    Perform longitudinal analysis on the given dataset to identify trends and patterns over time.
    Parameters:
    - data (pd.DataFrame): DataFrame containing the time series data.
    - time_var (str): Name of the column representing time.
    - target_var (str): Name of the column representing the variable of interest.
    - freq (str): Frequency of the time series data ('M' for monthly, 'D' for daily, etc.).

    Returns:
    - DecomposeResult: Object with seasonal, trend, and residual components of the time series.
    """
    # Convert the time variable to datetime and set it as the index
    data[time_var] = pd.to_datetime(data[time_var])
    data.set_index(time_var, inplace=True)

    # Resample the data according to the specified frequency and compute mean
    data_resampled = data[target_var].resample(freq).mean()

    # Decompose the time series data to identify trend, seasonality, and residuals
    result = seasonal_decompose(data_resampled, model='additive')

    # Plot the decomposition
    result.plot()
    plt.show()

    return result

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

# Analyze Substance Mentions
def analyze_substance_mentions(data, content_col='content', time_col='timestamp', output_dir='/mnt/disks/data/analysis'):
    """
    Analyze how often substances are mentioned in the data and generate visualizations and CSV outputs.
    
    Args:
        data (pd.DataFrame): The input data.
        content_col (str): The name of the column containing the text content.
        time_col (str): The name of the column representing time.
        output_dir (str): The directory path where output files will be saved.
    """
    # Drop rows with NaN in the content column
    data = data[data[content_col].notnull()]
    
    # Convert the time column to datetime
    data[time_col] = pd.to_datetime(data[time_col], errors='coerce')
    
    # Resample data by month and count substance mentions
    substance_mentions = data.set_index(time_col)[content_col].resample('M').apply(lambda x: sum(count_substance_mentions(' '.join(x.dropna()), substance_list).values()))
    
    # Save substance mention counts to CSV
    substance_mentions.to_csv(f"{output_dir}/substance_mentions.csv")
    
    # Plot line chart of substance mention trends
    fig, ax = plt.subplots(figsize=(12, 6))
    substance_mentions.plot(ax=ax, legend=False)
    ax.set_title("Substance Mention Trends Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Substance Mentions")
    plt.savefig(f"{output_dir}/substance_mention_trends.png", bbox_inches='tight')
    
    # Count total mentions per substance
    total_mentions = sum(count_substance_mentions(' '.join(data[content_col].dropna()), substance_list).values())
    substance_counts = count_substance_mentions(' '.join(data[content_col].dropna()), substance_list)
    
    # Save substance counts to CSV
    substance_counts_df = pd.DataFrame.from_dict(substance_counts, orient='index', columns=['count'])
    substance_counts_df.to_csv(f"{output_dir}/substance_counts.csv")
    
    # Plot bar chart of top substance mentions
    top_substances = substance_counts_df.sort_values('count', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(top_substances.index, top_substances['count'])
    ax.set_title(f"Top {len(top_substances)} Substances Mentioned (Total Mentions: {total_mentions})")
    ax.set_xlabel("Substance")
    ax.set_ylabel("Number of Mentions")
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f"{output_dir}/top_substance_mentions.png", bbox_inches='tight')

# Example usage
analyze_substance_mentions(data, content_col='content', time_col='timestamp', output_dir='/mnt/disks/data/analysis')
