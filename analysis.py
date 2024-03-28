from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd

# Assuming 'topics_from_topic_modeling' and 'topic_assignments' are available from topic modeling
# topics_from_topic_modeling: Dictionary with topic IDs and their keywords
# topic_assignments: List with a topic ID assigned to each message

def enhanced_analyze_sentiment(messages, topic_assignments, topics_from_topic_modeling):
    sia = SentimentIntensityAnalyzer()
    
    # Analyze sentiment of each message
    sentiment_scores = [sia.polarity_scores(message) for message in messages]
    
    # Adjust sentiment scores based on context
    contextual_scores = [adjust_score_based_on_context(score, message) for score, message in zip(sentiment_scores, messages)]
    
    # Aggregate sentiment scores for overall sentiment
    avg_sentiment = np.mean([score['compound'] for score in contextual_scores])
    
    # Sentiment over time with a 7-day rolling average
    sentiment_over_time = pd.Series([score['compound'] for score in contextual_scores]).rolling(window=7).mean()
    
    # Sentiment by topic
    sentiment_by_topic = analyze_sentiment_by_topic(messages, topic_assignments, topics_from_topic_modeling, contextual_scores)

    return contextual_scores, avg_sentiment, sentiment_over_time, sentiment_by_topic

def adjust_score_based_on_context(score, message):
    # Logic to adjust sentiment scores based on the context of the message
    # This could involve checking for the presence of key terms related to harm reduction and adjusting the score accordingly
    adjusted_score = score  # Placeholder for actual logic
    return adjusted_score

def analyze_sentiment_by_topic(messages, topic_assignments, topics, scores):
    # Analyze sentiment within specific topics
    sentiment_by_topic = {}
    for topic_id, topic_keywords in topics.items():
        topic_messages_indices = [i for i, topic in enumerate(topic_assignments) if topic == topic_id]
        topic_scores = [scores[i]['compound'] for i in topic_messages_indices]
        sentiment_by_topic[topic_id] = np.mean(topic_scores)
    return sentiment_by_topic
