from flask import Flask, request, jsonify, render_template
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import re
import os
import nltk
from transformers import pipeline

app = Flask(__name__)

# Ensure the static folder exists
if not os.path.exists('static'):
    os.makedirs('static')

# Download VADER lexicon
nltk.download('vader_lexicon')

# YouTube API credentials
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
API_KEY = os.getenv('api_key') # Replace with your YouTube API key

def extract_video_id(youtube_link):
    regex = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(regex, youtube_link)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube video link. Could not extract video ID.")

def get_video_comments(video_id, max_comments=500):
    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)
        comments = []
        next_page_token = None

        while len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                pageToken=next_page_token,
                textFormat='plainText',
                maxResults=min(100, max_comments - len(comments))
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

        return comments
    except HttpError as e:
        print(f"An HTTP error occurred: {e}")
        return []

def analyze_sentiment(comment):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(comment)['compound']

def summarizing(comments, max_length=130):
    summarizer = pipeline("summarization")
    combined_text = " ".join(comments)
    summary = summarizer(combined_text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']
    
def get_comments_with_sentiment(youtube_link, max_comments=500):
    try:
        video_id = extract_video_id(youtube_link)
        comments = get_video_comments(video_id, max_comments)

        comments_with_sentiment = [
            {'Comment': comment, 'Sentiment': analyze_sentiment(comment)}
            for comment in comments
        ]
        
        comments_df = pd.DataFrame(comments_with_sentiment)
        return comments_df
    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()

def summarize_comments(comments_df):
    if comments_df.empty:
        return None
    
    Positive = summarizing(comments_df[comments_df['Sentiment'] >= 0.4]['Comment'].values)
    Negative = summarizing(comments_df[comments_df['Sentiment'] <= -0.4]['Comment'].values)
    
    summary = {
        "Total Comments": len(comments_df),
        "Average Sentiment": round(comments_df["Sentiment"].mean(), 2),
        "Positive Comments": (comments_df["Sentiment"] > 0).sum(),
        "Neutral Comments": (comments_df["Sentiment"] == 0).sum(),
        "Negative Comments": (comments_df["Sentiment"] < 0).sum(),
        "Most Positive Comment": Positive,
        "Most Negative Comment": Negative,
    }
    return summary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    youtube_link = request.form['youtube_link']
    comments_df = get_comments_with_sentiment(youtube_link)
    summary = summarize_comments(comments_df)

    if summary is None:
        return "No comments found or an error occurred. Please check the video link and try again."
    
    return render_template('results.html', summary=summary)

if __name__ == '__main__':
    app.run()
