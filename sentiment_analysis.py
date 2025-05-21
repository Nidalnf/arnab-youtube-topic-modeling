import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

# Load your cleaned comments CSV
df = pd.read_csv('sample_youtube_comments.csv')

# Function to get sentiment polarity (-1 = negative, 0 = neutral, 1 = positive)
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Apply to each comment
df['sentiment'] = df['comment_text'].astype(str).apply(get_sentiment)

# Label the sentiments
def label_sentiment(score):
    if score > 0.1:
        return 'Positive'
    elif score < -0.1:
        return 'Negative'
    else:
        return 'Neutral'

df['sentiment_label'] = df['sentiment'].apply(label_sentiment)

# Show some of them
print(df[['comment_text', 'sentiment', 'sentiment_label']].head(10))

# Plotting
sentiment_counts = df['sentiment_label'].value_counts()
sentiment_counts.plot(kind='bar', color=['green', 'red', 'gray'])
plt.title('Sentiment Distribution of Comments')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.tight_layout()
plt.show()
