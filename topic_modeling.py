import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the comments CSV
df = pd.read_csv('sample_youtube_comments.csv')

# Clean comments: remove special characters, convert to lowercase
df['comment_text'] = df['comment_text'].str.replace(r'[^a-zA-Z\s]', '', regex=True)
df['comment_text'] = df['comment_text'].str.lower()

# Optional: Remove very short comments (less than 5 words)
df = df[df['comment_text'].str.split().str.len() > 4]

# Vectorize using bigrams (1-2 word phrases) and remove rare terms (min_df=2)
vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=2)
X = vectorizer.fit_transform(df['comment_text'])

# Fit LDA with 4 topics
lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(X)

# Show the top 10 keywords for each topic
feature_names = vectorizer.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"\nTopic #{idx + 1}:")
    print(", ".join([feature_names[i] for i in topic.argsort()[-10:][::-1]]))



