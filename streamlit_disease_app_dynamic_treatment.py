
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Symptom2Disease_cleaned_final.csv")
    df.dropna(subset=["text", "label", "treatment"], inplace=True)
    return df

df = load_data()

# Build dynamic treatment lookup
treatment_lookup = df.groupby("label")["treatment"].first().to_dict()

# Preprocessing
basic_stopwords = set([
    'i','me','my','myself','we','our','ours','ourselves','you',
    'your','yours','yourself','yourselves','he','him','his','himself',
    'she','her','hers','herself','it','its','itself','they','them',
    'their','theirs','themselves','what','which','who','whom','this',
    'that','these','those','am','is','are','was','were','be','been',
    'being','have','has','had','having','do','does','did','doing',
    'a','an','the','and','but','if','or','because','as','until',
    'while','of','at','by','for','with','about','against','between',
    'into','through','during','before','after','above','below','to',
    'from','up','down','in','out','on','off','over','under','again',
    'further','then','once','here','there','when','where','why','how',
    'all','any','both','each','few','more','most','other','some',
    'such','no','nor','not','only','own','same','so','than','too',
    'very','s','t','can','will','just','don','should','now'
])

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in basic_stopwords]
    return ' '.join(tokens)

df['clean_symptoms'] = df['text'].apply(preprocess)

# TF-IDF + Model Training
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_symptoms'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Streamlit Interface
st.title("🧠 Disease Prediction from Symptoms")
user_input = st.text_area("Enter your symptoms (comma-separated):")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter at least one symptom.")
    else:
        cleaned = preprocess(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = nb_model.predict(vector)[0]
        treatment = treatment_lookup.get(prediction, "No safety measures or treatment found.")
        st.success(f"🦠 Predicted Disease: {prediction}")
        st.info(f"💊 Suggested Treatment or Safety Measures \n{treatment}")
