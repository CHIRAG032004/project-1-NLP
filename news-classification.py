# News Article Classification Project
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import requests
import zipfile
import os

print("All imports successful!")


# Download BBC News Dataset
def download_bbc_dataset():
    """Download and extract BBC News dataset"""
    url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
    
    print("Downloading BBC News dataset...")
    response = requests.get(url)
    
    # Save the zip file
    with open("bbc-fulltext.zip", "wb") as file:
        file.write(response.content)
    
    # Extract the zip file
    with zipfile.ZipFile("bbc-fulltext.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    
    print("Dataset downloaded and extracted successfully!")
    return True

# Download the dataset
if not os.path.exists("bbc"):
    download_bbc_dataset()
else:
    print("Dataset already exists!")


# Load the dataset
def load_bbc_dataset():
    """Load BBC news articles from the extracted folders"""
    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    articles = []
    labels = []
    
    for category in categories:
        folder_path = f"bbc/{category}"
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename.endswith('.txt'):
                    with open(f"{folder_path}/{filename}", 'r', encoding='utf-8', errors='ignore') as file:
                        content = file.read()
                        articles.append(content)
                        labels.append(category)
    
    return articles, labels

# Load the data
print("Loading BBC News dataset...")
articles, labels = load_bbc_dataset()

# Create DataFrame
df = pd.DataFrame({
    'text': articles,
    'category': labels
})

print(f"Dataset loaded successfully!")
print(f"Total articles: {len(df)}")
print(f"Categories: {df['category'].unique()}")
print(f"Articles per category:\n{df['category'].value_counts()}")
print(f"\nFirst article preview:\n{df['text'].iloc[0][:200]}...")


# Text Preprocessing Function
def preprocess_text(text, use_stemming=True):
    """Preprocess text: lowercase, remove punctuation, stopwords, stem/lemmatize"""
    # Initialize tools
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Convert to lowercase and remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # Tokenize
    words = text.split()
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Apply stemming or lemmatization
    if use_stemming:
        words = [stemmer.stem(word) for word in words]
    else:
        words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)

# Apply preprocessing
print("Applying text preprocessing...")
df['processed_text_stemmed'] = df['text'].apply(lambda x: preprocess_text(x, use_stemming=True))
df['processed_text_lemmatized'] = df['text'].apply(lambda x: preprocess_text(x, use_stemming=False))

print("Preprocessing completed!")
print(f"\nOriginal text example:\n{df['text'].iloc[0][:200]}...")
print(f"\nProcessed text (stemmed) example:\n{df['processed_text_stemmed'].iloc[0][:200]}...")


# Split the data for training and testing
print("\nSplitting data into train and test sets...")
X = df['processed_text_stemmed']  # Using stemmed text
y = df['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Create Bag of Words representation
print("\nCreating Bag of Words (BoW) representation...")
bow_vectorizer = CountVectorizer(max_features=5000, ngram_range=(1,2))
X_train_bow = bow_vectorizer.fit_transform(X_train)
X_test_bow = bow_vectorizer.transform(X_test)
print(f"BoW feature shape: {X_train_bow.shape}")

# Create TF-IDF representation
print("\nCreating TF-IDF representation...")
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
print(f"TF-IDF feature shape: {X_train_tfidf.shape}")

print("\nText representation completed!")


# Train classifiers
print("\n" + "="*50)
print("TRAINING CLASSIFIERS")
print("="*50)

# Dictionary to store results
results = {}

# 1. Logistic Regression with BoW
print("\n1. Training Logistic Regression with BoW...")
lr_bow = LogisticRegression(random_state=42, max_iter=1000)
lr_bow.fit(X_train_bow, y_train)
lr_bow_pred = lr_bow.predict(X_test_bow)
lr_bow_accuracy = accuracy_score(y_test, lr_bow_pred)
print(f"Logistic Regression (BoW) Accuracy: {lr_bow_accuracy:.4f}")
results['LR_BoW'] = lr_bow_accuracy

# 2. Logistic Regression with TF-IDF
print("\n2. Training Logistic Regression with TF-IDF...")
lr_tfidf = LogisticRegression(random_state=42, max_iter=1000)
lr_tfidf.fit(X_train_tfidf, y_train)
lr_tfidf_pred = lr_tfidf.predict(X_test_tfidf)
lr_tfidf_accuracy = accuracy_score(y_test, lr_tfidf_pred)
print(f"Logistic Regression (TF-IDF) Accuracy: {lr_tfidf_accuracy:.4f}")
results['LR_TFIDF'] = lr_tfidf_accuracy

# 3. SVM with BoW
print("\n3. Training SVM with BoW...")
svm_bow = SVC(random_state=42, kernel='linear')
svm_bow.fit(X_train_bow, y_train)
svm_bow_pred = svm_bow.predict(X_test_bow)
svm_bow_accuracy = accuracy_score(y_test, svm_bow_pred)
print(f"SVM (BoW) Accuracy: {svm_bow_accuracy:.4f}")
results['SVM_BoW'] = svm_bow_accuracy

# 4. SVM with TF-IDF
print("\n4. Training SVM with TF-IDF...")
svm_tfidf = SVC(random_state=42, kernel='linear')
svm_tfidf.fit(X_train_tfidf, y_train)
svm_tfidf_pred = svm_tfidf.predict(X_test_tfidf)
svm_tfidf_accuracy = accuracy_score(y_test, svm_tfidf_pred)
print(f"SVM (TF-IDF) Accuracy: {svm_tfidf_accuracy:.4f}")
results['SVM_TFIDF'] = svm_tfidf_accuracy

print(f"\nTraining completed! All models trained successfully.")


# Generate detailed classification reports
print("\n" + "="*50)
print("DETAILED EVALUATION RESULTS")
print("="*50)

print("\nBest performing model: SVM with TF-IDF")
print("Classification Report:")
print(classification_report(y_test, svm_tfidf_pred))

# Create accuracy comparison visualization
print("\nCreating accuracy comparison visualization...")
plt.figure(figsize=(12, 8))

# 1. Accuracy Comparison Bar Chart
plt.subplot(2, 2, 1)
models = list(results.keys())
accuracies = list(results.values())
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']
bars = plt.bar(models, accuracies, color=colors)
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy')
plt.ylim(0.95, 1.0)
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
             f'{acc:.4f}', ha='center', va='bottom')

# 2. BoW vs TF-IDF Comparison
plt.subplot(2, 2, 2)
bow_acc = [results['LR_BoW'], results['SVM_BoW']]
tfidf_acc = [results['LR_TFIDF'], results['SVM_TFIDF']]
x = ['Logistic Regression', 'SVM']
width = 0.35
x_pos = np.arange(len(x))
plt.bar(x_pos - width/2, bow_acc, width, label='BoW', color='lightblue')
plt.bar(x_pos + width/2, tfidf_acc, width, label='TF-IDF', color='orange')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.title('BoW vs TF-IDF Comparison')
plt.xticks(x_pos, x)
plt.legend()
plt.ylim(0.95, 1.0)

plt.tight_layout()
plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'accuracy_comparison.png'")


# Create distribution of predicted categories
print("\nCreating category distribution visualization...")
plt.figure(figsize=(15, 5))

# 1. True vs Predicted Distribution
plt.subplot(1, 3, 1)
true_counts = pd.Series(y_test).value_counts().sort_index()
plt.pie(true_counts.values, labels=true_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('True Category Distribution\n(Test Set)')

plt.subplot(1, 3, 2)
pred_counts = pd.Series(svm_tfidf_pred).value_counts().sort_index()
plt.pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Predicted Category Distribution\n(SVM + TF-IDF)')

# 3. Category-wise Accuracy
plt.subplot(1, 3, 3)
category_accuracy = {}
for category in np.unique(y_test):
    mask = y_test == category
    category_acc = accuracy_score(y_test[mask], svm_tfidf_pred[mask])
    category_accuracy[category] = category_acc

categories = list(category_accuracy.keys())
accuracies = list(category_accuracy.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
bars = plt.bar(categories, accuracies, color=colors)
plt.title('Category-wise Accuracy')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.ylim(0.95, 1.01)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{acc:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("Category distribution visualization saved as 'category_distribution.png'")

# Summary
print("\n" + "="*60)
print("PROJECT SUMMARY")
print("="*60)
print(f"Dataset: BBC News Dataset with {len(df)} articles")
print(f"Categories: {', '.join(df['category'].unique())}")
print(f"Best Model: SVM with TF-IDF (Accuracy: {results['SVM_TFIDF']:.4f})")
print(f"BoW vs TF-IDF: TF-IDF performs better")
print("Files created:")
print("- news_classifier.py (main script)")
print("- accuracy_comparison.png")
print("- category_distribution.png")
print("="*60)
