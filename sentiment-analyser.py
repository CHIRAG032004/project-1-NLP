import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string

class MovieSentimentAnalyzer:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer()
        self.model = None
        
    def preprocess_text(self, text):
        """Apply preprocessing: tokenization, stopword removal, lemmatization"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove HTML tags and special characters
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized_token = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized_token)
        
        return ' '.join(processed_tokens)

# Initialize the analyzer
analyzer = MovieSentimentAnalyzer()

print("Movie Review Sentiment Analyzer initialized!")
print("Preprocessing functions ready.")

# Load and preprocess the dataset
def load_and_preprocess_data():
    """Load dataset and apply preprocessing"""
    try:
        # Load the dataset
        df = pd.read_csv('data/movie_reviews.csv')
        print(f"Dataset loaded successfully! Shape: {df.shape}")
        
        # Check the columns and data
        print(f"Columns: {list(df.columns)}")
        print(f"Sentiment distribution:")
        print(df['sentiment'].value_counts())
        
        # Take a smaller sample for faster processing (first 5000 reviews)
        df_sample = df.head(5000).copy()
        print(f"Working with {len(df_sample)} reviews for faster processing")
        
        # Apply preprocessing to reviews
        print("Applying preprocessing...")
        df_sample['processed_review'] = df_sample['review'].apply(analyzer.preprocess_text)
        
        print("Preprocessing completed!")
        print(f"Sample processed review:")
        print(f"Original: {df_sample['review'].iloc[0][:200]}...")
        print(f"Processed: {df_sample['processed_review'].iloc[0][:200]}...")
        
        return df_sample
        
    except FileNotFoundError:
        print("Dataset file not found. Please run download_dataset.py first.")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Load and preprocess the data
data = load_and_preprocess_data()

# Train the models
def train_models(data):
    """Convert text to TF-IDF vectors and train classifiers"""
    if data is None:
        return None, None, None, None
    
    # Prepare the data
    X = data['processed_review']
    y = data['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Convert text to TF-IDF vectors
    print("Converting text to TF-IDF vectors...")
    X_train_tfidf = analyzer.vectorizer.fit_transform(X_train)
    X_test_tfidf = analyzer.vectorizer.transform(X_test)
    
    print(f"TF-IDF matrix shape: {X_train_tfidf.shape}")
    
    # Train Naive Bayes classifier
    print("Training Naive Bayes classifier...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    
    # Train Logistic Regression classifier
    print("Training Logistic Regression classifier...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train_tfidf, y_train)
    
    print("Model training completed!")
    
    return nb_model, lr_model, X_test_tfidf, y_test

# Train the models
print("\n" + "="*50)
print("TRAINING MODELS")
print("="*50)
nb_model, lr_model, X_test_tfidf, y_test = train_models(data)

# Evaluate models
def evaluate_models(nb_model, lr_model, X_test_tfidf, y_test):
    """Evaluate model performance with accuracy, precision, recall, F1-score"""
    if nb_model is None or lr_model is None:
        return
    
    # Make predictions
    nb_predictions = nb_model.predict(X_test_tfidf)
    lr_predictions = lr_model.predict(X_test_tfidf)
    
    # Calculate metrics for Naive Bayes
    nb_accuracy = accuracy_score(y_test, nb_predictions)
    nb_precision = precision_score(y_test, nb_predictions)
    nb_recall = recall_score(y_test, nb_predictions)
    nb_f1 = f1_score(y_test, nb_predictions)
    
    # Calculate metrics for Logistic Regression
    lr_accuracy = accuracy_score(y_test, lr_predictions)
    lr_precision = precision_score(y_test, lr_predictions)
    lr_recall = recall_score(y_test, lr_predictions)
    lr_f1 = f1_score(y_test, lr_predictions)
    
    # Display results
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    print(f"\nNAIVE BAYES CLASSIFIER:")
    print(f"Accuracy:  {nb_accuracy:.4f}")
    print(f"Precision: {nb_precision:.4f}")
    print(f"Recall:    {nb_recall:.4f}")
    print(f"F1-Score:  {nb_f1:.4f}")
    
    print(f"\nLOGISTIC REGRESSION CLASSIFIER:")
    print(f"Accuracy:  {lr_accuracy:.4f}")
    print(f"Precision: {lr_precision:.4f}")
    print(f"Recall:    {lr_recall:.4f}")
    print(f"F1-Score:  {lr_f1:.4f}")
    
    # Determine best model
    if nb_accuracy > lr_accuracy:
        print(f"\nBest Model: Naive Bayes (Accuracy: {nb_accuracy:.4f})")
        best_predictions = nb_predictions
    else:
        print(f"\nBest Model: Logistic Regression (Accuracy: {lr_accuracy:.4f})")
        best_predictions = lr_predictions
    
    return best_predictions

# Evaluate the models
best_predictions = evaluate_models(nb_model, lr_model, X_test_tfidf, y_test)

# Create confusion matrix visualization
def create_confusion_matrix(y_test, predictions):
    """Create and save confusion matrix visualization"""
    if predictions is None:
        return
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    
    # Create the visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'], 
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix - Movie Review Sentiment Analysis')
    plt.xlabel('Predicted Sentiment')
    plt.ylabel('Actual Sentiment')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved as 'confusion_matrix.png'")
    
    # Show the plot
    plt.show()
    
    # Print confusion matrix interpretation
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"True Negatives (Correct negative predictions): {tn}")
    print(f"False Positives (Incorrectly predicted as positive): {fp}")
    print(f"False Negatives (Incorrectly predicted as negative): {fn}")
    print(f"True Positives (Correct positive predictions): {tp}")

# Create confusion matrix
print("\n" + "="*50)
print("CREATING CONFUSION MATRIX")
print("="*50)
create_confusion_matrix(y_test, best_predictions)

print("\n" + "="*50)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*50)
print("Deliverables created:")
print("✓ Python script with preprocessing, training, and evaluation")
print("✓ Confusion matrix visualization (confusion_matrix.png)")
