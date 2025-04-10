# Sentiment Analysis for Indonesian Text

This project performs sentiment analysis on Indonesian text data using machine learning models. The models predict whether the sentiment of a given text is **positive**, **negative**, or **neutral**. The models use TF-IDF and Word2Vec techniques for text feature extraction, with the following machine learning models:
1. **Support Vector Machine (SVM)**
2. **Random Forest Classifier (with TF-IDF and Word2Vec)**

## Table of Contents
- [Introduction](#introduction)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Training the Models](#training-the-models)
- [Model Evaluation](#model-evaluation)
- [Inference](#inference)
- [Requirements](#requirements)

## Introduction

This project uses sentiment-labeled text reviews and applies machine learning techniques to classify the sentiment of new text. The models use techniques like TF-IDF and Word2Vec to process and analyze text, followed by the use of classifiers like SVM and Random Forest to predict sentiment.

## Technologies Used

- **Python** (Programming Language)
- **Pandas** (Data manipulation and analysis)
- **NumPy** (Numerical computing)
- **Scikit-learn** (Machine learning and data mining)
- **Gensim** (Text vectorization with Word2Vec)
- **Matplotlib** (Plotting and visualization)
- **NLTK** (Natural Language Toolkit for text processing)
- **Joblib** (Serialization for model storage)
- **Google Colab** (For running the code in a cloud environment)

## Installation

To use this project locally, make sure to install the required dependencies listed in the `requirements.txt` file.

### Steps:
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis-indonesian.git
   cd sentiment-analysis-indonesian

2. Create a virtual environment (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt

4. Download the NLTK stopwords:
    ```bash
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')

## Data Preprocessing

The dataset consists of text reviews labeled with sentiments. The text is processed through several stages:

- **Tokenization**: Using `nltk.tokenize.word_tokenize` to split the text into individual words.
- **Stopwords Removal**: Removing common stopwords (words like "the", "and", etc.).

### Feature Extraction:
- **TF-IDF**: Converts the text into numerical features.
- **Word2Vec**: Converts the text into vectors using Word2Vec.

The processed text is then used to train the models.

## Training the Models

There are three models in this project:

1. **Support Vector Machine (SVM) with TF-IDF**:
   - Trained using the TF-IDF features.
   
2. **Random Forest Classifier with Word2Vec**:
   - Trained using Word2Vec features.
   
3. **Random Forest Classifier with TF-IDF**:
   - Trained using TF-IDF features with a different train-test split ratio.

The models are trained on 80% of the dataset and evaluated on 20% of the dataset.

## Model Evaluation

Each model is evaluated using:

- **Accuracy**: The percentage of correct predictions.
- **Classification Report**: Precision, recall, and F1-score for each sentiment class.
- **Confusion Matrix**: To visualize the classification results.

## Inference

Once the models are trained, they can be used to predict the sentiment of new text reviews. The text needs to be processed with the same vectorizer (TF-IDF or Word2Vec) used during training.

### Example Usage:
```python
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the trained model and vectorizer
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Predict sentiment for a new text
sample_text = "Saya sangat senang dengan produk ini!"
prediction = predict_sentiment(sample_text, svm_model, vectorizer)
print(f"Sentimen Prediksi: {prediction}")

## Requirements

To run the project, you'll need the following libraries:

- pandas
- numpy
- scikit-learn
- gensim
- nltk
- matplotlib
- joblib

You can install all the dependencies using:

    ```bash
    pip install -r requirements.txt

## Contributing

Feel free to fork this repository and submit pull requests. Contributions are welcome!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
