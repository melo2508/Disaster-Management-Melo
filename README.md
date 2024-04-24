# Disaster-Management-Melo

This project involves creating a machine learning pipeline designed to classify a dataset comprising actual messages sent during disaster incidents. The goal is to automate the process of routing these messages to the relevant disaster relief agencies based on their content.

Step 1: Dataset: Used Dataset from https://github.com/joshasgard/Disaster-Response-App/tree/master/data

Step 2: Data preparation is done

Step 3: Model :
	Following algorithms are used for the same : 
		The code provided uses the following algorithms and methods:

1. **Random Forest Classifier (`RandomForestClassifier`):**
   - This is a popular ensemble learning method that builds multiple decision trees during training and outputs the class that is the mode of the classes predicted by individual trees.
   - It's used as the estimator in the `MultiOutputClassifier` to handle multi-label classification.

2. **Grid Search Cross-Validation (`GridSearchCV`):**
   - It's a method for hyperparameter tuning that exhaustively searches through a specified grid of hyperparameters to find the best combination based on a specified evaluation metric.
   - In the code, it's used to optimize hyperparameters for the Random Forest Classifier, such as `n_estimators` (number of trees) and `min_samples_split` (minimum samples required to split a node).

3. **Count Vectorizer (`CountVectorizer`):**
   - It's used to convert a collection of text documents into a matrix of token counts, where each row represents a document, and each column represents a unique word in the corpus.
   - It's configured with a tokenizer function (`tokenize` in this code) to preprocess text data before tokenizing.

4. **TF-IDF Transformer (`TfidfTransformer`):**
   - It transforms a count matrix obtained from Count Vectorizer into a TF-IDF (Term Frequency-Inverse Document Frequency) representation.
   - TF-IDF reflects the importance of a word in a document relative to its frequency across multiple documents in the corpus.

5. **MultiOutputClassifier (`MultiOutputClassifier`):**
   - It's used to extend binary classifiers to multi-output classifiers, which can handle multiple target variables simultaneously.
   - In this code, it's wrapped around the Random Forest Classifier to enable multi-label classification for predicting multiple categories simultaneously.

6. **Tokenization (`word_tokenize` from NLTK):**
   - It's a process of breaking down a text into smaller units called tokens, which can be words, phrases, symbols, or other meaningful elements.
   - Tokenization is used in the `tokenize` function to tokenize raw text messages before further processing.

7. **Text Preprocessing (`WordNetLemmatizer`, stopwords removal):**
   - The `WordNetLemmatizer` from NLTK is used to lemmatize tokens, which reduces words to their base or root form.
   - Stopwords removal is performed to filter out common words that do not carry significant meaning for classification.

8. **Train-Test Split (`train_test_split`):**
   - It's used to split the dataset into training and testing sets for model training and evaluation.

Overall, the code implements a machine learning pipeline that preprocesses text data, builds a Random Forest Classifier model optimized with Grid Search, and evaluates the model's performance on a test set.