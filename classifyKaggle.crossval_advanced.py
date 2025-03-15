'''
  This program shell reads phrase data for the kaggle phrase sentiment classification problem.
  The input to the program is the path to the kaggle directory "corpus" and a limit number.
  The program reads all of the kaggle phrases, and then picks a random selection of the limit number.
  It creates a "phrasedocs" variable with a list of phrases consisting of a pair
    with the list of tokenized words from the phrase and the label number from 1 to 4
  It prints a few example phrases.
  In comments, it is shown how to get word lists from the two sentiment lexicons:
      subjectivity and LIWC, if you want to use them in your features
  Your task is to generate features sets and train and test a classifier.

  Usage:  python classifyKaggle.py  <corpus directory path> <limit number>

  This version uses cross-validation with the Naive Bayes classifier in NLTK.
  It computes the evaluation measures of precision, recall and F1 measure for each fold.
  It also averages across folds and across labels.
'''
# open python and nltk packages needed for processing
import os
import sys
import random
import nltk
from nltk.corpus import stopwords
import string
import random
import re
import numpy as np
from nltk.collocations import *
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import warnings
from sklearn.exceptions import ConvergenceWarning

# Set the random state for reproducibility
random.seed(42)

## this code is commented off now, but can be used for sentiment lists
import sentiment_read_subjectivity
# initialize the positive, neutral and negative word lists
(positivelist, neutrallist, negativelist) = sentiment_read_subjectivity.read_subjectivity_three_types('SentimentLexicons/subjclueslen1-HLTEMNLP05.tff')

# initialize SL_subjectivity dictionary
SL_subjectivity = sentiment_read_subjectivity.readSubjectivity('SentimentLexicons/subjclueslen1-HLTEMNLP05.tff')

import sentiment_read_LIWC_pos_neg_words
# initialize positve and negative word prefix lists from LIWC 
#   note there is another function isPresent to test if a word's prefix is in the list
(poslist, neglist) = sentiment_read_LIWC_pos_neg_words.read_words()

import afinn
#initialize AFINN_Lexicon
afinn_lex = afinn.read_afinn("SentimentLexicons/AFINN-111.txt")

## define a feature definition function here

# this function define features (keywords) of a document for a BOW/unigram baseline
# each feature is 'V_(keyword)' and is true or false depending
# on whether that keyword is in the document
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)
    return features


# define a new combined featuresets that also includes afinn features
def combined_sentiment_features(document, word_features, sl_positivelist, sl_neutrallist, sl_negativelist,liwc_poslist, liwc_neglist, afinn_lex):
    document_words = set(document)
    features = {}

    #unigram features
    for word in word_features:
        features['V_{}'.format(word)] = (word in document_words)

    #SL features
    # Count the number of positive and negative words in the document
    sl_pos_count = sum(1 for word in document_words if word in sl_positivelist)
    sl_neu_count = sum(1 for word in document_words if word in sl_neutrallist)
    sl_neg_count = sum(1 for word in document_words if word in sl_negativelist)

    # Add sl sentiment counts as features
    features['sl_Positive_Count'] = sl_pos_count
    features['sl_Negative_Count'] = sl_neg_count
    features['sl_Neutral_Count'] = sl_neu_count

    #LIWC features
    # Count the number of positive and negative words in the document
    liwc_pos_count = sum(1 for word in document_words if word in liwc_poslist)
    liwc_neg_count = sum(1 for word in document_words if word in liwc_neglist)

    # Add presence using prefix-based matching
    liwc_pos_present = any(sentiment_read_LIWC_pos_neg_words.isPresent(word, liwc_poslist) for word in document_words)
    liwc_neg_present = any(sentiment_read_LIWC_pos_neg_words.isPresent(word, liwc_neglist) for word in document_words)

    # Add LIWC sentiment counts as features
    features['LIWC_Positive_Count'] = liwc_pos_count
    features['LIWC_Negative_Count'] = liwc_neg_count
    features['LIWC_Positive_Present'] = liwc_pos_present
    features['LIWC_Negative_Present'] = liwc_neg_present 

    #afinn features
    afinn_pos_sum = 0
    afinn_neg_sum = 0
    for word in document_words:
      score = afinn_lex.get(word, 0)  # Get score, default to 0 if not in lexicon
      if score > 0:
          afinn_pos_sum += score
      elif score < 0:
            afinn_neg_sum += score

    features['afinn_pos_count'] = afinn_pos_sum
    features['afinn_neg_count'] = afinn_neg_sum

    return features

## cross-validation ##
# this function takes the number of folds, the feature sets and the labels
# it iterates over the folds, using different sections for training and testing in turn
#   it prints the performance for each fold and the average performance at the end
def cross_validation_PRF(num_folds, featuresets, test_featuresets, labels, classifier_type):
    subset_size = int(len(featuresets) / num_folds)
    print('Each fold size:', subset_size)

    vec = DictVectorizer(sparse=False)
    X = vec.fit_transform([features for (features, label) in featuresets])
    y = np.array([label for (features, label) in featuresets])

    # Scale Data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape input for RNN: (samples, time steps, features)
    if classifier_type == 'Recurrent Neural Network':
        X_rnn = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])


    # Initialize RNN model once
    model = None
    # Initialize RNN model once
    if classifier_type == 'Recurrent Neural Network':
        print("Predefining RNN model...")
        model = Sequential([
            Input(shape=(1, X.shape[1])),
            Bidirectional(LSTM(64, activation='tanh', return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            LSTM(64, activation='tanh', return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(len(labels), activation='softmax')
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    classifier = None  # Default value for other classifiers
    # Initialize totals
    num_labels = len(labels)
    total_precision_list = [0] * num_labels
    total_recall_list = [0] * num_labels
    total_F1_list = [0] * num_labels

    for fold in range(num_folds):
        test_start = fold * subset_size
        test_end = test_start + subset_size

        if classifier_type == 'Recurrent Neural Network':
            # Prepare training and test sets for RNN
            X_test = X_rnn[test_start:test_end]
            y_test = y[test_start:test_end]
            X_train = np.concatenate((X_rnn[:test_start], X_rnn[test_end:]), axis=0)
            y_train = np.concatenate((y[:test_start], y[test_end:]), axis=0)

            # Convert labels to categorical (one-hot encoding)
            y_train_categorical = to_categorical(y_train, num_classes=num_labels)
            y_test_categorical = to_categorical(y_test, num_classes=num_labels)

            # Early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            # Train the RNN model
            model.fit(X_train, y_train_categorical,
                      epochs=50, batch_size=32, validation_split=0.1,
                      callbacks=[early_stopping], verbose=0)

            # Predict probabilities and take the argmax for class predictions
            y_pred_probs = model.predict(X_test)
            y_pred = np.argmax(y_pred_probs, axis=1)

        elif classifier_type == 'Random Forest':
            X_test, y_test = X_scaled[test_start:test_end], y[test_start:test_end]
            X_train = np.concatenate((X_scaled[:test_start], X_scaled[test_end:]), axis=0)
            y_train = np.concatenate((y[:test_start], y[test_end:]), axis=0)

            classifier = RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                class_weight='balanced', random_state=42
            )
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

        elif classifier_type == 'Multi-layer Perceptron':
            X_test, y_test = X_scaled[test_start:test_end], y[test_start:test_end]
            X_train = np.concatenate((X_scaled[:test_start], X_scaled[test_end:]), axis=0)
            y_train = np.concatenate((y[:test_start], y[test_end:]), axis=0)

            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            classifier = MLPClassifier(
                hidden_layer_sizes=(128, 64),  # Two hidden layers with 128 and 64 neurons
                activation='relu',
                solver='adam',
                alpha=0.001,  # L2 regularization
                batch_size=32,
                max_iter=10,  # Increased number of iterations
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

        else:
            raise ValueError("Unsupported classifier type. Choose from: RandomForest, Multi-layer Perceptron, RNN.")

        # Calculate metrics
        precision = precision_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
        recall = recall_score(y_test, y_pred, average=None, labels=labels, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=None, labels=labels, zero_division=0)

        # Aggregate metrics across folds
        for i in range(num_labels):
            total_precision_list[i] += precision[i]
            total_recall_list[i] += recall[i]
            total_F1_list[i] += f1[i]

    # Compute averages
    precision_list = [tot / num_folds for tot in total_precision_list]
    recall_list = [tot / num_folds for tot in total_recall_list]
    F1_list = [tot / num_folds for tot in total_F1_list]

    # Print results
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]),
              "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))

    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list) / num_labels),
          "{:10.3f}".format(sum(recall_list) / num_labels),
          "{:10.3f}".format(sum(F1_list) / num_labels))
    
    label_counts = {}
    for lab in labels:
      label_counts[lab] = 0 
    # count the labels
    for (doc, lab) in featuresets:
      label_counts[lab] += 1
    # make weights compared to the number of documents in featuresets
    num_docs = len(featuresets)
    label_weights = [(label_counts[lab] / num_docs) for lab in labels]
    #print('\nLabel Counts', label_counts)
    #print('Label weights', label_weights)
    # print macro average over all labels
    print('Micro Average Precision\tRecall\t\tF1 \tOver All Labels')
    precision = sum([a * b for a,b in zip(precision_list, label_weights)])
    recall = sum([a * b for a,b in zip(recall_list, label_weights)])
    F1 = sum([a * b for a,b in zip(F1_list, label_weights)])
    print( '\t', "{:10.3f}".format(precision), \
      "{:10.3f}".format(recall), "{:10.3f}".format(F1))
    

    # Predict sentiments for test features
    print("\nPredicting sentiments for test set...")
    test_X = vec.transform([features for (_, features) in test_featuresets])

    if classifier_type in ['Random Forest', 'Multi-layer Perceptron']:
        test_predictions = classifier.predict(test_X)
    elif classifier_type == 'Recurrent Neural Network':
        test_X_rnn = test_X.reshape(test_X.shape[0], 1, test_X.shape[1])
        test_predictions_probs = model.predict(test_X_rnn)
        test_predictions = np.argmax(test_predictions_probs, axis=1)

    output_file = 'test_predictions.csv'
    with open(output_file, 'w') as outfile:
        outfile.write('PhraseID,Sentiment\n')
        for i, (phrase_id, _) in enumerate(test_featuresets):
            outfile.write(f"{phrase_id},{test_predictions[i]}\n")

    print(f"Test predictions saved to {output_file}")



## function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
  #Train set

  f = open('./train.tsv', 'r')
  # loop over lines in the file and use the first limit of them
  phrasedata = []
  for line in f:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      # each line has 4 items separated by tabs
      # ignore the phrase and sentence ids, and keep the phrase and sentiment
      phrasedata.append(line.split('\t')[2:4])
  
  # pick a random sample of length limit because of phrase overlapping sequences
  random.shuffle(phrasedata)
  phraselist = phrasedata[:limit]

  print('Read', len(phrasedata), 'phrases, using', len(phraselist), 'random phrases')
  
  # create list of phrase documents as (list of words, label)
  phrasedocs = []
  
  # add all the phrases
  # each phrase has a list of tokens and the sentiment label (from 0 to 4)
  ### bin to only 3 categories for better performance
  for phrase in phraselist:
    tokens = nltk.word_tokenize(phrase[0])
    phrasedocs.append((tokens, int(phrase[1])))

  # possibly filter tokens
  # lowercase - each phrase is a pair consisting of a token list and a label
  # Prepare stopwords and punctuation


   #Test Set
  
  f_test = open('./test.tsv', 'r')
  
  # loop over lines in the file and use the first limit of them
  test_phrasedata = []
  
  for line in f_test:
    # ignore the first line starting with Phrase and read all lines
    if (not line.startswith('Phrase')):
      # remove final end of line character
      line = line.strip()
      columns = line.split('\t')  # Split the line by tabs

      # Ensure the line has at least three columns
      if len(columns) >= 3:
                test_phrasedata.append((columns[0], columns[2]))  # Append only the Phrase column


  # Create list of test phrases as (PhraseId, list of words)
  test_phrasedocs = []
  for phrase in test_phrasedata:
        tokens = nltk.word_tokenize(phrase[1])  # Tokenize the Phrase column
        test_phrasedocs.append((int(phrase[0]), tokens))


  # Define additional unnecessary words
  additional_words = {
        "'s", "n't", "'re", "'ve", "'d", "also", "thing", "maybe", 
        "would", "could", "should", "might", "must", "lot", "etc", "ok", "okay", "oh", "uh", "um"}

  stop_words = set(stopwords.words('english'))
  # Combine NLTK stopwords and additional unnecessary words
  
  combined_stopwords = stop_words.union(additional_words)
  punctuation_pattern = re.compile(r'[\s!"#$%&,\-./:;?@^_`\']{1,}')  # Match 2+ punctuations

  docs = []
  # original tokens 
  #for phrase in phrasedocs:
  #  lowerphrase = ([w.lower() for w in phrase[0]], phrase[1])
  #  docs.append (lowerphrase)

  #remove stop words and punctuations  
  for phrase in phrasedocs:
    filtered_tokens = [w.lower() for w in phrase[0] if w.lower() not in combined_stopwords and not punctuation_pattern.match(w)]
    docs.append((filtered_tokens, phrase[1]))


  test_docs = []
  #remove stop words and punctuations
  for phrase in test_phrasedocs:
        filtered_tokens = [w.lower() for w in phrase[1] if w.lower() not in combined_stopwords and not punctuation_pattern.match(w)]
        test_docs.append((phrase[0], filtered_tokens))  # Retain PhraseId with filtered tokens

      
    

  # continue as usual to get all words and create word features
  all_words_list = [word for (sent,cat) in docs for word in sent]
  all_words = nltk.FreqDist(all_words_list)

  print('Total unique words:', len(all_words))

  # get the 1500 most frequently appearing keywords in the corpus
  word_items = all_words.most_common(1500)
  word_features = [word for (word,count) in word_items]


  

  # feature sets from a feature definition function
  #featuresets = [(document_features(d, word_features), c) for (d, c) in docs]


  # feature sets from a new combined featuresets that include afinn features 
  featuresets = [(combined_sentiment_features(d, word_features, positivelist, neutrallist, negativelist, poslist, neglist, afinn_lex), c) for (d, c) in docs]

  #create test featuresets
  test_featuresets = [(phrase_id, combined_sentiment_features(d, word_features, positivelist, neutrallist, negativelist, poslist, neglist, afinn_lex)) for (phrase_id, d) in test_docs]



  # train classifier and show performance in cross-validation
  # make a list of labels
  label_list = [c for (d,c) in docs]
  labels = list(set(label_list))    # gets only unique labels
  num_folds = 5

  #classifier_type = 'Random Forest'  # change the classifier to Recurrent Neural Network, Multi-layer Perceptron, Random Forest
  #classifier_type = 'Multi-layer Perceptron'  # change the classifier to Recurrent Neural Network, Multi-layer Perceptron, Random Forest
  classifier_type = 'Recurrent Neural Network'  # change the classifier to Recurrent Neural Network, Multi-layer Perceptron, Random Forest
 
  print("Use Classifier: ", classifier_type) #use RandomForest classifier
  cross_validation_PRF(num_folds, featuresets, test_featuresets, labels, classifier_type)

"""
commandline interface takes a directory name with kaggle subdirectory for train.tsv
   and a limit to the number of kaggle phrases to use
It then processes the files and trains a kaggle movie review sentiment classifier.

"""
if __name__ == '__main__':
    if (len(sys.argv) != 3):
        print ('usage: classifyKaggle.py <corpus-dir> <limit>')
        sys.exit(0)
    processkaggle(sys.argv[1], sys.argv[2])