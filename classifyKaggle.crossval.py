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
from nltk.collocations import *
from sklearn.ensemble import RandomForestClassifier
from nltk import NaiveBayesClassifier

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

#define featuresets using unigrams and negations
def NOT_features(document, word_features, negationwords):
    features = {}
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)
    return features

#define featuresets using subjectivity lexicon 
def SL_features(document, sl_positivelist, sl_neutrallist, sl_negativelist):
    document_words = set(document)
    features = {}
        # Count the number of positive and negative words in the document
    pos_count = sum(1 for word in document_words if word in sl_positivelist)
    neu_count = sum(1 for word in document_words if word in sl_neutrallist)
    neg_count = sum(1 for word in document_words if word in sl_negativelist)

    # Add sl sentiment counts as features
    features['sl_Positive_Count'] = pos_count
    features['sl_Negative_Count'] = neg_count
    features['sl_Neutral_Count'] = neu_count
    return features


#define featuresets using LIWC
def liwc_features(document, liwc_poslist, liwc_neglist):
    document_words = set(document)
    features = {}
   
    # Count the number of positive and negative words in the document
    pos_count = sum(1 for word in document_words if word in liwc_poslist)
    neg_count = sum(1 for word in document_words if word in liwc_neglist)

    # Add presence using prefix-based matching
    pos_present = any(sentiment_read_LIWC_pos_neg_words.isPresent(word, liwc_poslist) for word in document_words)
    neg_present = any(sentiment_read_LIWC_pos_neg_words.isPresent(word, liwc_neglist) for word in document_words)

    # Add LIWC sentiment counts as features
    features['LIWC_Positive_Count'] = pos_count
    features['LIWC_Negative_Count'] = neg_count
    features['LIWC_Positive_Present'] = pos_present
    features['LIWC_Negative_Present'] = neg_present 
    return features



# define a new combined featuresets that also includes afinn features
def unigram_negation_sentiment_features(document, word_features, negationwords, sl_positivelist, sl_neutrallist, sl_negativelist,liwc_poslist, liwc_neglist):
    document_words = set(document)
    features = {}

    #unigram features and negations
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)

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

    return features




#define combined featuresets using unigrams, bigrams and sentiment word counts all in one feature set
def combined_features(document, word_features,negationwords, bigram_features,sl_positivelist, sl_neutrallist, sl_negativelist,liwc_poslist, liwc_neglist):

    document_words = set(document)
    document_bigrams = nltk.bigrams(document)
    features = {}

    #unigram features and negations
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False
    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)

    #bigram features
    for bigram in bigram_features:
        features['B_{}_{}'.format(bigram[0], bigram[1])] = (bigram in document_bigrams)  

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

    return features

# define a new combined featuresets that also includes afinn features
def combined_sentiment_features(document, word_features,negationwords, sl_positivelist, sl_neutrallist, sl_negativelist,liwc_poslist, liwc_neglist, afinn_lex):
    document_words = set(document)
    features = {}

    #unigram features and negations
    for word in word_features:
        features['V_{}'.format(word)] = False
        features['V_NOT{}'.format(word)] = False

    # go through document words in order
    for i in range(0, len(document)):
        word = document[i]
        if ((i + 1) < len(document)) and ((word in negationwords) or (word.endswith("n't"))):
            i += 1
            features['V_NOT{}'.format(document[i])] = (document[i] in word_features)
        else:
            features['V_{}'.format(word)] = (word in word_features)

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
def cross_validation_PRF(num_folds, featuresets, labels):
    subset_size = int(len(featuresets)/num_folds)
    print('Each fold size:', subset_size)
    # for the number of labels - start the totals lists with zeroes
    num_labels = len(labels)
    total_precision_list = [0] * num_labels
    total_recall_list = [0] * num_labels
    total_F1_list = [0] * num_labels

    # iterate over the folds
    for i in range(num_folds):
        test_this_round = featuresets[(i*subset_size):][:subset_size]
        train_this_round = featuresets[:(i*subset_size)] + featuresets[((i+1)*subset_size):]
        # train using train_this_round

        classifier = NaiveBayesClassifier.train(train_this_round)
        
        # evaluate against test_this_round to produce the gold and predicted labels
        goldlist = []
        predictedlist = []
        for (features, label) in test_this_round:
            goldlist.append(label)
            predictedlist.append(classifier.classify(features))

        # computes evaluation measures for this fold and
        #   returns list of measures for each label
        #print('Fold', i)
        (precision_list, recall_list, F1_list) \
                  = eval_measures(goldlist, predictedlist, labels)
        # take off triple string to print precision, recall and F1 for each fold
        '''
        print('\tPrecision\tRecall\t\tF1')
        # print measures for each label
        for i, lab in enumerate(labels):
            print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
              "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
        '''
        # for each label add to the sums in the total lists
        for i in range(num_labels):
            # for each label, add the 3 measures to the 3 lists of totals
            total_precision_list[i] += precision_list[i]
            total_recall_list[i] += recall_list[i]
            total_F1_list[i] += F1_list[i]

    # find precision, recall and F measure averaged over all rounds for all labels
    # compute averages from the totals lists
    precision_list = [tot/num_folds for tot in total_precision_list]
    recall_list = [tot/num_folds for tot in total_recall_list]
    F1_list = [tot/num_folds for tot in total_F1_list]
    # the evaluation measures in a table with one row per label
    print('\nAverage Precision\tRecall\t\tF1 \tPer Label')
    # print measures for each label
    for i, lab in enumerate(labels):
        print(lab, '\t', "{:10.3f}".format(precision_list[i]), \
          "{:10.3f}".format(recall_list[i]), "{:10.3f}".format(F1_list[i]))
    
    # print macro average over all labels - treats each label equally
    print('\nMacro Average Precision\tRecall\t\tF1 \tOver All Labels')
    print('\t', "{:10.3f}".format(sum(precision_list)/num_labels), \
          "{:10.3f}".format(sum(recall_list)/num_labels), \
          "{:10.3f}".format(sum(F1_list)/num_labels))

    # for micro averaging, weight the scores for each label by the number of items
    #    this is better for labels with imbalance
    # first intialize a dictionary for label counts and then count them
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
    
    print("Show 10 most informative features: \n")
    print(classifier.show_most_informative_features(10))

# Function to compute precision, recall and F1 for each label
#  and for any number of labels
# Input: list of gold labels, list of predicted labels (in same order)
# Output: returns lists of precision, recall and F1 for each label
#      (for computing averages across folds and labels)
def eval_measures(gold, predicted, labels):
    
    # these lists have values for each label 
    recall_list = []
    precision_list = []
    F1_list = []

    for lab in labels:
        # for each label, compare gold and predicted lists and compute values
        TP = FP = FN = TN = 0
        for i, val in enumerate(gold):
            if val == lab and predicted[i] == lab:  TP += 1
            if val == lab and predicted[i] != lab:  FN += 1
            if val != lab and predicted[i] == lab:  FP += 1
            if val != lab and predicted[i] != lab:  TN += 1
        # use these to compute recall, precision, F1
        # for small numbers, guard against dividing by zero in computing measures
        if (TP == 0) or (FP == 0) or (FN == 0):
          recall_list.append (0)
          precision_list.append (0)
          F1_list.append(0)
        else:
          recall = TP / (TP + FP)
          precision = TP / (TP + FN)
          recall_list.append(recall)
          precision_list.append(precision)
          F1_list.append( 2 * (recall * precision) / (recall + precision))

    # the evaluation measures in a table with one row per label
    return (precision_list, recall_list, F1_list)

## function to read kaggle training file, train and test a classifier 
def processkaggle(dirPath,limitStr):
  # convert the limit argument from a string to an int
  limit = int(limitStr)
  
  os.chdir(dirPath)
  
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


  ## Define additional unnecessary words
  additional_words = {
        "'s", "'re", "'ve", "'d", "also", "thing", "maybe", 
        "would", "could", "should", "might", "must", "lot", "etc", "ok", "okay", "oh", "uh", "um"}
  
  # Define negation words
  negation_words = [
    'no', 'not', 'never', 'none', 'nowhere', 'nothing', 'noone', 
    'rather', 'hardly', 'scarcely', 'rarely', 'seldom', 'neither', 'nor'
  ]

  old_stop_words = set(stopwords.words('english'))

  stop_words = old_stop_words -set(negation_words)
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

  # continue as usual to get all words and create word features
  all_words_list = [word for (sent,cat) in docs for word in sent]
  all_words = nltk.FreqDist(all_words_list)

  print('Total unique words:', len(all_words))

  # get the 1500 most frequently appearing keywords in the corpus
  word_items = all_words.most_common(5000)
  word_features = [word for (word,count) in word_items]

  bigram_measures = nltk.collocations.BigramAssocMeasures()
  # create the bigram finder on all the words in sequence
  finder = nltk.BigramCollocationFinder.from_words(all_words_list)
  # define the top 500 bigrams using the chi squared measure
  bigram_features = finder.nbest(bigram_measures.chi_sq, 500)

  # feature sets from a feature definition function
  #featuresets = [(document_features(d, word_features), c) for (d, c) in docs]

  # feature sets from a NOT_feature definition function using unigram and negations
  #featuresets = [(NOT_features(d, word_features,negation_words), c) for (d, c) in docs]

  # feature sets from a sl feature definition function
  #featuresets = [(SL_features(d, positivelist, neutrallist, negativelist), c) for (d, c) in docs]

  # feature sets from a liwc feature definition function
  #featuresets = [(liwc_features(d, poslist, neglist), c) for (d, c) in docs]

  # feature sets from unigrams, negations, and sentiments such as LIWC and Subjectivity
  #featuresets = [(unigram_negation_sentiment_features(d, word_features, negation_words, positivelist, neutrallist, negativelist, poslist, neglist), c) for (d, c) in docs]

  # feature sets from combined featuresets using unigrams, bigrams, SL sentiment, LIWC sentiment all in one feature set
  #featuresets = [(combined_features(d, word_features, negation_words, bigram_features, positivelist, neutrallist, negativelist, poslist, neglist), c) for (d, c) in docs]

  # feature sets from a new combined featuresets that include afinn features 
  #featuresets = [(combined_sentiment_features(d, word_features,negation_words, positivelist, neutrallist, negativelist, poslist, neglist, afinn_lex), c) for (d, c) in docs]


  # train classifier and show performance in cross-validation
  # make a list of labels
  label_list = [c for (d,c) in docs]
  labels = list(set(label_list))    # gets only unique labels
  num_folds = 5
  

  classifier_type = "Naive Bayes"
  print("Use Classifier: ", classifier_type) #use RandomForest classifier
  cross_validation_PRF(num_folds, featuresets, labels)

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