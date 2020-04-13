# Sentiment analysis with feature engineering based method
# using naïve bayes classifier with bag of words features
# Author: Jialun Shen
# Student No.: 16307110030

import nltk
from data_utils import *

# Load the dataset
dataset = StanfordSentiment()
tokens = dataset.tokens()
nWords = len(tokens)
allWords = list(tokens.keys())

# Define bag of words feature
def feature(words):
    return nltk.FreqDist(words)

def featureSet(trainset):
    return [(feature(words), c) for (words, c) in trainset]

# Load the train set and train
trainset = dataset.getTrainSentences()
featureTrainset = featureSet(trainset)
classifier = nltk.NaiveBayesClassifier.train(featureTrainset)
accTrain = nltk.classify.accuracy(classifier, featureTrainset) * 100

# Prepare dev set features
devset = dataset.getDevSentences()
featureDevset = featureSet(devset)
accDev = nltk.classify.accuracy(classifier, featureDevset) * 100

# Test your findings on the test set
testset = dataset.getTestSentences()
featureTestset = featureSet(testset)
accTest = nltk.classify.accuracy(classifier, featureTestset) * 100

print("=== Naive Bayes Accuracy ===")
print("Train accuracy (%%): %f" % accTrain)
print("Dev accuracy (%%): %f" % accDev)
print("Test accuracy (%%): %f" % accTest)
print()
print(classifier.show_most_informative_features(20))
#print(nWords)