#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# 
# NLP Homework 3
# Chinese event extraction with HMM(Hidden Markov Model)
# Author: Jialun Shen
# Student No.: 16307110030

import codecs
from nltk import FreqDist, ConditionalFreqDist
from nltk import ConditionalProbDist, MLEProbDist
import math


def train(para):
    """train HMM model"""
    # read file
    with codecs.open(para+'_train.txt', 'r', 'utf8') as handler:
        trainlines = handler.readlines()
        
    inititial = FreqDist() # initial distribution P(X_1)
    transition = ConditionalFreqDist() # Transitions P(X_t|X_t-1)
    emission = ConditionalFreqDist() # Emissions P(E_t|X_t)
    all_labels = [] # list of all different labels
    newSent = True # whether a new sentence is observed

    for line in trainlines:
        # Each line contains {word  real_label} or is empty
        if line.strip(): # if line is not empty
            li = line.strip().split()
            word = li[0]
            label = li[1]
            if newSent:
                inititial[label] += 1
                newSent = False
            else:
                transition[label_before][label] += 1
            if label not in all_labels:
                all_labels.append(label)
            label_before = label
            emission[label][word] += 1
        else: # line is empty, next line will be the beginning of a new sentence
            newSent = True
            
    # convert freqdist to probdist, use MLE here
    inititial = MLEProbDist(inititial)
    transition = ConditionalProbDist(transition, MLEProbDist)
    emission = ConditionalProbDist(emission, MLEProbDist)

    model = [inititial, transition, emission, all_labels]
    return model


def test_sent_viterbi(words, model):
    """
    Viterbi Decoding
    M is a Probability lattice, M[(i,j)]: max prob ending with state j at time i
    use log for probability to prevent underflow
    for an unseen label, assign a prob. of pp=1e-50
    """
    pp = 1e-50
    initial, transition, emission, all_labels = model
    labels_pred = []
    Mlst = [] # Mlst[i] is Probability lattice of time i
    backpointerlst = []
    l = len(words)
    assert l >= 1

    # for the first word
    M = dict()
    for lab in all_labels:
        M[lab] = math.log(max(initial.prob(lab), pp))
    current_best = max(M.keys(), key = lambda lab: M[lab])
    Mlst.append(M)
    
    # for other words
    for i in range(1,l):
        word = words[i]
        M = dict()
        backpointer = dict()
        prev_M = Mlst[-1]
        for lab in all_labels:
            # Suppose the current evdience is w, label is j
            # We want a label X_{i-1} = argmax_k M[i-1,k] * P(j|k) * P(w|j)
            # and then M[i,j] = max_k M[i-1,k] * P(j|k) * P(w|j)
            #best_prev = max(prev_M.keys(), key = lambda prev_lab: \
            #    prev_M[prev_lab] * transition[prev_lab].prob(lab) * emission[lab].prob(word))
            #M[lab] = prev_M[best_prev] * transition[best_prev].prob(lab) * emission[lab].prob(word)
            best_prev = max(prev_M.keys(), key = lambda prev_lab: \
                prev_M[prev_lab] + math.log(max(transition[prev_lab].prob(lab), pp)) + math.log(max(emission[lab].prob(word), pp)))
            M[lab] = prev_M[best_prev] + math.log(max(transition[best_prev].prob(lab), pp)) + math.log(max(emission[lab].prob(word), pp))
            backpointer[lab] = best_prev
        current_best = max(M.keys(), key = lambda lab: M[lab])
        Mlst.append(M)
        backpointerlst.append(backpointer)
    
    # end of sentence, backtrack
    labels_pred.append(current_best)
    for backpointer in backpointerlst:
        best_prev = backpointer[current_best]
        labels_pred.append(best_prev)
        current_best = best_prev
    # need to reverse the order
    labels_pred.reverse()

    return labels_pred


def make_lines(words, labels1, labels2):
    newlines = []
    l = len(words)
    l1 = len(labels1)
    l2 = len(labels2)
    assert l == l1 == l2
    for i in range(l):
        newline = "{}\t{}\t{}\n".format(words[i], labels1[i], labels2[i])
        newlines.append(newline)
    return newlines


def test(para, model):
    """test with model and make result.txt"""
    # read file
    with codecs.open(para+'_test.txt', 'r', 'utf8') as handler:
        testlines = handler.readlines()
        
    resultlines = []
    lastEmpty = False # whether the last line is empty
    words = []
    labels_real = []

    for line in testlines:
        # Each line contains {word  real_label} or is empty
        if line.strip(): # if line is not empty
            lastEmpty = False
            li = line.strip().split()
            words.append(li[0])
            labels_real.append(li[1])
        else: # line is empty, next line will be the beginning of a new sentence
            if not lastEmpty:
                labels_pred = test_sent_viterbi(words, model)
                newlines = make_lines(words, labels_real, labels_pred)
                resultlines.extend(newlines)
            lastEmpty = True
            words = []
            labels_real = []
            resultlines.append(line)

    with codecs.open(para+'_result.txt', 'w', 'utf8') as handler:
        handler.writelines(resultlines)


if __name__ == "__main__":
    lst = ['trigger', 'argument']
    for c in lst:
        #initial, transition, emission, all_labels = train(c)
        #print(all_labels)
        #minlab = min(initial.freqdist().keys(), key = lambda lab: initial.prob(lab))
        #print("initial min: {}, {}".format(minlab, initial.prob(minlab)))
        test(c, train(c))