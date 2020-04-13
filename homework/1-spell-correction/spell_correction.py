'''
NLP Homework 1
Jialun Shen 16307110030
School of Data Science, Fudan University
Oct.17, 2019
latest:Oct.20 20:49
result20 373.0920 sec 90.7%
result22 374.7691 sec 90.9% missing 0.1%?
'''

# 0 Import packages
import math
import nltk
import string
import time
from nltk import bigrams, trigrams, word_tokenize, FreqDist
from nltk.corpus import reuters, brown
from collections import Counter
from nltk.tokenize.treebank import TreebankWordDetokenizer

# 1 Build a language model
'''
Use the reuters corpus in nltk (since it is based on news), and news in brown corpus
'''
words = reuters.words() + brown.words(categories="news")
N = len(words)
words_set = set(words)
V = len(words_set)
words2 = list(bigrams(words)) #list of (w1,w2)
words3 = list(trigrams(words)) #list of (w1,w2,w3)
words_freq_uni = FreqDist(words) #count(w)
words_freq_bi = FreqDist(words2) #count(w1,w2)
words_freq_tri = FreqDist(words3) #count(w1,w2,w3)
prob_uni = dict() #Unigram probability P(w)
prob_bi = dict() #Bigram probability P(wi|wi-1), raw
prob_bia1 = dict() #Bigram probability P(wi|wi-1), add-1 smoothing
prob_tri = dict() #Trigram probability P(wi|wi-2wi-1)

#Unigram
for key in words_freq_uni.keys():
    prob_uni[key] = words_freq_uni[key] / N
    #P(w) = count(w) / N
def Unigram_prob(word):
    '''
    get unigram probabiliy of word
    '''
    if word in prob_uni.keys():
        return prob_uni[word]
    else:
        return 0

#Bigram
for key in words_freq_bi.keys():
    w1 = key[0]
    w2 = key[1]
    prob_bi[(w2, w1)] = words_freq_bi[key] / words_freq_uni[w1]
    #P(wi|wi-1) = count(wi-1,wi) / count(wi-1)
    prob_bia1[(w2, w1)] = (words_freq_bi[key]+1) / (words_freq_uni[w1]+V)
    #P(wi|wi-1) = (count(wi-1,wi)+1) / (count(wi-1)+V)

def Bigram_prob(word2, word1):
    '''
    get MLE bigram probabiliy of word2|word1
    '''
    if (word2,word1) in prob_bi.keys():
        return prob_bi[(word2,word1)]
    else:
        return 0

def Bigram_prob_add1(word2, word1):
    '''
    get add-i smooth bigram probabiliy of word2|word1
    '''
    if (word2,word1) in prob_bia1.keys():
        return prob_bia1[(word2,word1)]
    else:
        return 1 / (words_freq_uni[word1]+V)

bigram_set = set(words_freq_bi.keys())
LL = len(bigram_set) #the total number of word bigram types

def continuation(word, fore=False):
    '''
    given a word
    if fore = False(default)
    return the list of all continuations of (xxx,word) in the training set
    if fore = True
    return return the list of all continuations of (word,xxx) in the training set
    '''
    if fore:
        continuations = [pair for pair in bigram_set if pair[0] == word]
    else:
        continuations = [pair for pair in bigram_set if pair[1] == word]
    return continuations

def Bigram_prob_KN(word2, word1, d=0.75):
    '''
    Kneser-Ney smoothing bigram probability
    P(w2|w1) = max(count(w1w2)-d,0)/count(w1) + k(w1)P_continuation(w2)
    where k(w2) = d*|#of pairs (w1,xxx)|/c(w1)
    P_continuation(w2) = |#of pairs (xxx,w2)|/|#of all pairs|
    '''
    cw1 = words_freq_uni[word1] #0 if count=0
    if cw1:
        k = d*len(continuation(word1, fore=True)) / cw1
        pc = len(continuation(word2)) / LL
        p = max(words_freq_bi[(word1,word2)]-d, 0) / cw1 + k * pc
        return p
    else:
        return 0

#Trigram
for key in words_freq_tri.keys():
    w1 = key[0]
    w2 = key[1]
    w3 = key[2]
    prob_tri[(w3, w1, w2)] = words_freq_tri[key] / words_freq_bi[(w1, w2)]
    #P(wi|wi-2,wi-1) = count(wi-2,wi-1,wi) / count(wi-2,wi-1)
def Trigram_prob(word3, word1, word2):
    '''
    get trigram probabiliy of word3|word1 word2
    '''
    if (word3, word1, word2) in prob_tri.keys():
        return prob_tri[(word3,word1,word2)]
    else:
        return 0

# count single letter and 2-letter combinations
def word2biletter(word):
    '''
    Input: a word, e.g. 'apple'
    Output: a list of all 2-word combinations, including the beginning of word
    e.g. ['a','ap','pp','pl','le']
    '''
    l = len(word)
    if l == 0:
        return []
    elif l == 1:
        return [word]
    else:
        lst = [word[i:i+2] for i in range(l-1)]
        lst.append(word[0])
        return lst

count1 = Counter() #Counter for single letter, including begin of word ''
count2 = Counter() #Counter for 2-letter combinations
c_head = Counter([''])
for word in words:
    if word in string.punctuation: #do not count single punctuation
        continue
    count1.update(c_head)
    word = word.lower()
    c1 = Counter(word)
    count1.update(c1)
    if len(word) >= 2:
        word2 = word2biletter(word)
        c2 = Counter(word2)
        count2.update(c2)

# 2 Build a channel model
'''
Use https://norvig.com/ngrams/count_1edit.txt
There is some encoding problems with the txt file, 
so I made a copy directly from the website. 
'''
channelpath = './count_1edit.txt'
with open(channelpath, 'r') as handler:
    count_1edit = handler.readlines()
    #count_1edit[i] is now like 'x|y\t123\n', where x or y may be '' or include ' '
    count_1edit = [edit.rstrip().split('\t') for edit in count_1edit]
    #count_1edit[i] is now like ['x|y',123']
    count_1edit = [[edit[0].split('|'), int(edit[1])] for edit in count_1edit]
    #count_1edit[i] is now in the form of [[x(str), y(str)], t(int)]
    #l = [i for i in count_1edit if len(i[0][1]) > 2 or len(i[0][1]) > 2]
    #l == [] #is True, so all xy's are shorter than 2

cm_del = dict() #channel model: deletion
cm_ins = dict() #channel model: insertion
cm_sub = dict() #channel model: substitution
cm_trans = dict() #channel model: transposition
'''
count of wi and wi-1wi comes from the language model
Thus p(x|w) depends on the size of the LM, but it does not matter,
since we just need a relative probability to get candidates.
'''
for lst in count_1edit:
    #incorrect|correct count
    w1 = lst[0][0] #incorrect letter(s)
    w2 = lst[0][1] #correct letter(s)
    l1 = len(w1) #0-2
    l2 = len(w2) #0-2
    t = lst[1]
    if l1 < l2:
        #deleiton
        #del(x,y) = count(xy(correct) typed as x(incorrect))
        x = w1
        if w1 == '':
            y = w2
        else: #w1 is e.g. 'a'
            y = w2[1]
        cm_del[(x, y)] = t
    elif l1 > l2:
        #insertion
        #ins(x,y) = count(x(correct) typed as xy(incorrect))
        x = w2
        if x == '':
            y = w1
        else:
            y = w1[1]
        cm_ins[(x, y)] = t
    else:
        if l1 < 2:
            #substitution
            #sub(x,y) = count(y(correct) typed as x(incorrect))
            x = w1
            y = w2
            cm_sub[(x, y)] = t
        else: #l1==2
            #transposition
            #trans(x,y) = count(xy(correct) typed as yx(incorrect))
            y = w2[1] #=w1[0]
            x = w2[0] #=w1[1]
            cm_trans[(x, y)] = t

# 3 Implementation of spell correction
A = string.ascii_letters
a = string.ascii_lowercase

def edit1(word, alphabet = A):
    '''
    given a word
    return a set of all possible words 1-edit away
    insersion of a letter from alphabet is possible
    '''
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    substitutes = [a + c + b[1:] for a, b in splits for c in alphabet if b]
    inserts = [a + c + b for a, b in splits for c in alphabet]
    return set(deletes + transposes + substitutes + inserts)

def known(words, vocabulary):
    '''
    Input:
    a set of words
    Output:
    a set of words in vocabulary
    '''
    return set(w for w in words if w in vocabulary)

def kind_of_edit1(wordx, wordw):
    '''
    Input: 
    wordx, wordw 1 edit aeay
    (Suppose wordx is wrong)
    Output: 
    kind of edit (deletion, insertion, substitution, transposition),
    and edit letters (x,y)
    '''
    lx = len(wordx)
    lw = len(wordw)
    if lx < lw: 
        #deletion of y after x, xy(correct) typed as x(incorrect)
        e = 'del'
        if wordx[0] != wordw[0]: #deletion of first letter
            x = ''
            y = wordw[0]
        elif wordx == wordw[:-1]: #deletion of last letter
            x = wordx[-1]
            y = wordw[-1]
        else:
            for i in range(1, lx):
                if wordx[i] != wordw[i]:
                    x = wordw[i-1]
                    y = wordw[i]
                    break
    elif lx > lw:
        #insertion of y after x, x(correct) typed as xy(incorrect)
        e = 'ins'
        if wordx[0] != wordw[0]: #insertion of first letter
            x = ''
            y = wordx[0]
        elif wordw == wordx[:-1]: #insertion of last letter
            x = wordw[-1]
            y = wordx[-1]
        else:
            for i in range(1, lw):
                if wordx[i] != wordw[i]:
                    x = wordx[i-1]
                    y = wordx[i]
                    break
    else:
        if set(wordx) == set(wordw):
            #transposition, xy(correct) typed as yx(incorrect)
            e = 'trans'
            for i in range(lx):
                if wordx[i] != wordw[i]:
                    x = wordw[i]
                    y = wordx[i]
                    break
        else:
            #substitution, y(correct) typed as x(incorrect)
            e = 'sub'
            for i in range(lx):
                if wordx[i] != wordw[i]:
                    x = wordx[i]
                    y = wordw[i]
                    break
    #change x,y into lowercase
    x = x.lower()
    y = y.lower()
    return e, x, y

def CM_prob(wordx, wordw, p=0.95):
    '''
    Input:
    wordx, wordw(candidate)
    p = P(w|w), probability of no typo, defaut 0.95 (1 error in 20 words)
    Output:
    P(x|w)
    '''
    if wordx == wordw:
        return p
    else:
        e, x, y = kind_of_edit1(wordx, wordw)
        xy = x + y
        if e == 'del':
            if (x,y) not in cm_del.keys():
                return 0
            return cm_del[(x,y)] / count2[xy]
        elif e == 'ins':
            if (x,y) not in cm_ins.keys():
                return 0
            return cm_ins[(x,y)] / count1[x]
        elif e == 'sub':
            if (x,y) not in cm_sub.keys():
                return 0
            return cm_sub[(x,y)] / count1[y]
        elif e == 'trans':
            if (x,y) not in cm_trans.keys():
                return 0
            return cm_trans[(x,y)] / count2[xy]
        else:
            return 0

def best_candidate_uni(word, vocabulary):
    '''
    get the best candidate of word from vocabulary
    use Unigram model
    '''
    c_prob = dict()
    candidate1 = known(edit1(word), vocabulary)
    if candidate1:
        candidate1 = known([word], vocabulary) | candidate1
        for c in candidate1:
            c_prob[c] = Unigram_prob(c)
    else:
        for c in vocabulary:
            if set(c) == set(word):
                c_prob[c] = Unigram_prob(c)
    if not c_prob:
        return word
    return max(c_prob, key=c_prob.get)

def best_candidate_bi(word, vocabulary, front_word=None, smooth=None, p=0.95):
    '''
    get the best candidate of word from vocabulary, given its front word(if applicable)
    use Bigram model with smooth in [None, 'a1', 'KN']
    '''
    if smooth == 'a1':
        bp = Bigram_prob_add1
    elif smooth == 'KN':
        bp = Bigram_prob_KN
    else:
        bp = Bigram_prob
    c_prob = dict()
    candidate1 = known(edit1(word), vocabulary)
    if candidate1:
        candidate1 = known([word], vocabulary) | candidate1
        for c in candidate1:
            if front_word:
                c_prob[c] = CM_prob(word, c, p) * bp(c, front_word)
            else:
                c_prob[c] = CM_prob(word, c, p) * Unigram_prob(c)
    else: #no valid candidates within 1 edit
        for c in vocabulary:
            if set(c) == set(word): #a way based on this problem
                if front_word:
                    c_prob[c] = bp(c, front_word)
                else:
                    c_prob[c] = Unigram_prob(c)
    if not c_prob: #still cannot find any candidate, give up
        return word
    return max(c_prob, key=c_prob.get)

def lst2sent(lst):
    '''
    Inupt:
    a list of words and punctuaions ['I','can','fly','.']
    (from nltk.word_tokenize())
    Output:
    an untokenized sentence
    '''
    return TreebankWordDetokenizer().detokenize(lst)

def best_candidate_real_word(word, vocabulary, front_words, p=0.95):
    '''
    this function is for real word correction
    word is in vocabulary
    '''
    c_prob = dict()
    #candi = candidates(word, vocabulary) #diff in prob for edit dist. 1 and 2, not comparable
    candi = known([word], vocabulary) | known(edit1(word, alphabet=a), vocabulary)
    w1 = front_words[0]
    w2 = front_words[1]
    for c in candi:
        if w1 and w2:
            c_prob[c] = CM_prob(word, c, p) * Trigram_prob(c, w1, w2)
        elif w2:
            c_prob[c] = CM_prob(word, c, p) * Bigram_prob_add1(c, w2)
        else:
            c_prob[c] = CM_prob(word, c, p) * Unigram_prob(c)
    if not c_prob:
        return word
    return max(c_prob, key=c_prob.get)

def best_candidate_tri(word, vocabulary, front_words, p=0.95):
    c_prob = dict()
    candidate1 = known(edit1(word, alphabet=a), vocabulary)
    w1 = front_words[0]
    w2 = front_words[1]
    if candidate1:
        candidate1 = known([word], vocabulary) | candidate1
        for c in candidate1:
            if w1 and w2:
                c_prob[c] = CM_prob(word, c, p) * Trigram_prob(c, w1, w2)
            elif w2:
                c_prob[c] = CM_prob(word, c, p) * Bigram_prob(c, w2)
            else:
                c_prob[c] = CM_prob(word, c, p) * Unigram_prob(c)
    else:
        for c in vocabulary:
            if set(c) == set(word): #a way based on this problem
                if w1 and w2:
                    c_prob[c] = Trigram_prob(c, w1, w2)
                elif w2:
                    c_prob[c] = Bigram_prob(c, w2)
                else:
                    c_prob[c] = Unigram_prob(c)
    if not c_prob:
        return word
    return max(c_prob, key=c_prob.get)

def nonword_correction(sent, vocabulary, smooth=None):
    '''
    spell correction step 1
    Input:
    (untokenized) sentence to be corrected, vocabulary
    Output:
    (untokenized) non-word error corrected sentence
    '''
    answer = []
    sent = word_tokenize(sent)
    i = 0 #record the current position in sent
    front_word = None #the word before
    front_ispunc = False #whether the word before is punctuaion
    numset = set(string.digits) | set(string.punctuation)
    for word in sent:
        if i>0:
            front_word = answer[i-1]
        if set(word) & numset:
            answer.append(word)
            front_ispunc = True
        else:
            if word not in vocabulary:
                if front_ispunc:
                    front_word = None
                correct = best_candidate_bi(word, vocabulary, front_word, smooth)
                answer.append(correct)
            else: #maybe real word error
                answer.append(word)
            front_ispunc = False
        i += 1
    #return answer
    return lst2sent(answer)

def nonword_tri(sent, vocabulary):
    '''
    non word spell correction using Trigram
    Input:
    (untokenized) sentence to be corrected, vocabulary
    Output:
    (untokenized) non-word error corrected sentence
    '''
    answer = []
    sent = word_tokenize(sent)
    i = 0 #record the current position in sent
    front_words = [None, None] #the word before
    numset = set(string.digits) | set(string.punctuation)
    for word in sent:
        if i>=1:
            front_words[0] = front_words[1]
            front_words[1] = answer[i-1]
        if set(word) & numset:
            answer.append(word)
        else:
            if word not in vocabulary:
                correct = best_candidate_tri(word, vocabulary, front_words)
                answer.append(correct)
            else: #maybe real word error
                #correct = best_candidate_real_word(word, vocabulary, front_words, p)
                answer.append(word)
        i += 1
    return lst2sent(answer)

def nonword_uni(sent, vocabulary):
    '''
    non word spell correction using Unigram
    Input:
    (untokenized) sentence to be corrected, vocabulary
    Output:
    (untokenized) non-word error corrected sentence
    '''
    answer = []
    sent = word_tokenize(sent)
    for word in sent:
        if word not in vocabulary:
            correct = best_candidate_uni(word, vocabulary)
            answer.append(correct)
        else:answer.append(word)
    return lst2sent(answer)

def spell_correction(sent, vocabulary, smooth=None, p=0.95):
    '''
    spell correction: non-word and real-word error
    Input:
    (untokenized) sentence to be corrected, vocabulary
    Output:
    (untokenized) corrected sentence
    '''
    answer = []
    sent = word_tokenize(sent)
    i = 0 #record the current position in sent
    front_words = [None, None] #the word before
    numset = set(string.digits) | set(string.punctuation)
    for word in sent:
        if i>=1:
            front_words[0] = front_words[1]
            front_words[1] = answer[i-1]
        if set(word) & numset:
            answer.append(word)
        else:
            if word not in vocabulary:
                correct = best_candidate_bi(word, vocabulary, front_words[1], smooth, p)
                answer.append(correct)
            else: #maybe real word error
                correct = best_candidate_real_word(word, vocabulary, front_words, p)
                answer.append(correct)
        i += 1
    return lst2sent(answer)

# 4 Test
def test(n):
    if n>1000 or n<=0:
        n=1000
    testpath = './testdata.txt'
    resultpath = './result.txt'
    vocabpath = './vocab.txt'
    #get all words in vocabulary
    with open(vocabpath, 'r') as handler:
        vocabulary = handler.readlines()
        vocabulary = [w.rstrip() for w in vocabulary] #remove "\r\n" at the right side of each line
    #get list of test sentences
    test_lst = []
    with open(testpath, 'r') as handler:
        for i in range(n):
            testcase = handler.readline().split('\t')[2]
            testcase = testcase.rstrip() #remove the last '\n'
            test_lst.append(testcase)
    #correct each sentence and write into result file
    with open(resultpath, 'w') as handler:
        i = 1
        for testcase in test_lst:
            #answer = spell_correction(testcase, vocabulary) #bi raw non+real
            answer = nonword_correction(testcase, vocabulary) #bi raw non
            #answer = nonword_correction(testcase, vocabulary, 'a1') #bi add1 non
            #answer = nonword_correction(testcase, vocabulary, 'KN') #bi raw non
            #answer = nonword_uni(testcase, vocabulary) #uni non
            #answer = nonword_tri(testcase, vocabulary) #tri non
            line = "{}\t{}\n".format(i, answer)
            handler.write(line)
            #print(i)
            i += 1

if __name__ == '__main__':
    n = 1000
    #start_time = time.time()
    test(n)
    #end_time = time.time()
    #print('Correction finished. Time elapsed = %.4f sec'%(end_time - start_time))
