from nltk import induce_pcfg, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.grammar import Nonterminal, standard_nonterm_parser
from ViterbiClass import ViterbiProbParser, contextProbDict
from PYEVALB.scorer import Scorer
from PYEVALB.summary import summary
import random
import numpy as np
import pickle


def testingBench(iterations, count, ruleCount, trainParse, trainRaw, goldParse, rawSentences, wordDict):
    finalSummaries = []
    tables = []

    for i in range(iterations):
        summaryTable = {}
        trainSentences = []
        contextTrain = []
        goldSentences = []
        testSentences = []
        while len(trainSentences) < ruleCount:
            newNumber = random.choice(range(0, len(trainParse)))
            if trainParse[newNumber][1] == 'S':
                trainSentences.append(trainParse[newNumber])
                contextTrain.append(trainRaw[newNumber].lower())
        while len(goldSentences) < count:
            newNumber = random.choice(range(0, len(goldParse)))
            if goldParse[newNumber][1] == 'S':
                goldSentences.append(goldParse[newNumber])
                testSentences.append(rawSentences[newNumber])
        trees = [Tree.fromstring(sent) for sent in trainSentences]
        productions = []
        for tree in trees:
            productions += tree.productions()
        start = Nonterminal('S')
        grammar = induce_pcfg(start, productions)
        rules = []
        for production in grammar.productions():
            lhs, rhs, prob = production._lhs, production._rhs, production._ProbabilisticMixIn__prob
            rhsList = [str(term) if type(term) != str else term for term in rhs]
            rules.append((str(lhs), rhsList, prob))
        PoS = {}
        posWords = pos_tag(wordDict)
        for word, part in posWords:
            if part not in PoS:
                PoS[part] = set(word)
            else:
                PoS[part].add(word)
        rulesPoS = []
        for part, words in PoS.items():
            for word in words:
                rulesPoS.append((part, [word], 1/len(words)))
        rules.extend(rulesPoS)
        with open('contextDict.pickle', 'rb') as handle:
            probD = pickle.load(handle)
        parserProb = ViterbiProbParser(rules, probD)
        parser = ViterbiProbParser(rules)
        gold = []
        test = []
        testP = []
        for i in range(len(testSentences)):
            sentParse = parser.parse(word_tokenize(testSentences[i].lower()))
            sentProbParse = parserProb.parse(word_tokenize(testSentences[i].lower()))
            print(testSentences[i], sentProbParse)
            if sentProbParse is not None:
                lowerCopy = ''
                for term in goldSentences[i].split(' '):
                    if term[0].isupper() and term[1].islower() or not term[1].isalpha():
                        lowerCopy += term.lower()
                    else:
                        lowerCopy += term
                    lowerCopy += ' '
                summaryTable[testSentences[i]] = {'Gold': lowerCopy, 'NoProb': (sentParse, sentParse.prob),
                                                  'Prob': (sentProbParse, sentProbParse.prob)}
                gold.append(lowerCopy)
                test.append(str(sentParse))
                testP.append(str(sentProbParse))

        myScorer = Scorer()
        results = myScorer.score_corpus(gold, test)
        resultsP = myScorer.score_corpus(gold, testP)
        finalSummaries.append((summary(results), summary(resultsP)))
        tables.append(summaryTable)

    recall = 0
    precision = 0
    fmeasure = 0
    accuracy = 0
    recallP = 0
    precisionP = 0
    fmeasureP = 0
    accuracyP = 0
    for nonP, P in finalSummaries:
        print(nonP)
        print(P)
        recall += nonP[4]
        precision += nonP[5]
        fmeasure += nonP[6]
        accuracy += nonP[10]
        recallP += P[4]
        precisionP += P[5]
        fmeasureP += P[6]
        accuracyP += P[10]
    print('Recall: ', recall / iterations, recallP / iterations)
    print('Precision: ', precision / iterations, precisionP / iterations)
    print('F-Measure: ', fmeasure / iterations, fmeasureP / iterations)
    print('Accuracy: ', accuracy / iterations, accuracyP / iterations)

    parseStats = []
    labels = ['Correct', 'BothWrong', 'NormalWrong', 'ProbWrong']
    for table in tables:
        correct = set()
        bothWrong = set()
        normalWrong = set()
        probWrong = set()
        for sent, answer in table.items():
            answerTuple = (sent, answer['NoProb'][1], answer['Prob'][1])
            print(answer['Prob'][0] == answer['Gold'], answer['NoProb'][0] == answer['Gold'])
            if answer['Prob'][0] == answer['Gold'] and answer['NoProb'][0] == answer['Gold']:
                correct.add(answerTuple)
            if answer['Prob'][0] != answer['Gold'] and answer['NoProb'][0] == answer['Gold']:
                probWrong.add(answerTuple)
            if answer['Prob'][0] == answer['Gold'] and answer['NoProb'][0] != answer['Gold']:
                normalWrong.add(answerTuple)
            if answer['Prob'][0] != answer['Gold'] and answer['NoProb'][0] != answer['Gold']:
                bothWrong.add(answerTuple)
            print(str(answer['Prob'][0]), answer['Prob'][1])
            print(answer['Gold'])
            print(str(answer['NoProb'][0]), answer['NoProb'][1])

        ansSets = [correct, bothWrong, normalWrong, probWrong]
        parseStats.append(ansSets)
    for ansSet in parseStats:
        print('-' * 30)
        for l in range(len(labels)):
            print(labels[l], ': ', len(ansSet[l]))
            if len(ansSet[l]) != 0:
                print('Normal: ', sum([ans[1] if ans is not np.nan else 0 for ans in ansSet[l]]) / len(ansSet[l]))
                print('Context: ', sum([ans[2] if ans is not np.nan else 0 for ans in ansSet[l]]) / len(ansSet[l]))


train = open('./ptb-data/ptb-train-10.txt')
trainRaw = open('./ptb-data/ptb-train-raw-10.txt')
goldParse = open('./ptb-data/ptb-test-10.txt')
testSent = open('./ptb-data/ptb-test-raw-10.txt')
wordDict = open('/home/michael/9.66Final/ptb-data/ptb.dict')

grammarSent = [sent.replace('\n', '') for sent in train]
grammarRaw = [sent.replace('\n', '') for sent in trainRaw]
goldParseSent = [sent.replace('\n', '') for sent in goldParse]
rawSentences = [sent.replace('\n', '') for sent in testSent]
words = [i.split(' ')[0] for i in wordDict]
ans = testingBench(1, 5, 1000, grammarSent, grammarRaw, goldParseSent, rawSentences, words)
