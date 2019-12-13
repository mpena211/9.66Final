from nltk import induce_pcfg
from nltk.tree import Tree
from nltk.grammar import Nonterminal
from nltk import word_tokenize
from ViterbiClass import ViterbiProbParser, contextProbDict
from PYEVALB.scorer import Scorer
from PYEVALB.summary import summary
import random


file = open('./ptb-data/ptb-train-10.txt')
test = open('./ptb-data/ptb-train-raw-10.txt')
sentences = [sent.replace('\n', '') for sent in file]
rawSentences = [sent.replace('\n', '') for sent in test]
finalSummaries = []
tables = []

for i in range(3):
    summaryTable = {}
    goldSentences = []
    testSentences = []
    while len(goldSentences) < 100:
        newNumber = random.choice(range(0, len(sentences)))
        if sentences[newNumber][1] == 'S':
            goldSentences.append(sentences[newNumber])
            testSentences.append(rawSentences[newNumber])
    trees = [Tree.fromstring(sent) for sent in goldSentences]
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
    probD = contextProbDict(testSentences)
    parserProb = ViterbiProbParser(rules, probD)
    parser = ViterbiProbParser(rules)
    gold = []
    test = []
    testP = []
    for i in range(len(testSentences)):
        sentParse = parser.parse(word_tokenize(testSentences[i]))
        sentProbParse = parserProb.parse(word_tokenize(testSentences[i]))
        if sentParse is not None:
            summaryTable[testSentences[i]] = {'Gold': goldSentences[i], 'NoProb': (sentParse, sentParse.prob), 'Prob':(sentProbParse, sentProbParse.prob)}
            gold.append(goldSentences[i])
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
    recall += nonP[4]
    precision += nonP[5]
    fmeasure += nonP[6]
    accuracy += nonP[10]
    recallP += P[4]
    precisionP += P[5]
    fmeasureP += P[6]
    accuracyP += P[10]
print('Recall: ', recall/5, recallP/5)
print('Precision: ', precision/5, precisionP/5)
print('F-Measure: ', fmeasure/5, fmeasureP/5)
print('Accuracy: ', accuracy/5, accuracyP/5)

parseStats = []
labels = ['Correct', 'BothWrong', 'NormalWrong', 'ProbWrong']
for table in tables:
    correct = set()
    bothWrong = set()
    normalWrong = set()
    probWrong = set()
    for sent, answer in table.items():
        answerTuple = (sent, answer['NoProb'][1], answer['Prob'][1])
        if str(answer['Prob'][0]) == answer['Gold'] and str(answer['NoProb'][0]) == answer['Gold']:
            correct.add(answerTuple)
        elif str(answer['Prob'][0]) != answer['Gold'] and str(answer['NoProb'][0]) == answer['Gold']:
            probWrong.add(answerTuple)
        elif str(answer['Prob'][0]) == answer['Gold'] and str(answer['NoProb'][0]) != answer['Gold']:
            normalWrong.add(answerTuple)
        elif str(answer['Prob'][0]) != answer['Gold'] and str(answer['NoProb'][0]) != answer['Gold']:
            bothWrong.add(answerTuple)
    ansSets = [correct, bothWrong, normalWrong, probWrong]
    parseStats.append(ansSets)
for ansSet in parseStats:
    print('-' * 30)
    for l in range(len(labels)):
        print(labels[l])
        print(len(ansSet[l]))
        print('Normal: ', sum([ans[1] for ans in ansSet[l]]) / len(ansSet[l]) )
        print('Context: ',  sum([ans[2] for ans in ansSet[l]]) / len(ansSet[l]) )



