from ViterbiClass import contextProbDict
import pickle

trainRaw = open('./ptb-data/ptb-train-raw-10.txt')
testSent = open('./ptb-data/ptb-test-raw-10.txt')
validraw = open('./ptb-data/ptb-valid-raw-10.txt')
grammarRaw = [sent.replace('\n', '') for sent in trainRaw]
rawSentences = [sent.replace('\n', '') for sent in testSent]
validRaw = [sent.replace('\n', '') for sent in validraw]
output = contextProbDict(grammarRaw + rawSentences + validRaw)

with open('contextDict.pickle', 'wb') as handle:
    pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('contextDict.pickle', 'rb') as handle:
    b = pickle.load(handle)

for word, p in b.items():
    print(word, p)