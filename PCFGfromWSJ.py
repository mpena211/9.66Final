from nltk import induce_pcfg
from nltk.tree import Tree
from nltk.grammar import Nonterminal
from nltk import word_tokenize
from ViterbiClass import ViterbiProbParser
from PYEVALB.scorer import Scorer


file = open('./ptb-data/ptb-train-10.txt')
sentences = [sent.replace('\n', '') for sent in file]
file.close()
trees = [Tree.fromstring(sent) for sent in sentences]
productions = []
for tree in trees:
    productions += tree.productions()
start = Nonterminal('S')
grammar = induce_pcfg(start, productions)
print(grammar)
rules = []
for production in grammar.productions():
    lhs, rhs, prob = production._lhs, production._rhs, production._ProbabilisticMixIn__prob
    rhsList = [str(term) if type(term) != str else term for term in rhs]
    rules.append((str(lhs), rhsList, prob))
test = open('./ptb-data/ptb-train-raw-10.txt')
testSentences = [sent.replace('\n', '') for sent in test]
test.close()
parser = ViterbiProbParser(rules)
ans = []
for sent in testSentences:
    p = parser.parse(word_tokenize(sent))
    ans.append(p)

results = Scorer.score_corpus(sentences, ans)
print(results)


