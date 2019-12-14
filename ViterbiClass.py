from functools import reduce
from nltk.tree import Tree
from nltk import Nonterminal, induce_pcfg, word_tokenize
import numpy as np


class ProbabilityTree():
    def __init__(self, node, children, prob):
        self.node = node
        self.children = children
        self.prob = prob

    def __str__(self):
        out = '(' + self.node
        for child in self.children:
            out += ' ' + str(child)
        return out + ')'

    def __eq__(self, other):
        for s, o in zip(str(self), other):
            if s != o:
                return False
        return True


    def leaves(self):
        output = []
        for child in self.children:
            if isinstance(child, ProbabilityTree):
                output.extend(child.leaves())
            else:
                output.append(child)
        return output


class Rule():
    def __init__(self, lhs, rhs, prob):
        self.lhs = lhs
        self.rhs = rhs
        self.prob = prob

    def __str__(self):
        mid = ' => ' + ', '.join(self.rhs)
        return self.lhs + mid + ' P: ' + str(self.prob)


class ViterbiProbParser:
    def __init__(self, rules, PWLD=None):
        self.grammar = self.makeGrammar(rules)
        self.PWLD = PWLD

    def makeGrammar(self, rules):
        output = []
        for lhs, rhs, prob in rules:
            output.append(Rule(lhs, rhs, prob))
        return output

    def matchProduct(self, rhs, start, end, constituents):
        if start >= end and rhs == []:
            return [[]]
        if start >= end or rhs == []:
            return []

        childlists = []
        for split in range(start, end+1):
            current = constituents.get((start, split, rhs[0]), None)
            if current is not None:
                nextProducts = self.matchProduct(rhs[1:], split, end, constituents)
                childlists += [[current] + p for p in nextProducts]

        return childlists

    def findPossibleConstituents(self, span, constituents):
        output = []
        for rule in self.grammar:
            childlists = self.matchProduct(rule.rhs, span[0], span[1], constituents)
            for child in childlists:
                output.append((rule, child))
        return output

    def addPossibleConstituents(self, span, constituents):
        changed = True
        while changed:
            changed = False
            possibles = self.findPossibleConstituents(span, constituents)
            for rule, children in possibles:
                subs = [c for c in children if isinstance(c, ProbabilityTree)]
                p = reduce(lambda pr, t: pr * t.prob, subs, rule.prob)
                if self.PWLD is not None:
                    if span[1] - span[0] > 1:
                        allLeaves = []
                        for child in children:
                            if isinstance(child, ProbabilityTree):
                                allLeaves.extend(child.leaves())
                        norms = []
                        for headNum in range(len(allLeaves)):
                            head = allLeaves[headNum]
                            if head in ['the', 'a', 'an']:
                                continue
                            z = 0
                            if head not in self.PWLD:
                                print(head)
                            for leafIndex in range(len(allLeaves)):
                                if leafIndex != headNum:
                                    prob = self.PWLD[head].get(allLeaves[leafIndex], 0)
                                    if prob != 0:
                                        z += np.log(prob)
                                    else:
                                        z = 0
                                        break
                            norms.append(z)
                        if np.sum(norms) != 0:
                            norms = [i / np.sum(norms) for i in norms]
                            p = 2 * np.exp(np.mean(norms)) * p / (np.exp(np.mean(norms)) + p)
                node = rule.lhs
                tree = ProbabilityTree(node, children, p)
                c = constituents.get((span[0], span[1], rule.lhs), None)
                if c is None or c.prob < tree.prob:
                    constituents[(span[0], span[1], rule.lhs)] = tree
                    changed = True

    def parse(self, tokens):
        tokens = list(tokens)
        constituents = {}

        for i in range(len(tokens)):
            current = tokens[i]
            constituents[i, i+1, current] = current

        for length in range(1, len(tokens) + 1):
            for start in range(len(tokens) - length + 1):
                span = (start, start + length)
                self.addPossibleConstituents(span, constituents)
        # for x in range(len(tokens)):
        #     for y in range(len(tokens), x, -1):
        #         output = constituents.get((x, y, 'S'), None)
        #         if output is not None:
        #             return output
        return constituents.get((0, len(tokens), 'S'), None)

def producePCFG(sentences):
    ruleCount = {}
    lhsCount = {}
    output = []
    rules = [Tree.fromstring(sent).productions() for sent in sentences]
    for rule in rules:
        if rule.lhs() not in lhsCount:
            lhsCount[rule.lhs()] = 0
        if rule not in ruleCount:
            ruleCount[rule] = 0
        lhsCount[rule.lhs()] += 1
        ruleCount[rule] += 1
    for rule in ruleCount:
        output.append(Rule(rule.lhs(), rule.rhs(), ruleCount[rule] / lhsCount[rule.lhs()]))
    return output

def contextProbDict(sentences):
    contextD = {}
    for sent in sentences:
        tokens = word_tokenize(sent)
        tokens = [tok.lower() for tok in tokens]
        for i in range(len(tokens) + 1):
            if tokens[i] not in contextD:
                contextD[tokens[i]] = {}
            for j in range(len(tokens) + 1):
                if i != j:
                    if tokens[j] not in contextD[tokens[i]]:
                        contextD[tokens[i]][tokens[j]] = 1
                    else:
                        contextD[tokens[i]][tokens[j]] += 1
    PWLD = {}
    for word, freq in contextD.items():
        sumWords = sum([v for k, v in freq.items()])
        PWLD[word] = {k: (v/sumWords) for k, v in freq.items()}
    return PWLD

if __name__ == '__main__':
    # grammar = toy_pcfg1
    # rules = []
    # for production in grammar.productions():
    #     lhs, rhs, prob = production._lhs, production._rhs, production._ProbabilisticMixIn__prob
    #     rhsList = [str(term) if type(term) != str else term for term in rhs]
    #     rules.append((str(lhs), rhsList, prob))
    # parse1 = '(S (NP I) (VP (V saw) (NP (DET an) (N astronaut)) (PP (P with) (NP (DET a) (N telescope)))))'
    # parse2 = '(S (NP I) (VP (V saw) (NP (NP (DET an) (N astronaut)) (PP (P with) (NP (DET a) (N telescope))))))'
    # print(Tree.fromstring(parse1))
    # print(Tree.fromstring(parse2))
    # rules = [('S', ['NP', 'VP'], 1.0),
    #            ('NP', ['DT', 'NN'], 0.25),
    #            ('NP', ['NP', 'PP'], 0.5),
    #            ('NP', ['I'], 0.25),
    #            ('VP', ['VB', 'NP'], 0.51),
    #            ('VP', ['VB', 'NP', 'PP'], 0.49),
    #            ('PP', ['IN', 'NP'], 1.0),
    #            ('DT', ['an'], 0.5),
    #            ('DT', ['a'], 0.5),
    #            ('NN', ['astronaut'], 0.5),
    #            ('NN', ['telescope'], 0.5),
    #            ('VB', ['saw'], 1.0),
    #            ('IN', ['with'], 1.0)]
    # parser = ViterbiProbParser(rules)
    # sent = 'I saw an astronaut with a telescope'
    # freq = {}
    # sents = ['I saw an astronaut with a telescope', 'the burglar threatened the student with the knife', 'I shot an elephant in my pajamas',
    #          'the professor said on Monday he would give an exam', 'I had forgotten how good beer tastes']
    # print(pos_tag(word_tokenize(sents[-1])))
    # for sent in sents:
    #     tokens = word_tokenize(sent)
    #     parts = pos_tag(tokens)
    #     for word, part in parts:
    #         if part == 'PRP$':
    #             part = 'PRP'
    #         if part in ['VBD', 'VBN', 'VBP']:
    #             part = 'VB'
    #         if part == 'WRB':
    #             part = 'RB'
    #         if part == 'NNP':
    #             part = 'NN'
    #         if part == 'NNS':
    #             part = 'VB'
    #         if part not in freq:
    #             freq[part] = set()
    #         freq[part].add(word)
    # for k,v in freq.items():
    #     print(k,v)
    # ans = parser.parse(word_tokenize(sent))
    # print(Tree.fromstring(str(ans)))
    # print(ans.prob)

    # parser = ViterbiParser(grammar)
    # parses = parser.parse(tokens)
    # for parse in parses:
    #     print(parse)
    sent1 = '(S (NP (NN I)) (VP (VB like) (NN dogs)))'
    sent2 = '(S (NP (NN dogs)) (VP (VB like) (NN food)))'
    sent3 = '(S (NP (NN I)) (VP (VB love) (NN food)))'
    raw1 = 'I like dogs'
    raw2 = 'dogs like food'
    raw3 = 'I love food'
    contextD = {}
    for sent in [raw1, raw2, raw3]:
        tokens = word_tokenize(sent)
        for i in range(len(tokens)):
            if tokens[i] not in contextD:
                contextD[tokens[i]] = {}
            for j in range(len(tokens)):
                if i != j:
                    if tokens[j] not in contextD[tokens[i]]:
                        contextD[tokens[i]][tokens[j]] = 1
                    else:
                        contextD[tokens[i]][tokens[j]] += 1
    PWLD = {}
    for word, freq in contextD.items():
        sumWords = sum([v for k, v in freq.items()])
        PWLD[word] = {k: (v / sumWords) for k, v in freq.items()}

    sents = [sent1, sent2, sent3]
    trees = [Tree.fromstring(sent) for sent in sents]
    productions = []
    for tree in trees:
        productions += tree.productions()
    start = Nonterminal('S')
    grammar = induce_pcfg(start, productions)
    print(grammar)
    for word, prob in PWLD.items():
        print(word, prob)
    rules = []
    for production in grammar.productions():
        lhs, rhs, prob = production._lhs, production._rhs, production._ProbabilisticMixIn__prob
        rhsList = [str(term) if type(term) != str else term for term in rhs]
        rules.append((str(lhs), rhsList, prob))
    parser = ViterbiProbParser(rules, PWLD)
    sents = [(raw1,sent1), (raw2,sent2), (raw3,sent3)]
    for raw, parse in sents:
        ans = parser.parse(word_tokenize(raw))
        print(ans, ans.prob)
        break





