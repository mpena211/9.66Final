from functools import reduce
from nltk.tree import Tree
from nltk import pos_tag, word_tokenize


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


class Rule():
    def __init__(self, lhs, rhs, prob):
        self.lhs = lhs
        self.rhs = rhs
        self.prob = prob

    def __str__(self):
        mid = ' => ' + ', '.join(self.rhs)
        return self.lhs + mid + ' P: ' + str(self.prob)


class ViterbiProbParser():
    def __init__(self, rules):
        self.grammar = self.makeGrammar(rules)

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
                print(span)
                self.addPossibleConstituents(span, constituents)

        return constituents.get((0, len(tokens), 'S'))

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
    rules = [('S', ['NP', 'VP'], 1.0),
               ('NP', ['DT', 'NN'], 0.25),
               ('NP', ['NP', 'PP'], 0.5),
               ('NP', ['I'], 0.25),
               ('VP', ['VB', 'NP'], 0.8),
               ('VP', ['VB', 'NP', 'PP'], 0.2),
               ('PP', ['IN', 'NP'], 1.0),
               ('DT', ['an'], 0.5),
               ('DT', ['a'], 0.5),
               ('NN', ['astronaut'], 0.5),
               ('NN', ['telescope'], 0.5),
               ('VB', ['saw'], 1.0),
               ('IN', ['with'], 1.0)]
    parser = ViterbiProbParser(rules)
    freq = {}
    sents = ['I saw an astronaut with a telescope', 'the burglar threatened the student with the knife', 'I shot an elephant in my pajamas',
             'the professor said on Monday he would give an exam', 'I had forgotten how good beer tastes']
    print(pos_tag(word_tokenize(sents[-1])))
    for sent in sents:
        tokens = word_tokenize(sent)
        parts = pos_tag(tokens)
        for word, part in parts:
            if part == 'PRP$':
                part = 'PRP'
            if part in ['VBD', 'VBN', 'VBP']:
                part = 'VB'
            if part == 'WRB':
                part = 'RB'
            if part == 'NNP':
                part = 'NN'
            if part == 'NNS':
                part = 'VB'
            if part not in freq:
                freq[part] = set()
            freq[part].add(word)
    for k,v in freq.items():
        print(k,v)
    # ans = parser.parse(tokens)
    # print(Tree.fromstring(str(ans)))
    # print(ans.prob)

    # parser = ViterbiParser(grammar)
    # parses = parser.parse(tokens)
    # for parse in parses:
    #     print(parse)
