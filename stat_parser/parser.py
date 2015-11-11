"""
CKY algorithm from the "Natural Language Processing" course by Michael Collins
https://class.coursera.org/nlangp-001/class
"""
from collections import defaultdict
from pprint import pprint

try:
    from nltk import Tree
    
    from nltk.draw.util import CanvasFrame
    from nltk.draw import TreeWidget

    def nltk_tree(t):
        return Tree(t[0], [c if isinstance(c, str) else nltk_tree(c) for c in t[1:]])
    
    nltk_is_available = True

except ImportError:
    nltk_is_available = False

from stat_parser.learn import build_model
from stat_parser.tokenizer import PennTreebankTokenizer
from stat_parser.treebanks.normalize import un_chomsky_normal_form
from stat_parser.word_classes import is_cap_word


def argmax(lst):
    return max(lst) if lst else (0.0, None)


def backtrace(back, bp):
    # Extract the tree from the backpointers
    if not back: return None
    if len(back) == 6:
        (X, Y, Z, i, s, j) = back
        return [X, backtrace(bp[i  , s, Y], bp),
                   backtrace(bp[s+1, j, Z], bp)]
    else:
        (X, Y, i, i) = back
        return [X, Y]

def two_largest(scores):
    count = 0
    m1 = m2 = (float('-inf'), defaultdict(tuple))
    for (s, p) in scores:
        count += 1
        if s > m2[0]:
            if s >= m1[0]:
                m1, m2 = (s, p), m1            
            else:
                m2 = (s, p)
    return m1, m2 if count >= 2 else m1


# modified CKY that returns a list of two highest scoring parsing and prints scores
def CKY(pcfg, norm_words): 
    x, n = [("", "")] + norm_words, len(norm_words)
    
    # Charts
    pi = defaultdict(float) # scores
    bp = defaultdict(tuple) # back pointers
    for i in range(1, n+1):
        for X in pcfg.N:
            norm, word = x[i]
            if (X, norm) in pcfg.q1:
                pi[i, i, X] = pcfg.q1[X, norm]
                bp[i, i, X] = (X, word, i, i)
    
    # Dynamic program
    for l in range(1, n):
        for i in range(1, n-l+1):
            j = i+l
            for X in pcfg.N:
                # Note that we only check rules that exist in training
                # and have non-zero probability
                score, back = argmax([(
                        pcfg.q2[X, Y, Z] * pi[i, s, Y] * pi[s+1, j, Z],
                        (X, Y, Z, i, s, j)
                    ) for s in range(i, j)
                        for Y, Z in pcfg.binary_rules[X]
                            if pi[i  , s, Y] > 0.0
                            if pi[s+1, j, Z] > 0.0
                ])
                
                if score > 0.0:
                    bp[i, j, X], pi[i, j, X] = back, score
    
    largest, second_largest = two_largest([(pi[1, n, X], bp[1, n, X]) for X in pcfg.N])
    f = open('TreeProbabilites.txt', 'w')
    score_message = "tree1: " + repr(largest[0]) + "\n"
    f.write(score_message)
    score_message = "tree2: " + repr(second_largest[0]) + "\n"
    f.write(score_message)
    f.close()
    return [backtrace(largest[1], bp), backtrace(second_largest[1], bp)]


class Parser:
    def __init__(self, pcfg=None):
        if pcfg is None:
            pcfg = build_model()
        
        self.pcfg = pcfg
        self.tokenizer = PennTreebankTokenizer()
        
        if nltk_is_available:
            self.parse = self.nltk_parse
        else:
            self.parse = self.raw_parse
    
    def norm_parse(self, sentence):
        words = self.tokenizer.tokenize(sentence)
        if is_cap_word(words[0]):
            words[0] = words[0].lower()
        
        norm_words = []
        for word in words:
            if isinstance(word, tuple):
                # This is already a word normalized to the Treebank conventions
                norm_words.append(word)
            else:
                # rare words normalization
                norm_words.append((self.pcfg.norm_word(word), word))
        return CKY(self.pcfg, norm_words)
    
    def raw_parse(self, sentence):
        [tree1, tree2] = self.norm_parse(sentence)
        un_chomsky_normal_form(tree1)
        un_chomsky_normal_form(tree2)
        return [tree1, tree2]
    
    def nltk_parse(self, sentence):
        [tree1, tree2] = self.raw_parse(sentence)
        return [nltk_tree(tree1), nltk_tree(tree2)]

# displays trees to a file
def display_tree(tree):
    if nltk_is_available:
        count = 0
        for t in tree:
            cf = CanvasFrame()
            tc = TreeWidget(cf.canvas(), t)
            cf.add_widget(tc, 10, 10)
            count += 1
            fileName = "tree" + repr(count) + ".ps"
            cf.print_to_file(fileName)
            cf.destroy()
    else:
        count = 0
        for t in tree:
            count += 1
            fileName = "tree" + repr(count) + ".txt"
            pprint.pprint(t, fileName)
