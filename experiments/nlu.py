import spacy

from nltk import CFG
from nltk.parse import RecursiveDescentParser


TAGGABLE_NODES = {
    'TOPIC': 'TOPIC',
    'NAME': 'NAME',
    'AUTHOR': 'AUTHOR',
    'JOURNAL': 'JOURNAL',
    'ORG': 'ORG',
    'TOP': 'TOP',
    'FRESH': 'FRESH',
    'OLD': 'OLD',
}


def get_top_label(text, parser):
    tree = try_parse(text, parser)
    if tree is None:
        return None
    return tree[0].label()


def try_parse(text, parser):
    try:
        tree = parser.parse_one(text.lower().split())
    except ValueError:
        return None
    return tree


class NLU:
    def __init__(self, find_grammar_file='find_grammar.txt', other_grammar_file='other_grammar.txt'):
        with open(find_grammar_file, 'r') as f:
            self.find_grammar_text = f.read()
        self.find_grammar = CFG.fromstring(self.find_grammar_text)

        with open(other_grammar_file, 'r') as f:
            self.other_grammar_text = f.read()
        self.other_grammar = CFG.fromstring(self.other_grammar_text)

        self.find_parser = RecursiveDescentParser(self.find_grammar)
        self.other_parser = RecursiveDescentParser(self.other_grammar)

        self.spacy_nlp = spacy.load('en_core_web_sm')

    def parse_text(self, text):
        other_label = get_top_label(text, self.other_parser)
        if other_label is not None:
            intent = other_label.lower()
            frame = {'intent': intent}
            if intent == 'choose':
                frame['index'] = text
            return frame
        find_tree = try_parse(text, self.find_parser)
        if find_tree is not None:
            frame = {'intent': 'find'}
            for subtree in find_tree.subtrees():
                tag_name = TAGGABLE_NODES.get(subtree.label())
                if tag_name is not None:
                    frame[tag_name] = ' '.join(subtree.leaves())
            return frame
        # todo: try ML parsing
        return {'intent': 'find', 'TOPIC': text}
