import random
import sys
from nltk.grammar import Nonterminal


def c2t(constituent, taggable):
    c = str(constituent)
    if c in taggable:
        return 'B-' + taggable[c]
    return 'O'


def sample_tags(grammar, start=None, depth=None, taggable=None):
    if not start:
        start = grammar.start()
    if depth is None:
        depth = sys.maxsize
    if not taggable:
        taggable = {}
    return _sample_all_tags(grammar, [(start, c2t(start, taggable))], depth, taggable)


def _sample_all_tags(grammar, items, depth, taggable):
    if items:
        frag1 = _sample_one_tag(grammar, items[0], depth, taggable)
        frag2 = _sample_all_tags(grammar, items[1:], depth, taggable)
        return frag1 + frag2
    else:
        return []


def _sample_one_tag(grammar, item, depth, taggable):
    try:
        item, tag = item
    except TypeError as e:
        print(item)
        raise e
    if depth > 0:
        if isinstance(item, Nonterminal):
            new_tag = c2t(item, taggable)
            if new_tag == 'O' and tag != 'O':
                new_tag = tag
            prod = random.choice(grammar.productions(lhs=item))
            new_items = [[c, new_tag] for c in prod.rhs()]
            if new_tag.startswith('B-'):
                for new_item in new_items[1:]:
                    new_item[1] = 'I-' + new_item[1][2:]
            frag = _sample_all_tags(grammar, new_items, depth - 1, taggable)
            return frag
        else:
            return [(item, tag)]


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
