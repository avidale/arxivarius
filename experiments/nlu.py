import spacy
import numpy as np
import json
import keras

import tensorflow as tf

import re
import unicodedata

from nltk import CFG
from nltk.parse import RecursiveDescentParser
from tqdm.auto import tqdm, trange

from experiments import grammar_tools

global graph
graph = tf.get_default_graph()

SPACY_NLP = spacy.load('en_core_web_sm')


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


class NLU:
    def __init__(
            self,
            find_grammar_file='find_grammar.txt', other_grammar_file='other_grammar.txt',
            tagger_file='tagger.h5', all_tags_file='all_tags.json'
    ):
        with open(find_grammar_file, 'r') as f:
            self.find_grammar_text = f.read()
        self.find_grammar = CFG.fromstring(self.find_grammar_text)

        with open(other_grammar_file, 'r') as f:
            self.other_grammar_text = f.read()
        self.other_grammar = CFG.fromstring(self.other_grammar_text)

        self.find_parser = RecursiveDescentParser(self.find_grammar)
        self.other_parser = RecursiveDescentParser(self.other_grammar)

        self.all_tags_file = all_tags_file
        with open(all_tags_file, 'r') as f:
            self.all_tags = json.load(f)

        self.tagger_file = tagger_file
        self.tagger = keras.models.load_model(tagger_file)

    def parse_text(self, text):
        text = normalize_text(text)
        other_label = grammar_tools.get_top_label(text, self.other_parser)
        if other_label is not None:
            intent = other_label.lower()
            frame = {'intent': intent}
            if intent == 'choose':
                frame['index'] = text
            return frame
        find_tree = grammar_tools.try_parse(text, self.find_parser)
        if find_tree is not None:
            frame = {'intent': 'find'}
            for subtree in find_tree.subtrees():
                tag_name = TAGGABLE_NODES.get(subtree.label())
                if tag_name is not None:
                    frame[tag_name] = ' '.join(subtree.leaves())
            return frame
        # todo: try ML classification
        tokens = tokenize_text(text)
        inp = np.stack([text2vecs(text)])
        print(inp.shape)
        with graph.as_default():
            tags_scores = self.tagger.predict(inp)[0]
        tags_names = [self.all_tags[i] for i in tags_scores.argmax(axis=1)]
        slots = tags_to_slots(tags_names, tokens)
        slots['intent'] = 'find'
        return slots


def text2vecs(text):
    return np.array([t.vector for t in SPACY_NLP(text)])


def generate_vector_samples(grammar, taggable, n=3000):
    sentences = [
        grammar_tools.sample_tags(grammar=grammar, taggable=taggable)
        for i in trange(n)
    ]
    sentences.sort(key=lambda k: len(k))
    texts = [' '.join([p[0] for p in raw_s]) for raw_s in sentences]
    tagss = [[p[1] for p in raw_s] for raw_s in sentences]
    vecss = [text2vecs(text) for text in tqdm(texts)]
    return texts, tagss, vecss


def normalize_text(text):
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode()
    text = re.sub('[^0-9a-z]', ' ', text.lower())
    text = re.sub('\s+', ' ', text)
    return text


def tokenize_text(text):
    return [str(tok) for tok in SPACY_NLP.tokenizer(text)]


def tags_to_slots(tags, tokens):
    slots = {}
    for tag, tok in zip(tags, tokens):
        if tag == 'O':
            continue
        slot = tag[2:]
        if slot not in slots:
            slots[slot] = tok
        else:
            slots[slot] = slots[slot] + ' ' + tok
    return slots
