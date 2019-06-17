import spacy
import numpy as np
import json
import keras

import tensorflow as tf

import os
import re
import unicodedata

from nltk import CFG
from nltk.parse import BottomUpLeftCornerChartParser

import grammar_tools

graph = tf.get_default_graph()  # we have to use a global var because Flask works in a multi-thread mode

SPACY_NLP = spacy.load('en_core_web_sm', disable=['ner', 'parser'])


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
            find_grammar_file='grammars/find_grammar.txt',
            other_grammar_file='grammars/other_grammar.txt',
            classifier_file='models/classifier.h5',
            tagger_file='models/tagger.h5',
            all_intents_file='models/all_intents.json',
            all_tags_file='models/all_tags.json',
    ):
        with open(find_grammar_file, 'r') as f:
            self.find_grammar_text = f.read()
        self.find_grammar = CFG.fromstring(self.find_grammar_text)

        with open(other_grammar_file, 'r') as f:
            self.other_grammar_text = f.read()
        self.other_grammar = CFG.fromstring(self.other_grammar_text)

        self.find_parser = BottomUpLeftCornerChartParser(self.find_grammar)
        self.other_parser = BottomUpLeftCornerChartParser(self.other_grammar)

        if os.path.exists(all_tags_file):
            with open(all_tags_file, 'r') as f:
                self.all_tags = json.load(f)
        else:
            self.all_tags = []

        if os.path.exists(tagger_file):
            self.tagger = keras.models.load_model(tagger_file)
        else:
            self.tagger = None

        if os.path.exists(all_intents_file):
            with open(all_intents_file, 'r') as f:
                self.all_intents = json.load(f)
        else:
            self.all_intents = []

        if os.path.exists(classifier_file):
            self.classifier = keras.models.load_model(classifier_file)
        else:
            self.classifier = None

    def parse_text(self, original_text):
        print(original_text)
        text = normalize_text(original_text)
        # grammar-based parsing
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
        # fallback to neural parsing
        inp = np.stack([text2vecs(text)])
        with graph.as_default():
            intent_proba = self.classifier.predict(inp)[0]
        print(list(zip(self.all_intents, intent_proba)))
        intent = self.all_intents[intent_proba.argmax()].lower()
        frame = {'intent': intent}
        if intent == 'choose':
            frame['index'] = text
        if intent != 'find':
            return frame
        with graph.as_default():
            tags_scores = self.tagger.predict(inp)[0]
        tags_names = [self.all_tags[i] for i in tags_scores.argmax(axis=1)]
        tokens = tokenize_text(text)
        slots = tags_to_slots(tags_names, tokens)
        slots['intent'] = 'find'
        return slots


def text2vecs(text):
    return np.array([t.vector for t in SPACY_NLP(normalize_text(text))])


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
