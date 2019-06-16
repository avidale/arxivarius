import argparse
import json
import numpy as np
import keras
import random

from keras.layers import GRU, Dense, Input, TimeDistributed, Softmax, Add
from keras.layers import Conv1D, GlobalMaxPooling1D, Dropout
from keras.layers.wrappers import Bidirectional
from keras.models import Model
from keras_tqdm import TQDMNotebookCallback
from nltk import CFG
from nltk.parse.generate import generate
from tqdm.auto import tqdm, trange

import nlu
import grammar_tools
from conversation import SimpleConversation


def generate_vector_samples(grammar, taggable, n=3000):
    sentences = [
        grammar_tools.sample_tags(grammar=grammar, taggable=taggable)
        for i in trange(n)
    ]
    sentences.sort(key=lambda k: len(k))
    texts = [' '.join([p[0] for p in raw_s]) for raw_s in sentences]
    tagss = [[p[1] for p in raw_s] for raw_s in sentences]
    vecss = [nlu.text2vecs(text) for text in tqdm(texts)]
    return texts, tagss, vecss


def build_tagger_model(emb_size=96, rnn_size=64, n_classes=11):
    inp = Input((None, emb_size))
    first_birnn = Bidirectional(GRU(rnn_size, return_sequences=True), name='bi-1')(inp)
    second_birnn = Bidirectional(GRU(rnn_size, return_sequences=True), name='bi-2')(first_birnn)
    birnn = Add()([first_birnn, second_birnn])
    logits = TimeDistributed(Dense(n_classes))(birnn)
    prediction = Softmax()(logits)
    model = Model(inp, prediction)
    return model


def build_clf_model(emb_size=96, n_classes=6):
    inp = Input((None, emb_size))
    conv = Conv1D(32, 5, padding='same')(inp)
    drop = Dropout(rate=0.5)(conv)
    pooled = GlobalMaxPooling1D()(drop)
    dense = Dense(n_classes)(pooled)
    prediction = Softmax()(dense)
    clf = Model(inp, prediction)
    return clf


def train_tagger(nlu_module):
    texts, tagss, vecss = generate_vector_samples(nlu_module.find_grammar, taggable=nlu.TAGGABLE_NODES, n=3000)
    # validation data
    ttexts, ttagss, tvecss = generate_vector_samples(nlu_module.find_grammar, taggable=nlu.TAGGABLE_NODES, n=300)
    all_tags = sorted(set(t for tags in tagss for t in tags))
    tag2id = {tag: i for i, tag in enumerate(all_tags)}

    def encode_tags(tags):
        y = np.zeros([len(tags), len(all_tags)])
        for i, tag in enumerate(tags):
            y[i, tag2id[tag]] = 1
        return y

    def generate_batches(vecss, tagss):
        while True:
            indices = list(range(len(vecss)))
            np.random.shuffle(indices)
            for t in range(0, len(indices)):
                i = indices[t]
                X = np.stack([vecss[i]])
                Y = np.stack([encode_tags(tagss[i])])
                yield X, Y

    model = build_tagger_model(n_classes=len(all_tags))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[keras.metrics.categorical_accuracy])
    model.fit_generator(
        generate_batches(vecss, tagss),
        steps_per_epoch=len(vecss),
        epochs=10,
        verbose=2,
        validation_data=generate_batches(tvecss, ttagss),
        validation_steps=len(tvecss)
    )

    model.save('models/tagger.h5')
    with open('models/all_tags.json', 'w') as f:
        json.dump(all_tags, f)


def train_classifier(nlu_module):
    joint_grammar = CFG.fromstring(nlu_module.find_grammar_text + '\n' + nlu_module.other_grammar_text)
    INTENT_NAMES = sorted({str(p.rhs()[0]) for p in joint_grammar.productions() if str(p.lhs()) == 'S'})
    INTENT_NAMES = {c: c for c in INTENT_NAMES}

    fnd_text, fnd_tag, fnd_vec = generate_vector_samples(nlu_module.find_grammar, taggable=INTENT_NAMES, n=3000)
    oth_text, oth_tag, oth_vec = generate_vector_samples(nlu_module.other_grammar, taggable=INTENT_NAMES, n=3000)

    with open('data/persona_positives.json', 'r') as f:
        raw_persona = json.load(f)
    persona_sents = [sent for s in raw_persona['train'] for sent in s['dialog'] if sent != '__ SILENCE __']
    prs_text = random.sample(persona_sents, 3000)
    # add some more handcrafted samples
    with open('grammars/gc_grammar.txt', 'r') as f:
        gc_grammar = CFG.fromstring(f.read())
    for sent in generate(gc_grammar):
        prs_text.append(' '.join(sent))

    prs_tag = [['B-GC'] for _ in prs_text]
    prs_vec = [nlu.text2vecs(text) for text in tqdm(prs_text)]
    all_text = fnd_text + prs_text + oth_text
    all_labl = [c[0][2:] for c in fnd_tag + prs_tag + oth_tag]
    all_vec = fnd_vec + prs_vec + oth_vec

    all_intents = sorted(set(all_labl))
    intent2id = {intent: i for i, intent in enumerate(all_intents)}

    targets = np.zeros([len(all_labl), len(all_intents)])
    for i, intent in enumerate(all_labl):
        targets[i, intent2id[intent]] = 1

    clf = build_clf_model(n_classes=len(all_intents))
    clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    train_indices = np.random.permutation(len(targets))[:int(len(targets)*0.9)]
    val_indices = list(set(np.arange(len(targets))).difference(set(train_indices)))

    def sample_clf(basis=None):
        while True:
            if basis is not None:
                indices = np.random.permutation(basis)
            else:
                indices = np.random.permutation(len(targets))
            for i in indices:
                yield np.stack([all_vec[i]]), np.stack([targets[i]])

    clf.fit_generator(
        sample_clf(train_indices), steps_per_epoch=len(train_indices),
        validation_data=sample_clf(val_indices), validation_steps=len(val_indices),
        epochs=3, verbose=2, callbacks=[TQDMNotebookCallback()]
    )

    clf.save('models/classifier.h5')
    with open('models/all_intents.json', 'w') as f:
        json.dump(all_intents, f)


def train_gc():
    with open('data/persona_positives.json', 'r') as f:
        raw_persona = json.load(f)
    persona_pairs = [
        (s['dialog'][i], sent)
        for s in raw_persona['train']
        for i, sent in enumerate(s['dialog'][1:])
        if sent != '__ SILENCE __' and s['dialog'][i] != '__ SILENCE __'
    ]
    random_pairs = random.sample(persona_pairs, k=10000)
    model = SimpleConversation()
    model.train(random_pairs)


parser = argparse.ArgumentParser()
parser.add_argument('--gc', action='store_true')
parser.add_argument('--clf', action='store_true')
parser.add_argument('--tag', action='store_true')
if __name__ == '__main__':
    args = parser.parse_args()
    nlu_module = nlu.NLU()
    if not args.gc and not args.clf and not args.tag:
        print('There is nothing to train! Please provide the command line options.')
    if args.gc:
        print('training conversation model')
        train_gc()
    if args.clf:
        print('training classifier')
        train_classifier(nlu_module)
    if args.tag:
        print('training tagger')
        train_tagger(nlu_module)
    print('all done!')
