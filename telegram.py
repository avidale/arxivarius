#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import tgalice

from business_logic import ArticleFinder
from nlu import NLU
from grammar_tools import sample_tags
from conversation import SimpleConversation

BASE_URL = 'https://arxivarius.herokuapp.com/'


class PapersDialogManager(tgalice.dialog_manager.CascadableDialogManager):
    def __init__(self, finder, nlu, *args, **kwargs):
        super(PapersDialogManager, self).__init__(*args, **kwargs)
        self.finder = finder
        self.nlu = nlu

    def try_to_respond(self, ctx):
        state = ctx.user_object
        state['text'] = ctx.message_text
        if ctx.source == tgalice.SOURCES.TELEGRAM:
            state['username'] = ctx.raw_message.from_user.username
        print('text is "{}"'.format(ctx.message_text))
        if not ctx.message_text:
            response = tgalice.dialog.Response('Hello! I am a bot that can find a paper for you.')
        else:
            semantic_frame = self.nlu.parse_text(ctx.message_text)
            response = self.finder.do(state, semantic_frame)
        for i in range(3):
            response.suggests.append(' '.join([p[0] for p in sample_tags(nlu_module.find_grammar)]))
        return response


if __name__ == '__main__':
    with open('models/citations.json', 'r') as f:
        citations = json.load(f)
    with open('models/area_to_code.json', 'r') as f:
        ontology = json.load(f)
    with open('models/topic2papers.json', 'r') as f:
        topic2papers = json.load(f)
    finder = ArticleFinder(
        conversation_model=SimpleConversation(), citations=citations, ontology=ontology, topic2papers=topic2papers
    )
    nlu_module = NLU()
    mongo_db = tgalice.storage.database_utils.get_mongo_or_mock()
    connector = tgalice.dialog_connector.DialogConnector(
        dialog_manager=PapersDialogManager(finder=finder, nlu=nlu_module),
        storage=tgalice.session_storage.MongoBasedStorage(database=mongo_db, collection_name='sessions'),
        log_storage=tgalice.storage.message_logging.MongoMessageLogger(
            database=mongo_db, collection_name='message_logs_v2', detect_pings=True
        )
    )
    server = tgalice.flask_server.FlaskServer(connector=connector, base_url=BASE_URL)
    server.parse_args_and_run()
