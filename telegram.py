#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import json
import os
import pymongo
import mongomock
import telebot

from datetime import datetime

from flask import Flask, request

from business_logic import ArticleFinder
from nlu import NLU
from grammar_tools import sample_tags
from conversation import SimpleConversation

with open('models/citations.json', 'r') as f:
    citations = json.load(f)
with open('models/area_to_code.json', 'r') as f:
    ontology = json.load(f)

finder = ArticleFinder(conversation_model=SimpleConversation(), citations=citations, ontology=ontology)
nlu_module = NLU()

TOKEN = os.environ['TOKEN']
bot = telebot.TeleBot(TOKEN)

server = Flask(__name__)
TELEBOT_URL = 'telebot_webhook/'
BASE_URL = 'https://arxivarius.herokuapp.com/'

MONGO_URL = os.environ.get('MONGODB_URI')
if MONGO_URL:
    mongo_client = pymongo.MongoClient(MONGO_URL)
    mongo_db = mongo_client.get_default_database()
else:
    # preserve state in RAM only - for debugging purposes
    mongo_client = mongomock.MongoClient()
    mongo_db = mongo_client.db
mongo_states = mongo_db.get_collection('states')
mongo_message_logs = mongo_db.get_collection('message_logs')


def render_markup(suggests=None, max_columns=3, initial_ratio=2):
    if suggests is None or len(suggests) == 0:
        return telebot.types.ReplyKeyboardRemove(selective=False)
    markup = telebot.types.ReplyKeyboardMarkup(row_width=max(1, min(max_columns, int(len(suggests) / initial_ratio))))
    markup.add(*suggests)
    return markup


class LoggedMessage:
    def __init__(self, text, user_id, from_user, collection, **kwargs):
        self.text = text
        self.user_id = user_id
        self.from_user = from_user
        self.timestamp = str(datetime.utcnow())
        self.kwargs = kwargs
        self.mongo_collection = collection

    def save(self):
        self.mongo_collection.insert_one(self.to_dict())

    def to_dict(self):
        result = {
            'text': self.text,
            'user_id': self.user_id,
            'from_user': self.from_user,
            'timestamp': self.timestamp
        }
        for k, v in self.kwargs.items():
            if k not in result:
                result[k] = v
        return result


@server.route("/" + TELEBOT_URL)
def web_hook():
    bot.remove_webhook()
    bot.set_webhook(url=BASE_URL + TELEBOT_URL + TOKEN)
    return "!", 200


@server.route("/wakeup/")
def wake_up():
    web_hook()
    return "Webhook has been reset!", 200


@bot.message_handler(func=lambda message: True, content_types=['document', 'text', 'photo'])
def process_message(msg):
    bot.send_chat_action(msg.chat.id, 'typing')
    LoggedMessage(
        text=msg.text,
        user_id=msg.from_user.id,
        from_user=True,
        message_id=msg.message_id,
        collection=mongo_message_logs,
    ).save()
    chat_id = msg.chat.id
    username = msg.from_user.username
    state_obj = mongo_states.find_one({'key': chat_id})
    if state_obj is None:
        state = {}
    else:
        state = state_obj.get('state', {})
    state['text'] = msg.text
    semantic_frame = nlu_module.parse_text(msg.text)
    response = finder.do(state, semantic_frame)
    buttons = response.buttons
    for i in range(3):
        buttons.append(' '.join([p[0] for p in sample_tags(nlu_module.find_grammar)]))

    final_reply = bot.reply_to(msg, text=response.text, reply_markup=render_markup(buttons))
    mongo_states.update_one(
        {'key': chat_id},
        {'$set': {'key': chat_id, 'state': state, 'username': username}},
        upsert=True
    )
    LoggedMessage(
        text=response.text,
        user_id=msg.from_user.id,
        from_user=False,
        message_id=final_reply.message_id,
        collection=mongo_message_logs,
    ).save()


@server.route('/' + TELEBOT_URL + TOKEN, methods=['POST'])
def get_message():
    bot.process_new_updates([telebot.types.Update.de_json(request.stream.read().decode("utf-8"))])
    return "!", 200


parser = argparse.ArgumentParser(description='Run the bot')
parser.add_argument('--poll', action='store_true')


def main():
    args = parser.parse_args()
    if args.poll:
        bot.remove_webhook()
        bot.polling()
    else:
        web_hook()
        server.run(host="0.0.0.0", port=int(os.environ.get('PORT', 5000)))


if __name__ == '__main__':
    main()
