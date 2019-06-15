#!/usr/bin/python
# -*- coding: utf-8 -*-
import argparse
import os
import telebot

from flask import Flask, request

from experiments.requester import ArticleFinder
from experiments.dialog import Response
from experiments.nlu import NLU
from experiments.grammar_tools import sample_tags

finder = ArticleFinder()
nlu_module = NLU()

state = {}

TOKEN = os.environ['TOKEN']
bot = telebot.TeleBot(TOKEN)

server = Flask(__name__)
TELEBOT_URL = 'telebot_webhook/'
BASE_URL = 'https://arxivarius.herokuapp.com/'

MONGO_URL = os.environ.get('MONGODB_URI')


def render_markup(suggests=None, max_columns=3, initial_ratio=2):
    if suggests is None or len(suggests) == 0:
        return telebot.types.ReplyKeyboardRemove(selective=False)
    markup = telebot.types.ReplyKeyboardMarkup(row_width=max(1, min(max_columns, int(len(suggests) / initial_ratio))))
    markup.add(*suggests)
    return markup


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
    semantic_frame = nlu_module.parse_text(msg.text)
    response = finder.do(state, semantic_frame)
    if response is None:
        response = Response('I cannot answer, so there should be a GC answer')

    buttons = response.buttons
    for i in range(2):
        buttons.append(' '.join([p[0] for p in sample_tags(nlu_module.find_grammar)]))

    bot.reply_to(msg, text=response.text, reply_markup=render_markup(buttons))


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
