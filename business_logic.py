import feedparser
import re
import requests

from nlu import normalize_text, tokenize_text


ARXIV_API_URL = 'http://export.arxiv.org/api/query'
UNIVERSAL_SEQRCH_QUERY = 'all:the'

ARXIV_ID_PATTERN = re.compile('https?://arxiv.org/abs/(\d+\.\d+)v\d+')
ARXIV_ID_PATTERN2 = re.compile('https?://arxiv.org/abs/(.*)?$')


class Response:
    def __init__(self, text, buttons=None):
        self.text = text
        self.buttons = buttons or []


def get_articles(params):
    data = requests.post(ARXIV_API_URL, data=params)
    feed = feedparser.parse(data.text)
    return feed.get('entries', [])


def get_id_from_url(url):
    m = re.match(ARXIV_ID_PATTERN, url)
    if m:
        return m.groups()[0]
    m = re.match(ARXIV_ID_PATTERN2, url)
    if m:
        return m.groups()[0]
    raise ValueError('url {} cannot be parsed'.format(url))


def popularity_first(article):
    return -article.get('citation_count', 0), article.get('relevance_position')


class ArticleFinder:
    def __init__(
            self, conversation_model,
            citations=None, ontology=None, topic2papers=None,
            page_size=5, max_button_len=20
    ):
        self.ontology = {normalize_text(k): v for k, v in (ontology or {}).items()}
        self.citations = citations or {}
        self.topic2papers = topic2papers or {}
        self.page_size = page_size
        self.max_button_len = max_button_len
        self.conversation_model = conversation_model

    def extract_topics(self, text):
        normalized = normalize_text(text)
        if normalized in self.ontology:
            return [self.ontology[normalized]]
        # todo: implement a fuzzier matcher - e.g. with textdistance or even synonyms
        return []

    def do(self, state, semantic_frame):
        print(semantic_frame)
        intent = semantic_frame['intent']
        if intent == 'find':
            resp = self.do_find(state, semantic_frame)
        elif intent == 'next' or intent == 'prev':
            resp = self.do_prev_next(state, semantic_frame)
        elif intent == 'choose':
            resp = self.do_choose(state, semantic_frame)
        elif intent == 'details':
            resp = self.do_details(state, semantic_frame)
        elif intent == 'help':
            resp = self.do_help(state, semantic_frame)
        else:
            resp = self.do_conversation(state, semantic_frame)
        state['last_frame'] = semantic_frame
        return resp

    def do_conversation(self, state, semantic_frame):
        return Response(self.conversation_model.reply(state['text']))

    def do_find(self, state, frame):
        params = {
            # 'search_query': UNIVERSAL_SEQRCH_QUERY,
            'start': 0,
            'max_results': 50,
            # 'sortBy': 'lastUpdatedDate',
            # 'sortOrder': 'descending',
            # 'id_list': '',
        }
        query_parts = []

        topic_text = frame.get('TOPIC')
        article_name = frame.get('NAME')
        author_text = frame.get('AUTHOR')
        journal_text = frame.get('JOURNAL')
        org_text = frame.get('ORG')
        is_top = frame.get('TOP')
        is_fresh = frame.get('FRESH')
        is_old = frame.get('OLD')

        if topic_text is not None:
            topics = self.extract_topics(topic_text)
            if len(topics) > 0:
                query_parts.append(
                    '(' + ' OR '.join(['cat:"{}"'.format(topic) for topic in topics]) + ')'
                )
            else:
                query_parts.append('(ti:"{0}" OR abs:"{0}")'.format(topic_text))

        if author_text is not None:
            query_parts.append('(au:"{}")'.format(author_text))

        if journal_text is not None:
            query_parts.append('(jr:"{}")'.format(journal_text))

        if article_name is not None:
            query_parts.append('(ti:"{}")'.format(article_name))

        if org_text is not None:
            query_parts.append('(au:"{}")'.format(org_text))
            # todo: filter by affiliation after the search

        if len(query_parts) > 0:
            params['search_query'] = ' AND '.join(query_parts)
        else:
            params['search_query'] = UNIVERSAL_SEQRCH_QUERY

        if is_fresh:
            params['sortBy'] = 'submittedDate'
            params['sortOrder'] = 'descending'
        elif is_old:
            params['sortBy'] = 'submittedDate'
            params['sortOrder'] = 'ascending'
        elif is_top:
            # we need to re-rank the top after search
            params['max_results'] = 1000

        if frame == {'intent': 'find', 'TOP': 'top'}:
            # just retrieve the top from cache and use their ids
            most_cited = sorted(self.citations.items(), key=lambda p: len(p[1]), reverse=True)
            del params['search_query']
            params['id_list'] = ','.join([p[0] for p in most_cited[:100]])
        elif set(frame.keys()) == {'intent', 'TOP', 'TOPIC'}:
            norm_topic = normalize_text(topic_text)
            most_cited = self.topic2papers.get(norm_topic, [])
            if len(most_cited) >= 50:
                del params['search_query']
                params['id_list'] = ','.join(most_cited[:100])

        print(params)
        articles = get_articles(params)

        if is_top and not is_fresh and not is_old:
            for i, article in enumerate(articles):
                # todo: maybe get the real citations count from SemanticScholar
                key = get_id_from_url(article['id'])
                article['citation_count'] = len(self.citations.get(key, []))
                article['relevance_position'] = i
            articles.sort(key=popularity_first)

        state['last_search'] = params
        state['found_articles'] = articles
        state['current_page'] = 0

        return self.render_page(state, frame)

    def do_prev_next(self, state, frame):
        if 'current_page' not in state:
            return Response('Sorry, I cannot show the search results.')
        if frame['intent'] == 'next':
            if (state['current_page'] + 1) * self.page_size >= len(state.get('found_articles', [])):
                return Response('Sorry, I cannot show more articles.')
            state['current_page'] += 1
        else:
            if state.get('last_frame', {}).get('intent') in {'choose', 'details'}:
                # 'back' means 'to the last viewed part of the list'
                pass
            else:
                if state['current_page'] <= 0:
                    return Response('Sorry, this is already the beginning.')
                state['current_page'] -= 1
        return self.render_page(state, frame)

    def render_page(self, state, frame):
        fa = state.get('found_articles', [])
        if len(fa) == 0:
            return Response('Sorry, no articles were found')
        p = state.get('current_page', 0)
        first = p * self.page_size
        last = min(len(fa), (p + 1) * self.page_size)
        if len(fa[first:last]) < 1:
            return Response('An error: no articles found on the current page')

        names = ['{}: {}'.format(i, fa[i]['title'].replace('\n', ' ')) for i in range(first, last)]
        buttons = [str(i) for i in range(first, last)]
        if p > 0:
            buttons.append('previous papers')
        if last < len(fa):
            buttons.append('next papers')

        return Response(text='\n'.join(names), buttons=buttons)

    def do_choose(self, state, frame):
        try:
            idx = int(frame['index'])  # todo: convert words to numbers
        except ValueError:
            return Response('Please enter a number to choose the paper.')
        state['current_article_index'] = idx
        a = state['found_articles'][idx]
        state['article'] = a
        return Response(
            text='{} ({})\n{}'.format(a['author'], a['published'][0:10], a['title'].replace('\n', ' ')),
            buttons=['show summary']
        )

    def do_details(self, state, frame):
        a = state['article']
        return Response(
            text='{} ({})\n{}\n{}\n{}'.format(
                a['author'],  # todo: maybe render more authors
                a['published'][0:10],
                a['title'].replace('\n', ' '),
                a['summary'].replace('\n', ' '),
                a['link']
            ),
            buttons=['show summary']
        )

    def do_help(self, state, frame):
        examples = ['find me popular papers by Mikolov', "get some fresh papers on neurobiology"]
        return Response('I am a bot that can search for articles on arXiv.org or just chat with you. '
                        'For example, you can ask "{}" or "{}".'.format(*examples), buttons=examples)
