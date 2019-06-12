import feedparser
import re
import requests

from experiments.nlu import normalize_text, tokenize_text

from experiments.dialog import Response

ARXIV_API_URL = 'http://export.arxiv.org/api/query'
UNIVERSAL_SEQRCH_QUERY = 'all:the'

ARXIV_ID_PATTERN = re.compile('https?://arxiv.org/abs/(\d+\.\d+)v\d+')
ARXIV_ID_PATTERN2 = re.compile('https?://arxiv.org/abs/(.*)?$')


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
    def __init__(self, citations=None, ontology=None, page_size=5, max_button_len=20):
        self.ontology = ontology or {}
        self.citations = citations or {}
        self.page_size = page_size
        self.max_button_len = max_button_len

    def extract_topics(self, text):
        normalized = normalize_text(text)
        if normalized in self.ontology:
            return [self.ontology[normalized]]
        words = tokenize_text(normalized)
        # todo: do the match
        matches = []
        return matches

    def do(self, state, semantic_frame):
        print(semantic_frame)
        intent = semantic_frame['intent']
        if intent == 'find':
            return self.do_find(state, semantic_frame)
        elif intent == 'next' or intent == 'prev':
            return self.do_prev_next(state, semantic_frame)
        elif intent == 'choose':
            return self.do_choose(state, semantic_frame)
        elif intent == 'details':
            return self.do_details(state, semantic_frame)
        return None

    def do_find(self, current_form, query):
        params = {
            # 'search_query': requester.UNIVERSAL_SEQRCH_QUERY,
            'start': 0,
            'max_results': 10,
            # 'sortBy': 'lastUpdatedDate',
            # 'sortOrder': 'descending',
            # 'id_list': '',
        }
        query_parts = []

        topic_text = query.get('TOPIC')
        article_name = query.get('NAME')
        author_text = query.get('AUTHOR')
        journal_text = query.get('JOURNAL')
        org_text = query.get('ORG')
        is_top = query.get('TOP')
        is_fresh = query.get('FRESH')
        is_old = query.get('OLD')

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

        print(params)
        articles = get_articles(params)

        if is_top and not is_fresh and not is_old:
            for i, article in enumerate(articles):
                # todo: maybe get the real citations count from SemanticScholar
                key = get_id_from_url(article['id'])
                article['citation_count'] = len(self.citations.get(key, []))
                article['relevance_position'] = i
            articles.sort(key=popularity_first)

        current_form['last_search'] = params
        current_form['found_articles'] = articles
        current_form['current_page'] = 0

        return self.render_page(current_form, query)

    def do_prev_next(self, current_form, query):
        if 'current_page' not in current_form:
            return Response('Sorry, I cannot show the search results.')
        if query['intent'] == 'next':
            if (current_form['current_page'] + 1) * self.page_size >= len(current_form.get('found_articles', [])):
                return Response('Sorry, I cannot show more articles.')
            current_form['current_page'] += 1
        else:
            if current_form['current_page'] <= 0:
                return Response('Sorry, this is already the beginning.')
            current_form['current_page'] -= 1
        return self.render_page(current_form, query)

    def render_page(self, current_form, query):
        fa = current_form.get('found_articles', [])
        if len(fa) == 0:
            return Response('Sorry, no articles were found')
        p = current_form.get('current_page', 0)
        first = p * self.page_size
        last = min(len(fa), (p + 1) * self.page_size)
        if len(fa[first:last]) < 1:
            return Response('An error: no articles found on the current page')

        names = ['{}: {}'.format(i, fa[i]['title']) for i in range(first, last)]
        # buttons = [n[:(self.max_button_len-3)]+'...' for n in names]  # todo: soft split
        buttons = [str(i) for i in range(first, last)]
        if p > 0:
            buttons.append('previous papers')
        if last < len(fa):
            buttons.append('next papers')

        return Response(text='\n'.join(names), buttons=buttons)

    def do_choose(self, current_form, query):
        idx = int(query['index'])  # todo: convert words to numbers
        current_form['current_article_index'] = idx
        a = current_form['found_articles'][idx]
        current_form['article'] = a
        return Response(
            text='{} ({})\n{}'.format(a['author'], a['published'][0:10], a['title']),
            buttons=['show summary']
        )

    def do_details(self, current_form, query):
        a = current_form['article']
        return Response(
            text='{} ({})\n{}\n{}\n{}'.format(
                a['author'], a['published'][0:10], a['title'], a['summary'].replace('\n', ' '), a['link']
            ),
            buttons=['show summary']
        )
