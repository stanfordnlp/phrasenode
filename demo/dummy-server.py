# If you want to test your element selector code on the extension
# Use this as a template!

import datetime
import json

from bottle import post, request, run


@post('/pred')
def process():
    q = request.forms
    print('[{}] Received {}'.format(datetime.datetime.now().time(), q.query))
    info = json.loads(q.info)
    answer = 1      # TODO: Replace with model's prediction
    return {'query': q.query, 'answer': answer}


def start_server(port):
    # This will open a global port!
    print('[{}] Starting server'.format(datetime.datetime.now().time()))
    run(host='0.0.0.0', port=port)
    print('\nGood bye!')


if __name__ == '__main__':
    start_server(6006)
