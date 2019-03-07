const DIR = '../../data/phrase-node-dataset/'
const DATA_FILE = DIR+'data/combined.jsonl'
const PAGE_PATH = (version, p) => DIR+'pages/'+version+'/'+p+'.html'

const PREPROCESS_OUTPUT = DIR+'all-nodes.jsonl'
const UNSTEMMED_PREPROCESS_OUTPUT = DIR+'all-nodes-unstemmed.jsonl'
const QUERY_OUTPUT = DIR+'tfidf-results.jsonl'

module.exports = {DIR, DATA_FILE, PAGE_PATH, PREPROCESS_OUTPUT, QUERY_OUTPUT, UNSTEMMED_PREPROCESS_OUTPUT}
