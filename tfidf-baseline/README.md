index.js takes a dataset file and runs the baseline selector against each phrase
in each website

- tokenize each leaf element (and some whitelist/blacklist els) by their text
    and all attributes
- compute string overlap. weight text nodes by 5 and attrs by 1.
- can run porter stemmer, vary the weights, change white/blacklist etc.


converter.js converts output from index.js into a viewer-compatible format


selector.js holds the original, browser-based selector



preprocess.js takes the raw data files and outputs a list of documents from ALL
websites by above criteria (leaves/whitelist/blacklist)


tfidf.js computes tfidf scores from all the documents output by preprocess.js.
It outputs 2 files: one of the tfidf object, the other of each document's
membership in each page, since queries will need to be scoped to a particular
page...


query.js runs the tfidf baseline


converter2.js converts it to results...
