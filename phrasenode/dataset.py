"""Loading the phrase-node dataset."""
import sys, os, json, copy

from phrasenode.webpage import WebPage, check_web_page


class PhraseNodeStorage(object):
    """Storage for phrase-node datasets."""

    def __init__(self, basedir):
        """Initialize a PhraseNodeStorage object.

        Args:
            basedir (str): Base directory containing the following subdirectories:
              - data: JSONL files for examples
              - infos: info JSON files
              - pages: raw HTML files
        """
        self.basedir = basedir

    def load_examples(self, set_name):
        """Load examples from a certain set name, grouped by web page codes
            [(version, webpage) from JSONL].

        Args:
            set_names (str)
        Returns:
            dict[(str, str) -> list[PhraseNodeExample]]:
                Mapping web page code to examples with that web page
        """
        groups = {}
        filename = os.path.join(self.basedir, 'data', set_name + '.jsonl')
        with open(filename) as fin:
            for line in fin:
                raw = json.loads(line)
                example = PhraseNodeExample(self, raw)
                groups.setdefault(example.web_page_code, []).append(example)
        print >> sys.stderr, 'Read {} examples ({} web pages) from {}'.format(
                sum(len(examples) for examples in groups.itervalues()),
                len(groups), filename)
        return groups

    def get_web_page(self, web_page_code, check=True):
        """
        Args:
            web_page_code: tuple (version (str), web_page_name (str))
            check (bool): Check if the web page is good for training
                (using the check_web_page method)
        Returns:
            WebPage
        """
        version, web_page_name = web_page_code
        filename = os.path.join(self.basedir,
                'infos', version, 'info-' + web_page_name + '.gz')
        web_page = WebPage(filename)
        if check and not check_web_page(web_page):
            web_page = None
        return web_page


class PhraseNodeExample(object):

    def __init__(self, storage, metadata):
        """Create a new PhraseNodeExample.

        Args:
            storage (PhraseNodeStorage)
            metadata (dict): Must have the following keys:
                - exampleId (str)
                - version (str)
                - webpage (str)
                - phrase (str)
                - xid (int)
        """
        self._storage = storage
        self._metadata = metadata
        self._example_id = metadata['exampleId']
        self._web_page_code = (metadata['version'], metadata['webpage'])
        self._phrase = unicode(metadata['phrase']).lower().strip()
        self._target_xid = metadata['xid']

    def __repr__(self):
        return 'Example[{}]({}, {}, {})'.format(
                self._example_id, self._web_page_code,
                self._phrase.encode('utf8'), self._target_xid)
    __str__ = __repr__

    @property
    def example_id(self):
        return self._example_id

    @property
    def web_page_code(self):
        """Return a tuple (version (str), web_page_name (str))."""
        return self._web_page_code

    def get_web_page(self):
        """Return a WebPage object (not cached)."""
        return self._storage.get_web_page(self._web_page_code)

    @property
    def phrase(self):
        """Return a unicode."""
        return self._phrase

    @property
    def target_xid(self):
        """Return the target xid as specified in the dataset file.

        Returns:
            int
        """
        return self._target_xid

    def clone_metadata(self):
        """Return a copy of the metadata dict."""
        return copy.copy(self._metadata)
