"""A node filter returns whether each node is a valid prediction candidate.

Args:
    web_page (WebPage)
    web_page_code (tuple(str, str))
Returns:
    list[bool] of length len(web_page.nodes)
"""
import json
import os
from phrasenode import data


def none_node_filter(web_page, web_page_code):
    return [True] * len(web_page.nodes)


def visibility_node_filter(web_page, web_page_code):
    mask = []
    for node in web_page:
        mask.append(node.visible)
    return mask


BLACKLIST = {'p', 'style', 'script', 'code', 'pre', 'small', 'center'}
WHITELIST = {'a', 'span', 'button'}

def baseline_node_filter(web_page, web_page_code):
    """Use the following filters:
    - must have all of (width > 0, height > 0, hidden = false)
    - must not be in the blacklist: ['p', 'style', 'script', 'code', 'pre', 'small', 'center']
    - must either be a leaf or be in the whitelist: ['a', 'span', 'button']
    """
    mask = []
    for node in web_page:
        ok = True
        if not (node.width > 0 and node.height > 0 and not node.hidden):
            ok = False
        elif node.tag in BLACKLIST:
            ok = False
        elif node.children and node.tag not in WHITELIST:
            ok = False
        mask.append(ok)
    return mask


class VimiumPrecomputedNodeFilter(object):
    """Load a precomputed list of valid XIDs based on Vimium."""
    DEFAULT_PATH = os.path.join(data.workspace.phrase_node, 'infos', 'good-xids.jsonl')

    def __init__(self, path=DEFAULT_PATH):
        """Initialize.

        Args:
            path (str): Path to load the XIDs. Each line contains
                {'version': ..., 'webpage': ..., 'xids': [...]}
        """
        self.valid_xids = {}
        with open(path) as fin:
            for line in fin:
                data = json.loads(line)
                web_page_code = (data['version'], data['webpage'])
                self.valid_xids[web_page_code] = set(data['xids'])

    def __call__(self, web_page, web_page_code):
        valid_xids = self.valid_xids[web_page_code]
        return [node.xid in valid_xids for node in web_page]


################################################

def get_node_filter(name):
    if not name or name == 'none':
        node_filter = none_node_filter
    elif name == 'visibility':
        node_filter = visibility_node_filter
    elif name == 'baseline':
        node_filter = baseline_node_filter
    elif name == 'vimium':
        node_filter = VimiumPrecomputedNodeFilter()
    else:
        raise ValueError('Unknown node filter {}'.format(name))
    return node_filter
