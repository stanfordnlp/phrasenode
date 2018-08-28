import gzip
import json
import logging

from gtd.utils import cached_property

from phrasenode.constants import GraphRels
from phrasenode.utils import rect_area, rect_overlap


class Node(object):
    """Represent a single DOM node in a web page."""

    def __init__(self, raw_info, web_page):
        """Initialize a node.

        The following attributes are assigned outside the initializer:
        - old_ref (int): Node index w.r.t. json data
        - ref (int): Node index w.r.t. web_page.nodes
        - x_ratio (float): Ratio of the x midpoint w.r.t. page width (0.0 to 1.0)
        - y_ratio (float): Ratio of the y midpoint w.r.t. page height (0.0 to 1.0)
        - visible (bool): Whether the node is visible

        Args:
            raw_info (dict): An info entry from the JSON file
            web_page (WebPage)
        """
        self.raw_info = raw_info
        self.web_page = web_page
        self.parent = None
        self.children = []
        self.tag = raw_info['tag'].lower()
        self.left = raw_info['left']
        self.top = raw_info['top']
        self.width = raw_info['width']
        self.height = raw_info['height']
        if 'text' in raw_info:
            self.text = unicode(raw_info['text'])
        else:
            self.text = None
        self.value = raw_info.get('value')
        self.id_ = raw_info.get('id') or ''
        self.classes = (raw_info.get('classes') or '').strip().split()
        self.attributes = raw_info.get('attributes', {})
        self.style_overrides = raw_info.get('styles')
        self.xid = raw_info.get('xid')
        self.hidden = raw_info.get('hidden')
        self.top_level = raw_info.get('topLevel')

    def add_child(self, child_node):
        assert child_node.parent is None
        self.children.append(child_node)
        child_node.parent = self

    def __str__(self):
        if self.text:
            text = self.text
            text = text[:20] + '...' if len(text) > 20 else text
            text_str = ' text={}'.format(repr(text))
        else:
            text_str = ''

        id_str = ' id={}'.format(repr(self.id_)) if self.id_ else ''
        classes_str = ' classes=[{}]'.format(repr(self.classes))
        num_children = len(self.children)
        children_str = ' children={}'.format(num_children) if num_children != 0 else ''

        return '{tag} @ ({left}, {top}){text}{id_}{classes}{children}'.format(
            tag=self.tag, left=round(self.left, 2), top=round(self.top, 2),
            text=text_str, id_=id_str, classes=classes_str, children=children_str)

    __repr__ = __str__

    def visualize(self, join=True, max_levels=10):
        """Return a string visualizing the tree structure."""
        lines = []
        lines.append('- {}'.format(self))
        if max_levels > 0:
            for i, child in enumerate(self.children):
                for j, line in enumerate(child.visualize(
                        join=False, max_levels=max_levels-1)):
                    prefix = '   ' if (i == len(self.children) - 1 and j) else '  |'
                    lines.append(prefix + line)
        return '\n'.join(lines) if join else lines

    ################################################
    # Node properties

    def all_texts(self, max_words=10000):
        """Return the concatenation of the texts of all descendants

        Args:
            max_words (int): Maximum number of space-delimited words to return
        Returns:
            list[unicode]
        """
        if max_words <= 0:
            return []
        words = []
        if self.text:
            words.extend(self.text.strip().split())
        else:
            for child in self.children:
                words.extend(child.all_texts(max_words=max_words-len(words)))
                if len(words) >= max_words:
                    break
        return words[:max_words]

    @property
    def right(self):
        return self.left + self.width

    @property
    def bottom(self):
        return self.top + self.height

    @property
    def left_offset(self):
        return self.left - (self.parent.left if self.parent else 0)

    @property
    def top_offset(self):
        return self.top - (self.parent.top if self.parent else 0)

    @property
    def prev_sibling(self):
        if not self.parent:
            return None
        siblings = self.parent.children
        idx = siblings.index(self)
        return self.parent.children[idx - 1] if idx > 0 else None

    @property
    def next_sibling(self):
        if not self.parent:
            return None
        siblings = self.parent.children
        idx = siblings.index(self)
        return self.parent.children[idx + 1] if idx + 1 < len(siblings) else None

    def style(self, key):
        assert self.tag != 't', 'Cannot call style on text nodes'
        try:
            return self.style_overrides[key]
        except KeyError:
            return self.web_page.common_styles[key]

    ################################################
    # Relationship to other nodes

    def is_leaf(self):
        return not self.children

    def neighbors(self, max_jumps=1):
        """Return the set of refs of neighboring nodes.

        Args:
            max_jumps (int): Maximum number of hops
        Returns:
            set[int]
        """
        neighbors = []
        if self.parent:
            neighbors.append(self.parent)
            neighbors.extend(self.parent.children)
        neighbors.extend(self.children)
        answer = set()
        for node in neighbors:
            if max_jumps == 1:
                if hasattr(node, 'ref'):
                    answer.add(node.ref)
            else:
                answer.update(node.neighbors(max_jumps-1))
        return answer

    @cached_property
    def ancestor_path(self):
        """Returns the path from root to self (list[Node])."""
        path = [self]
        curr = self
        while curr.parent:
            path.append(curr)
            curr = curr.parent
        return path[::-1]

    @property
    def depth(self):
        return len(self.ancestor_path)


class WebPage(object):
    """Represent a web page."""

    def __init__(self, filename):
        self.filename = filename
        ref_to_node = {}
        opener = gzip.open if filename.endswith('.gz') else open
        with opener(filename) as fin:
            data = json.load(fin)
        assert (isinstance(data['metadata'], dict)
                and isinstance(data['info'], list)
                and isinstance(data['common_styles'], dict))
        self.metadata = data['metadata']
        self._add_node(0, data['info'], ref_to_node)
        # Reassign the refs; ignore text nodes
        self.nodes = []
        self.text_nodes = []
        self.old_ref_to_new_ref = {}
        for old_ref, node in sorted(ref_to_node.items()):
            node.old_ref = old_ref
            if node.tag != 't':
                node.ref = len(self.nodes)
                self.nodes.append(node)
                self.old_ref_to_new_ref[old_ref] = node.ref
            else:
                self.text_nodes.append(node)
        # Compute global statistics
        self.page_width = max(1., float(data['metadata']['dimensions']['clientWidth']))
        self.page_height = max(1., float(max(node.top + node.height for node in self.nodes)))
        for node in self.nodes:
            node.x_ratio = min(1., max(0., (node.left + node.width / 2) / self.page_width))
            node.y_ratio = min(1., max(0., (node.top + node.height / 2) / self.page_height))
            node.visible = (node.width > 0 and node.height > 0)
        # Styles
        self.common_styles = data['common_styles']

    def _add_node(self, index, data, ref_to_node):
        """Construct a Node and add to the node list.

        Ignore nodes that:
        - are children of SVG

        Return the node is successfully constructed; None otherwise.
        """
        raw_info = data[index]
        #if not raw_info['height'] and not raw_info['width']:
        #    return
        node = Node(raw_info, self)
        ref_to_node[index] = node
        if raw_info['tag'] != 'svg':
            for child_index in raw_info.get('children', []):
                child_node = self._add_node(child_index, data, ref_to_node)
                if child_node:
                    node.add_child(child_node)
        return node

    def visualize(self, join=True):
        return self.nodes[0].visualize(join=join)

    def __iter__(self):
        return iter(self.nodes)

    def __getitem__(self, index):
        return self.nodes[index]

    @cached_property
    def graph(self):
        return GraphCreator(self).graph

    @cached_property
    def xid_to_ref(self):
        """A dict for converting xid in phrase-node dataset to ref."""
        xid_to_ref = {}
        for node in self.nodes:
            if hasattr(node, 'xid'):
                xid_to_ref[node.xid] = node.ref
        return xid_to_ref

    def overlap_eval(self, true, pred):
        """Compute the precision, recall, and f1 or the predicted element
        in terms of rendered area.

        Args:
            true (Node or int): Target Node object or Node's ref
            pred (Node or int): Predicted Node object or Node's ref
        Returns:
            precision, recall, f1
        """
        if true is None or pred is None:
            return 0., 0., 0.
        true_node = true if isinstance(true, Node) else self.nodes[true]
        pred_node = pred if isinstance(pred, Node) else self.nodes[pred]
        true_area = rect_area(true_node)
        pred_area = rect_area(pred_node)
        if true_area == 0 or pred_area == 0:
            return 0., 0., 0.
        overlap_area = rect_overlap(true_node, pred_node)
        precision = overlap_area / pred_area
        recall = overlap_area / true_area
        if precision == 0 or recall == 0:
            return precision, recall, 0.
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    SPATIAL_RELATIONS = {
            GraphRels.ABOVE: 0,
            # GraphRels.BELOW: 1,
            GraphRels.LEFT: 1,
            # GraphRels.RIGHT: 3
            }

    def get_spatial_neighbors(self):
        """Get indices of the spatial neighbors of each node.
        Will get at most one node from each direction.

        The order is [above, below, left, right]

        Returns:
            neighbors (list[list[int]] of size len(self.nodes) x 4)
                neighbors[i][j] contains the index of the neighbor #j of node i
                (j comes from SPATIAL_RELATIONS)
                Use 0 if the neighbor is not present
            masks (list[list[bool]]) of size len(self.nodes) x 4)
                masks[i][j] indicates if the neighbor #j of node i is present
        """
        G = self.graph
        neighbors = [[0] * len(self.SPATIAL_RELATIONS) for _ in xrange(len(self.nodes))]
        masks = [[0] * len(self.SPATIAL_RELATIONS) for _ in xrange(len(self.nodes))]
        for src, tgts in G.nodes.iteritems():
            for tgt, rels in tgts.iteritems():
                for rel in rels:
                    if rel not in self.SPATIAL_RELATIONS:
                        continue
                    rel_idx = self.SPATIAL_RELATIONS[rel]
                    if not masks[src][rel_idx]:
                        neighbors[src][rel_idx] = tgt
                        masks[src][rel_idx] = 1
        return neighbors, masks


################################################

def check_web_page(web_page, max_nodes=7000):
    """Check if the web page is good to be used for training.

    Args:
        web_page (WebPage)
        max_nodes (int)
    Returns:
        boolean
    """
    if len(web_page.nodes) == 1:
        logging.warn('%s has only 1 node; skip', web_page.filename)
        return False
    if len(web_page.nodes) > max_nodes:
        logging.warn('%s has too many nodes (%d > %d); skip',
                web_page.filename, len(web_page.nodes), max_nodes)
        return False
    if len(web_page.graph.nodes) < 2:
        logging.warn('%s has less than 2 graph nodes; skip', web_page.filename)
        return False
    return True


################################################
# Graph

class Graph(object):
    """Directed multi-graph.

    Things in |nodes| are not read-only, but please don't change it.
    """

    def __init__(self):
        self._nodes = {}

    def add_edge(self, s, t, label):
        """Add an edge.

        Args:
            s (any hashable)
            t (any hashable)
            label (any hashable)
        """
        s_targets = self._nodes.setdefault(s, {})
        s_targets.setdefault(t, set()).add(label)

    @property
    def nodes(self):
        return self._nodes

    def __iter__(self):
        return iter(self._nodes)


class GraphCreator(object):
    """Create a graph relating the nodes in the web page.
    Note: Ignore all "t" pseudo-nodes.
    """

    def __init__(self, web_page, max_pixels=50., visual_on_leaf_only=True):
        """Create a new GraphCreator.

        Args:
            web_page (WebPage or str): If a string is supplied, create a new
                WebPage using the string as the filename.
            max_pixels (int): Maximum pixel distance for two nodes to be
                visual neighbors
            visual_on_leaf_only (bool): Whether to only compute visual edges
                on the leaf nodes
        """
        if isinstance(web_page, basestring):
            from phrasenode.webpage import WebPage
            web_page = WebPage(web_page)
        self.web_page = web_page
        self.max_pixels = max_pixels
        self.visual_on_leaf_only = visual_on_leaf_only
        self.G = Graph()
        self._create_logical_edges(web_page[0])
        self._create_visual_edges()

    @property
    def graph(self):
        return self.G

    ################################

    def _create_logical_edges(self, node):
        non_t = [x for x in node.children if x.tag != 't']
        for child in non_t:
            self.G.add_edge(node.ref, child.ref, GraphRels.CHILD)
            self.G.add_edge(child.ref, node.ref, GraphRels.PARENT)
            self._create_logical_edges(child)
        for i in xrange(len(non_t) - 1):
            self.G.add_edge(non_t[i].ref, non_t[i+1].ref, GraphRels.NSIB)
            self.G.add_edge(non_t[i+1].ref, non_t[i].ref, GraphRels.PSIB)

    def _create_visual_edges(self):
        """Create visual edges using a sweeping algorithm."""
        # Collect the box boundaries
        # vertical boundaries
        boundaries = []
        for node in self.web_page:
            if (node.visible and node.tag != 't' and
                    (not self.visual_on_leaf_only or node.is_leaf())):
                boundaries.append((node.left, True, node.top, node.bottom, node.ref))
                boundaries.append((node.right, False, node.top, node.bottom, node.ref))
        self._create_visual_edges_from_boundaries(
                boundaries, GraphRels.ABOVE, GraphRels.BELOW)
        # horizontal boundaries
        boundaries = []
        for node in self.web_page:
            if (node.visible and node.tag != 't' and
                    (not self.visual_on_leaf_only or node.is_leaf())):
                boundaries.append((node.top, True, node.left, node.right, node.ref))
                boundaries.append((node.bottom, False, node.left, node.right, node.ref))
        self._create_visual_edges_from_boundaries(
                boundaries, GraphRels.LEFT, GraphRels.RIGHT)

    def _create_visual_edges_from_boundaries(self, boundaries, name_prec, name_succ):
        boundaries.sort()
        segments = []
        for x, is_start, y0, y1, ref in boundaries:
            if is_start:
                segments.append((y0, True, ref))
                segments.append((y1, False, ref))
                segments.sort()
                i = segments.index((y0, True, ref))
                if (i > 0 and segments[i-1][1] is False
                        and y0 - segments[i-1][0] <= self.max_pixels):
                    self.G.add_edge(ref, segments[i-1][2], name_prec)
                    self.G.add_edge(segments[i-1][2], ref, name_succ)
                i = segments.index((y1, False, ref))
                if (i+1 < len(segments) and segments[i+1][1] is True
                        and segments[i+1][0] - y1 <= self.max_pixels):
                    self.G.add_edge(segments[i+1][2], ref, name_prec)
                    self.G.add_edge(ref, segments[i+1][2], name_succ)
            else:
                i = segments.index((y0, True, ref))
                j = segments.index((y1, False, ref))
                if (i > 0 and segments[i-1][1] is False
                        and j+1 < len(segments) and segments[j+1][1] is True
                        and segments[j+1][0] - segments[i-1][0] <= self.max_pixels):
                    self.G.add_edge(segments[j+1][2], segments[i-1][2], name_prec)
                    self.G.add_edge(segments[i-1][2], segments[j+1][2], name_succ)
                segments.remove((y0, True, ref))
                segments.remove((y1, False, ref))
