"""Data storage for saved websites."""
import sys, os, re, json, glob, gzip
import urllib, urlparse

from collections import namedtuple
StoragePath = namedtuple('StoragePath', ['url', 'quoted_url', 'dirname'])


class WebPageStorage(object):
    """Represent a web page storage.

    Directory structure:
    basedir --> top-level domain --> first character --> files

    The files include:
    - info-[quoted_url].gz: gzipped JSON containing extracted page info
    - scrn-[quoted_url].png: screenshot
    """

    def __init__(self, basedir, url_path_only=False):
        """Create a new WebPageStorage.

        Args:
            basedir (str): Base directory for the storage
            url_path_only (bool): When generating a filename,
                only use the path portion of the URL
                (ignore URL scheme, netloc, etc.).
        """
        self.basedir = basedir
        self.url_whitelist = None
        self.url_path_only = url_path_only

    def url_to_storage_path(self, url, quoted=False):
        """Convert URL to StoragePath.

        Args:
            url (str)
            quoted (bool): Whether the input URL is quoted
        Returns:
            StoragePath
        """
        if quoted:
            quoted_url = url
            url = urllib.unquote(url)
        else:
            quoted_url = urllib.quote(url, '')
        hostname = urlparse.urlparse(url).hostname
        tld = hostname.split('.')[-1]
        prefix = hostname[:1]
        if self.url_path_only:
            path = urlparse.urlparse(url).path
            path = re.sub('^/', '', path)
            quoted_url = urllib.quote(path)
        return StoragePath(url, quoted_url, os.path.join(tld, prefix))

    def path_to_storage_path(self, path):
        """Convert file path into StoragePath."""
        assert not self.url_path_only, 'Cannot call path_to_storage_path when url_path_only is True.'
        filename = os.path.basename(path)
        match = re.match(r'^(?:info|scrn)-(.*)(?:\.gz|\.png)$', filename)
        quoted_url = match.group(1)
        return self.url_to_storage_path(quoted_url, quoted=True)

    def list_storage_paths(self):
        """Return the list of all storage paths.
        If whitelist is present, only use the URLs from the whitelist.
        """
        storage_paths = []
        for path in glob.glob(os.path.join(self.basedir, '*', '*', 'info-*.gz')):
            storage_path = self.path_to_storage_path(path)
            if not self.url_whitelist or storage_path in self.url_whitelist:
                storage_paths.append(storage_path)
        return storage_paths

    def set_url_whitelist(self, url_whitelist, quoted=False):
        """Restrict the URLs returned to this list."""
        self.url_whitelist = set(
                self.url_to_storage_path(url, quoted=quoted)
                for url in url_whitelist)

    def mkdir(self, storage_path):
        dirname = os.path.join(self.basedir, storage_path.dirname)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    def make_archive_dir(self):
        dirname = os.path.join(self.basedir, 'ARCHIVED')
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    def html_path(self, storage_path):
        return os.path.join(self.basedir, storage_path.dirname, 'html-' + storage_path.quoted_url + '.gz')

    def info_path(self, storage_path):
        return os.path.join(self.basedir, storage_path.dirname, 'info-' + storage_path.quoted_url + '.gz')

    def scrn_path(self, storage_path):
        return os.path.join(self.basedir, storage_path.dirname, 'scrn-' + storage_path.quoted_url + '.png')

    def error_path(self, storage_path):
        return os.path.join(self.basedir, storage_path.dirname, 'error-' + storage_path.quoted_url + '.txt')

    def archive_path(self, storage_path):
        url = re.sub('^.*://', '', storage_path.url).replace('/', '_')
        return os.path.join(self.basedir, 'ARCHIVED', url + '.html')
