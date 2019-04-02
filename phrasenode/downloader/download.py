#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil, re, argparse, json, gzip
import time, urllib, logging, traceback
from codecs import open
from itertools import izip
from collections import defaultdict, Counter

from phrasenode.downloader.instance import DownloaderInstance
from phrasenode.downloader.storage import WebPageStorage

join = os.path.join
# To use adblock, do something like
#   export ADBLOCK_PATH=/home/ppasupat/.config/google-chrome/Default/Extensions/cjpalhdlnbpafiamejdnhcphjbkeiagm/
# Change the directory to the Chrome Adblock extension
ADBLOCK_PATH = os.environ.get('ADBLOCK_PATH')
if os.path.exists(ADBLOCK_PATH):
    ADBLOCK_PATH = join(ADBLOCK_PATH, os.listdir(ADBLOCK_PATH)[0])


def download(storage_path, instance, storage, index, args):
    url = storage_path.url
    if (os.path.exists(storage.error_path(storage_path))
            or os.path.exists(storage.info_path(storage_path))):
        print u'[{}] Already downloaded {}'.format(index, url)
        return
    print u'[{}] Openning {}'.format(index, url)
    if args.auto_resize:
        instance.resize_window()    # Reset window size
    if not instance.visit(url):
        raise RuntimeError('Timeout')
    print 'Page opened.'
    time.sleep(args.sleep)
    if args.manual:
        reply = (raw_input('Is the page OK? (Y/n): ').lower()[:1] != 'n')
        if not reply:
            raise RuntimeError('Not OK')
    dimensions = instance.dimensions()
    if args.auto_resize and dimensions['scrollHeight'] > instance.WINDOW_HEIGHT:
        print 'Increasing window height to {}'.format(dimensions['scrollHeight'])
        instance.resize_window(height=dimensions['scrollHeight'])
        time.sleep(args.sleep)  # Wait for infinite scroll stuff
    dom_info = instance.get_dom_info()
    if len(dom_info['info']) <= 1:
        # Possibly the Chrome error page
        raise RuntimeError('len(dom_info) == {}'.format(len(dom_info['info'])))
    dom_source = instance.get_dom_html()
    if args.archive:
        dom_archived = instance.get_archived()
    metadata = {
        'timestamp': time.time(),
        'original_url': url,
        'redirected_url': instance.current_url(),
        'title': instance.current_title(),
        'dimensions': dimensions,
        }
    # Save data!
    with gzip.open(storage.info_path(storage_path), 'w') as fout:
        json.dump({
            'metadata': metadata,
            'common_styles': dom_info['common_styles'],
            'info': dom_info['info'],
        }, fout)
    with gzip.open(storage.html_path(storage_path), 'w') as fout:
        fout.write(dom_source.encode('utf8'))
    if args.archive:
        with open(storage.archive_path(storage_path), 'w') as fout:
            fout.write(dom_archived.encode('utf8'))
    if args.screenshot:
        instance.save_screenshot(storage.scrn_path(storage_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-L', '--localhost-port', type=int,
            help='Prepend each URL with http://127.0.0.1:[localhost_port]/')
    parser.add_argument('-s', '--start-index', type=int, default=0)
    parser.add_argument('-l', '--limit', type=int, default=100)
    parser.add_argument('-t', '--timeout', type=int, default=10)
    parser.add_argument('--sleep', type=int, default=2)
    parser.add_argument('-o', '--outdir', default='data/saved-web-pages-archived')
    parser.add_argument('-a', '--adblock', action='store_true')
    parser.add_argument('-H', '--headless', action='store_true')
    parser.add_argument('-S', '--screenshot', action='store_true')
    parser.add_argument('-m', '--manual', action='store_true',
            help='Will prompt before saving the page')
    parser.add_argument('-r', '--auto-resize', action='store_true')
    parser.add_argument('-i', '--infile',
            help='File with URLs, one per line')
    args = parser.parse_args()

    if not os.path.isdir(args.outdir):
        print 'Creating directory {}'.format(args.outdir)
        os.makedirs(args.outdir)
    storage = WebPageStorage(args.outdir,
            url_path_only=bool(args.localhost_port))

    instance = DownloaderInstance(
            headless=bool(args.headless),
            adblock=(ADBLOCK_PATH if args.adblock else None),
            timeout=args.timeout)

    with open(args.infile) as fin:
        for index, line in enumerate(fin):
            if index >= args.limit:
                print 'Limit reached.'
                break
            if index < args.start_index:
                continue
            url = line.strip()
            if args.localhost_port:
                url = 'http://127.0.0.1:{}/{}'.format(args.localhost_port, url)
            elif '://' not in url:
                url = 'http://' + url
            storage_path = storage.url_to_storage_path(url)
            storage.mkdir(storage_path)
            if args.archive:
                storage.make_archive_dir()
            try:
                download(storage_path, instance, storage, index, args)
            except Exception as e:
                print 'Something bad happened.'
                traceback.print_exc()
                with open(storage.error_path(storage_path), 'w', 'utf8') as fout:
                    print >> fout, 'ERROR', repr(e)
                    traceback.print_exc(file=fout)
                instance.reset()

    instance.close()
    

if __name__ == '__main__':
    main()
