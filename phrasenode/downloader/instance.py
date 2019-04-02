import json
import logging
import os
import requests
import sys
import time
import traceback
import urlparse

from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


with open(os.path.join(os.path.dirname(__file__), 'get-dom-info.js')) as fin:
    GET_DOM_INFO = fin.read()

DIMENSIONS_SCRIPT = '''return {
outerHeight: window.outerHeight,
outerWidth: window.outerWidth,
innerHeight: window.innerHeight,
innerWidth: window.outerHeight,
clientHeight: document.documentElement.clientHeight,
clientWidth: document.documentElement.clientWidth,
scrollHeight: document.documentElement.scrollHeight,
scrollWidth: document.documentElement.scrollWidth,
};'''


class DownloaderInstance(object):
    """Interface between Python and Chrome driver via Selenium.
    Manages a single instance.
    """

    # Added some space for title bar
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 1024
    
    def __init__(self, headless=False, adblock=None, timeout=5):
        """Starts a new Selenium WebDriver session.

        Args:
            headless (bool): Whether to render GUI
            adblock (str): Path to adblocker
            timeout (int): Set page load timeout
        """
        self.headless = headless
        self.adblock = adblock
        self.timeout = timeout
        self.initialize()

    def initialize(self):
        options = webdriver.ChromeOptions()
        options.add_argument('disable-infobars')
        options.add_argument('user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36')
        if self.headless:
            options.add_argument('headless')
            options.add_argument('disable-gpu')
            #options.add_argument('no-sandbox')
        if self.adblock:
            options.add_argument('load-extension=' + self.adblock)
        self.driver = webdriver.Chrome(chrome_options=options)
        self.driver.set_window_size(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)
        if self.timeout:
            self.driver.set_page_load_timeout(self.timeout)
            self.driver.set_script_timeout(self.timeout * 2)

    def close(self):
        """Tear down the WebDriver."""
        try:
            self.driver.quit()
            logging.info('WebDriver successfully closed.')
        except Exception as e:
            logging.error('Error closing the WebDriver.')
            traceback.print_exc()

    def reset(self):
        self.close()
        self.initialize()

    def visit(self, url):
        """Go to a URL.

        Args:
            url (str)
        """
        try:
            self.driver.get(url)
            return True
        except TimeoutException:
            logging.warning('Page load timed out: %s', url)
            return False

    def current_url(self):
        return self.driver.current_url

    def current_title(self):
        return self.driver.title

    def dimensions(self):
        return self.driver.execute_script(DIMENSIONS_SCRIPT)
    
    def get_dom_html(self):
        """Get the DOM as HTML.

        Returns:
            str
        """
        return self.driver.page_source

    def get_dom_info(self):
        """Get the DOM information from the custom script.
        
        Returns:
            {'common_styles': {...}, 'info': {...}}
        """
        return self.driver.execute_script(GET_DOM_INFO)

    def save_screenshot(self, filename):
        """Save screenshot to the given filename."""
        self.driver.get_screenshot_as_file(filename)

    def resize_window(self, width=None, height=None):
        """Resize the window"""
        self.driver.set_window_size(
                width or self.WINDOW_WIDTH,
                height or self.WINDOW_HEIGHT)
