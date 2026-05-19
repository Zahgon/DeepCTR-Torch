# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,weichenswc@163.com
"""

import json
import logging
from threading import Thread

import requests

try:
    from packaging.version import parse
except ImportError:
    from pip._vendor.packaging.version import parse


def check_version(version):
    """Return version of package on pypi.python.org using json."""


    Thread(target=check, args=(version,)).start()
