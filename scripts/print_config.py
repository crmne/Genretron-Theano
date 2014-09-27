#!/usr/bin/env python
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import config

if __name__ == '__main__':
    config.print_config()
