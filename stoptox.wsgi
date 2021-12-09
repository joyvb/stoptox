#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/home/rodolpho/Apps/stoptox")

from stoptox import app as application
application.secret_key = 'reallyhardtoguess'
