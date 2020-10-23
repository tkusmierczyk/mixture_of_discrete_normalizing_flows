# -*- coding: utf-8 -*-
""" Auxiliary functions. """

import sys
import re
import numpy as np
import traceback
import json

import logging
logger = logging.getLogger(__name__)


def is_valid(w):
    if w is None: return False
    if np.isnan(np.sum(w)): return False
    if (w==float('inf')).any().item(): return False
    if (w==float('-inf')).any().item(): return False
    return True
    

def assert_valid(w, msg=""):
    assert is_valid(w), msg
  

def parse_args(s):    
    """ Converts a string of the form key=value1,key=[strval2] into a dictionary. """
    if s is None or s=="" or s=="-f": return {}
    s = s.replace("=", ":")
    s = s.replace("," , ",\"")
    s = s.replace(":" , "\":")
    s = s.replace("[" , "\"").replace("]" , "\"")
    s = "{\""+s+"}"
    return json.loads(s)


def parse_script_args():
    """ Converts script args of the form key=value1,key=[strval2] into a dictionary. """
    args_str = sys.argv[1] if len(sys.argv)>1 else "" 
    logger.info("parsing: <%s>" % args_str)
    return parse_args(args_str)


def parse_dict(string, entries_separator="/", key2val_separator="-"):
    def parse_key2value(pair):
        k,v = pair.split(key2val_separator)
        return k.strip(), v.strip()
    return dict(parse_key2value(k2v) for k2v in string.split("/")) 






