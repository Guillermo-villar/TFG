# -*- coding: utf-8 -*-

# copyright GMV 2022
# created by ggnn

# ----------------------------------------------------------------------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ----------------------------------------------------------------------------------------------------------------------
import configparser
import os
import re


# ----------------------------------------------------------------------------------------------------------------------
# FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------------
def get_config(conf_file):
    """ Parse configuration .ini files.

    Args:
        conf_file: (string) file name with extension of of configuration .ini file

    Returns:
        outDict: (dict) dict with configuration parameters parsed from configuration file

    """
    parser = configparser.ConfigParser()
    parser.read(conf_file)

    outDict = dict()
    for section_name in parser.sections():
        for name, value in parser.items(section_name):
            outDict[name] = eval(re.sub(r'#.*','',value))

    return outDict


# ----------------------------------------------------------------------------------------------------------------------
def get_config_with_comments(conf_file):
    """ Parse configuration .ini files. including comments

    Args:
        conf_file: (string) file name with extension of of configuration .ini file

    Returns:
        outDict: (dict) dict with configuration parameters parsed from configuration file

    """
    parser = configparser.ConfigParser(allow_no_value=True, comment_prefixes='#')
    parser.read(conf_file)

    outDict = dict()
    for section_name in parser.sections():
        for name, value in parser.items(section_name):
            outDict[name] = eval(re.sub(r'#.*','',value))

    return outDict