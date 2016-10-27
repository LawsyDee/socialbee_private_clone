# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 14:30:40 2016

@author: Laura Drummer
"""

from testing import CypherStatement
from testing import func_test

laura = CypherStatement("Laura")

laura.say_hello()

func_test(laura.name, 6)