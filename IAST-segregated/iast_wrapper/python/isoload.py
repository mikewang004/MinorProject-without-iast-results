#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pyiast 
import scipy as sp
import os 
import subprocess

def load_raspa(temp, path = "../../../Raspa"):
    mol_names = []
    for root, dir, files in os.walk(path):
        for names in files:
            if str(temp) in names:
                mol_names.append(names.partition("-")[0]) 
    print(mol_names)

load_raspa(600)