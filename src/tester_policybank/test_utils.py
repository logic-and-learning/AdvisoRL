import numpy as np
import os, json, math


def save_json(file, data):
    with open(file, 'w') as outfile:
        json.dump(data, outfile)

def read_json(file):
    with open(file) as data_file:
        data = json.load(data_file)
    return data

def get_precentiles_str(a):
    p25, p50, p75 = get_precentiles(a)
    p25 = "%0.2f"%p25
    p50 = "%0.2f"%p50
    p75 = "%0.2f"%p75
    return p25, p50, p75

def get_precentiles_in_seconds(a):
    p25, p50, p75 = reward2steps(get_precentiles(a))
    p25 = ("%0.2f"%(p25/10) if p25 > 0 else "----")
    p50 = ("%0.2f"%(p50/10) if p50 > 0 else "----")
    p75 = ("%0.2f"%(p75/10) if p75 > 0 else "----")
    return p25, p50, p75

def get_precentiles(a):
    p25 = float(np.percentile(a, 25))
    p50 = float(np.percentile(a, 50))
    p75 = float(np.percentile(a, 75))
    return p25, p50, p75

def reward2steps(rewards):
    ret = []
    for r in rewards:    
        if r > 0: r = round(math.log(r, 0.9))
        else: r = -1
        ret.append(r+1)
    return tuple(ret)

