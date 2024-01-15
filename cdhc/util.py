import re

from cdt.metrics import SHD
from cdt.metrics import SID

def precision(set_maybe, set_def, set_true):
    if set_def:
        weight_maybe = precision(set_maybe, set(), set_true)
    else:
        weight_maybe = 1
    try:
        prec = (weight_maybe*len(set_maybe.intersection(set_true)) + len(set_def.intersection(set_true))) / (weight_maybe*len(set_maybe) + len(set_def))
    except ZeroDivisionError:
        prec = 0
    return prec

def recall(set_maybe, set_def, set_true):
    weight_maybe = precision(set_maybe, set(), set_true)
    try:
        recall = (weight_maybe*len(set_maybe.intersection(set_true)) + len(set_def.intersection(set_true))) / len(set_true)
    except ZeroDivisionError:
        recall = 0
    return recall

def f1(prec, rec):
    try:
        f1 = 2 * prec*rec / (prec + rec)
    except ZeroDivisionError:
        f1 = 0
    return f1

def prec_rec_f1(set_maybe, set_def, set_true):
    prec = precision(set_maybe, set_def, set_true)
    rec = recall(set_maybe, set_def, set_true)
    f1_ = f1(prec, rec)
    return prec, rec, f1_


def _precision(s0, s1):
    s = set(s0).intersection(s1)
    return len(s)/len(s1)

def _recall(s0, s1):
    s = set(s0).intersection(s1)
    try:
        val = len(s)/len(s0)
    except ZeroDivisionError:
        val = 1
    return val

def _precision_recall(s1, s2):
    return _precision(s1, s2), _recall(s1, s2)

def metrics(G_inf, G_true):
    e0 = G_true.edges()
    e1 = G_inf.edges()
    prec, rec = _precision_recall(e1, e0)
    sid = SID(G_inf, G_true)
    sid_rev = SID(G_true, G_inf)
    shd = SHD(G_inf, G_true)
    return prec, rec, shd, sid, sid_rev

def _dot_list_parse(dot_list):
    maybe_conf = set()
    def_conf = set()
    causal = set()
    for i, x in enumerate(dot_list):
        head, tail = map(int, re.findall('\"(.*)\" -> \"(.*)?\"', x)[0])
        try:
            direction = re.findall('(?<=dir=).*?(?=,)', x)[0]
        except IndexError:
            direction = None
        tail_caus = re.findall('(?<=arrowtail=).*?(?=,)', x)[0]
        head_caus = re.findall('(?<=arrowhead=).*?(?=])', x)[0]
        if ((head_caus == "odot" or tail_caus == "odot")):
            maybe_conf.add((tail, head))
        if (head_caus == "normal" and tail_caus == "normal" and direction == "both"):
            def_conf.add((tail, head))
        if (direction is None) or (head_caus == 'none'):
            causal.add((tail, head))
        assert ((tail, head) in maybe_conf or (tail, head) in def_conf or (tail, head) in causal)

    return causal, maybe_conf, def_conf
