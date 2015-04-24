#!/usr/bin/env python
#
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fontTools.ttLib import TTFont
from fontTools import cffLib
from fontTools.misc import psCharStrings
from fontTools.pens import basePen
import matplotlib.pyplot as plt
import functools
import itertools
import os
import argparse

"""
Prints out some stats about a set of fonts, mostly 
related to subroutines.

Dependencies:
  - matplotlib
  - fontTools

Usage:
>>> ./subr_grapher.py font1.otf font2.otf font3.otf cff_table.cff

NOTE: if the file extension is `cff`, it will be
interpreted as a raw CFF table.
"""

SINGLE_BYTE_OPS = set(['hstem',
                       'vstem',
                       'vmoveto',
                       'rlineto',
                       'hlineto',
                       'vlineto',
                       'rrcurveto',
                       'callsubr',
                       'return',
                       'endchar',
                       'blend',
                       'hstemhm',
                       'hintmask',
                       'cntrmask',
                       'rmoveto',
                       'hmoveto',
                       'vstemhm',
                       'rcurveline',
                       'rlinecurve',
                       'vvcurveto',
                       'hhcurveto',
                     # 'shortint',  # not really an operatr
                       'callgsubr',
                       'vhcurveto',
                       'hvcurveto'])

def tokenCost(token):
    """Calculate the bytecode size of a T2 Charstring token"""

    tp = type(token)
    if issubclass(tp, basestring):
        if token[:8] in ("hintmask", "cntrmask"):
            return 1 + len(token[9:])
        elif token in SINGLE_BYTE_OPS:
            return 1
        else:
            return 2
    elif tp == tuple:
        assert token[0] in ("hintmask", "cntrmask")
        return 1 + len(token[1])
    elif tp == int:
        if -107 <= token <= 107:
            return 1
        elif 108 <= token <= 1131 or -1131 <= token <= -108:
            return 2
        else:
            return 3
    elif tp == float:
        return 5
    assert 0

def get_cff(filename):
    if os.path.splitext(filename)[1] == '.cff':
        res = cffLib.CFFFontSet()
        res.decompile(open(filename), None)
        return res
    else:
        return TTFont(filename)['CFF '].cff

def get_cs_bytes(td, fds):
    count = 0
    for cs in td.GlobalSubrs:
        count += len(cs.bytecode)
    for fd in fds:
        try:
            for cs in fd.Private.Subrs:
                count += len(cs.bytecode)
        except AttributeError:
            pass
    for cs in td.CharStrings.values():
        count += len(cs.bytecode)
    return count

def print_n_subroutines(name, td, fds):
    print("%s:\n\tGlobal Subrs: %d" % (name, len(td.GlobalSubrs)))
    for i, fd in enumerate(fds):
        try:
            x = len(fd.Private.Subrs)
        except AttributeError:
            x = 0
        print("\tFD %d Subrs: %d" % (i, x))

def get_savings(td, fds):
    gsavings = [-(s.subr_cost + 2) if s.program else 0 for s in td.GlobalSubrs]
    lsavings = [[-(s.subr_cost + 2) if s.program else 0 for s in fd.Private.Subrs] for fd in fds]
    gusages = [0 for _ in td.GlobalSubrs]
    lusages = [[0 for _ in fd.Private.Subrs] for fd in fds]
    gbias = psCharStrings.calcSubrBias(td.GlobalSubrs)
    lbias = map(lambda fd: psCharStrings.calcSubrBias(fd.Private.Subrs)
                           if hasattr(fd.Private, 'Subrs') else 0,
                fds)
    
    def count_subr(idx, is_global, fdidx=-1):
        if is_global:
            gsavings[idx + gbias] += (td.GlobalSubrs[idx + gbias].subr_saving - tokenCost(idx) - 1)
            gusages[idx + gbias] += 1
            subr = td.GlobalSubrs[idx + gbias]
        else:
            assert fdidx >= 0
            lsavings[fdidx][idx + lbias[fdidx]] += (fds[fdidx].Private.Subrs[idx + lbias[fdidx]].subr_saving - tokenCost(idx) - 1)
            lusages[fdidx][idx + lbias[fdidx]] += 1
            subr = fds[fdidx].Private.Subrs[idx + lbias[fdidx]]

        # follow called subrs:
        for before, tok in zip(subr.program, subr.program[1:]):
            if tok == 'callgsubr':
                count_subr(before, True, fdidx)
            elif tok == 'callsubr':
                count_subr(before, False, fdidx)
    
    for g in td.charset:
        cs, sel = td.CharStrings.getItemAndSelector(g)
        for before, tok in zip(cs.program, cs.program[1:]):
            if tok == 'callgsubr':
                count_subr(before, True, sel)
            elif tok == 'callsubr':
                count_subr(before, False, sel)
                
    return ((gsavings, lsavings), (gusages, lusages))

def decompile_charstrings(td, fds):
    for cs in td.GlobalSubrs:
        cs.subr_cost = cs.subr_saving = len(cs.bytecode)
    for fd in fds:
        try:
            for cs in fd.Private.Subrs:
                cs.subr_cost = cs.subr_saving = len(cs.bytecode)
        except AttributeError:
            pass
    for g in td.charset:
        cs, sel = td.CharStrings.getItemAndSelector(g)
        cs.decompile()
    for cs in td.GlobalSubrs:
        if cs.program and cs.program[-1] == 'return':
            cs.subr_saving -= 1
    for fd in fds:
        try:
            for cs in fd.Private.Subrs:
                if cs.program and cs.program[-1] == 'return':
                    cs.subr_saving -= 1
        except AttributeError:
            pass

def get_raw_usages(td, fds):
    gusages = [0 for _ in td.GlobalSubrs]
    lusages = [[0 for _ in fd.Private.Subrs] for fd in fds]
    gbias = psCharStrings.calcSubrBias(td.GlobalSubrs)
    lbias = map(lambda fd: psCharStrings.calcSubrBias(fd.Private.Subrs)
                           if hasattr(fd.Private, 'Subrs') else 0,
                fds)
    gsels = [None for _ in td.GlobalSubrs]
    
    for g in td.charset:
        cs, sel = td.CharStrings.getItemAndSelector(g)
        for before, tok in zip(cs.program, cs.program[1:]):
            if tok == 'callgsubr':
                gusages[before + gbias] += 1
                gsels[before + gbias] = sel
            elif tok == 'callsubr':
                lusages[sel][before + lbias[sel]] += 1

    for cs, sel in zip(td.GlobalSubrs, gsels):
        for before, tok in zip(cs.program, cs.program[1:]):
            if tok == 'callgsubr':
                gusages[before + gbias] += 1
            elif tok == 'callsubr':
                lusages[sel][before + lbias[sel]] += 1

    for sel, fd in enumerate(fds):
        if hasattr(fd.Private, 'Subrs'):
            for cs in fd.Private.Subrs:
                for before, tok in zip(cs.program, cs.program[1:]):
                    if tok == 'callgsubr':
                        gusages[before + gbias] += 1
                    elif tok == 'callsubr':
                        lusages[sel][before + lbias[sel]] += 1

    return (gusages, lusages)

def main(filenames, show_graphs):
    names = map(os.path.basename, filenames)
    cffs = map(get_cff, filenames)
    tds = map(lambda f: f.topDictIndex[0], cffs)
    fds = map(lambda td: td.FDArray if hasattr(td, 'FDArray') else [], tds)

    n_bytes = map(get_cs_bytes, tds, fds)
    for name, b in zip(names, n_bytes):
        print("%s:\n\t%d bytes" % (name, b))

    map(decompile_charstrings, tds, fds)
    
    map(print_n_subroutines, names, tds, fds)

    sav_usag = map(get_savings, tds, fds)
    for name, (savings, usages) in zip(names, sav_usag):
        tot_savings = savings[0] + list(itertools.chain.from_iterable(savings[1]))
        tot_usages = usages[0] + list(itertools.chain.from_iterable(usages[1]))
        avg = float(sum(tot_savings)) / len(tot_savings)
        print("%s:\n\tAverage savings per subr: %f\n\tMax saving subr: %d\n\tMax usage subr: %d" % (name, avg, max(tot_savings), max(tot_usages)))

    if show_graphs:
        # plot subrs
        SHOW_START = 0
        SHOW_LEN = 200
        mins = []
        maxes = []
        plt.figure(0)
        for savings, usages in sav_usag:
            tot_savings = savings[0] + list(itertools.chain.from_iterable(savings[1]))
            plot_savings = sorted(tot_savings, reverse=True)[SHOW_START:SHOW_START+SHOW_LEN]
            plt.plot(range(len(plot_savings)), plot_savings)
            mins.append(min(plot_savings))
            maxes.append(max(plot_savings))
        plt.ylim([min(mins) - 1, max(maxes) + 1])
        plt.title("Subroutine Savings")
        plt.xlabel("Subroutine")
        plt.ylabel("Savings (bytes)")

        raw_usages = map(get_raw_usages, tds, fds)
        fig = 1
        for gusages, lusages in raw_usages:
            for idx, usages in zip(['Global'] + range(len(lusages)), [gusages] + lusages):
                if usages:
                    bias = psCharStrings.calcSubrBias(usages)
                    if bias == 1131:
                        orig_order_usages = usages[1024:1240] + usages[0:1024] + usages[1240:]
                    elif bias == 32768:
                        orig_order_usages = (usages[32661:32877] + usages[31637:32661] +
                                             usages[32877:33901] + usages[0:31637] + 
                                             usages[33901:])
                    else:
                        orig_order_usages = usages
                    plt.figure(fig)
                    plt.plot(range(len(orig_order_usages)), orig_order_usages, color='b')
                    plt.title("Subroutine usages for FD %s" % idx)
                    plt.axvline(215, 0, max(orig_order_usages), color='r')
                    plt.axvline(2263, 0, max(orig_order_usages), color='r')
                    plt.ylim([0, max(orig_order_usages)])
                    plt.xlim([0, len(orig_order_usages)])
                    fig += 1
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        description="""FontTools Compreffor will take a CFF-flavored
                                       OpenType font and automatically detect
                                       repeated routines and generate subroutines
                                       to minimize the disk space needed to
                                       represent a font.""")
    parser.add_argument('filenames', help="the path to font files", nargs='+')
    parser.add_argument('-g', '--show-graphs', help="show graphs", action='store_true',
                        default=False)

    kwargs = vars(parser.parse_args())

    main(**kwargs)
