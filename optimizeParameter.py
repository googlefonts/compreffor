#!/usr/bin/env python

import argparse
import os
import sys
import time
import numpy as np
from subprocess import call

ACCURACY = 0.005

# COMMAND_TEMPLATE = "python cffCompressor.py /usr/local/google/home/sfishman/Documents/logotype.otf --chunkratio %f"
COMMAND_TEMPLATE = "python cffCompressor.py /usr/local/google/home/sfishman/Documents/KiloGram-decompress.otf --chunkratio %f"

def time_run(value):
    command = COMMAND_TEMPLATE % value

    null = open(os.devnull, "wb")

    start_time = time.time()
    call(command.split(), stdout=null)
    run_time = time.time() - start_time

    return run_time

def minimize_runtime(start_val):
    cur = start_val
    cur_time = time_run(cur)
    print "First time: %gs" % cur_time
    print
    last_time = float("inf")

    while cur_time < last_time:
        cur *= 2
        last_time = cur_time
        cur_time = time_run(cur)
        print "Exponential increase yielded %gs" % cur_time
        print

    left = cur / 4
    right = cur

    while right - left > ACCURACY:
        cur = left + (right - left) / 2
        midleft = left + (cur - left) / 2
        midright = cur + (cur - left) / 2
        midleft_time = time_run(midleft)
        midright_time = time_run(midright)

        print "Left (%.4f) time: %gs" % (midleft, midleft_time)
        print "Right (%.4f) time %gs" % (midright, midright_time)

        if midleft_time > midright_time:
            print "Moving Right"
            left = cur
        else:
            print "Moving Left"
            right = cur
        print

    return left + (right - left) / 2

def plot_values(start_val, stop_val, step):
    from matplotlib import pyplot

    values = np.arange(start_val, stop_val + step, step)
    times = map(time_run, values)
    pyplot.plot(values, times)
    pyplot.title("Time to run vs. Changing parameter")
    pyplot.xlabel("Parameter Value")
    pyplot.ylabel("Time to run (seconds)")
    pyplot.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Minimize runtime.')
    parser.add_argument('-p', help='Plot values', action='store_true', default=False,
                        required=False, dest="plot")
    parser.add_argument('start_val', help='What value to start at', type=float)
    parser.add_argument('stop_val', help='What value to stop at in plot only', type=float,
                        nargs="?")
    parser.add_argument('step', help='What value to step by in plot only', type=float,
                        nargs="?")

    args = parser.parse_args()

    if not args.plot:
        min_val = minimize_runtime(args.start_val)
        print "Minimized time value: %f" % min_val
    else:
        plot_values(args.start_val, args.stop_val, args.step)
