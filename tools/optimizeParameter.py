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

import argparse
import os
import sys
import time
import numpy as np
from subprocess import call

COMMAND_TEMPLATE = "python ../compreffor/pyCompressor.py /path/to/font.otf --chunkratio %f"

def time_run(value):
    command = COMMAND_TEMPLATE % value

    null = open(os.devnull, "wb")

    start_time = time.time()
    call(command.split(), stdout=null)
    run_time = time.time() - start_time

    return run_time

def minimize_runtime(start_val, stop_val, samples, passes):
    left = start_val
    right = stop_val
    step = float(right - left) / samples

    for i in range(passes):
        print "Testing range (%f, %f) with %f steps" % (left, right, step)
        values = np.arange(left, right + step, step)
        times = map(time_run, values)

        lowest = min(enumerate(times), key=lambda x: x[1])
        low_val, low_time = values[lowest[0]], lowest[1]
        print "Current lowest: %f with %gs" % (low_val, low_time)
        left = low_val - 2 * step
        while left <= 0: left += step 
        right = low_val + 2 * step
        step = float(right - left) / samples

    return low_val

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
    parser.add_argument('stop_val', help='What value to stop at', type=float)
    parser.add_argument('step', help='What value to step by (for plotter) or number of samples (for minimizer)', type=float)
    parser.add_argument('passes', help='How many passes to run (for minimizer)', type=int,
                        nargs="?")

    args = parser.parse_args()

    if not args.plot:
        assert args.passes != None, "passes argument required"
        min_val = minimize_runtime(args.start_val, args.stop_val, args.step, args.passes)
        print "Minimized time value: %f" % min_val
    else:
        plot_values(args.start_val, args.stop_val, args.step)
