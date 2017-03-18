#!/usr/bin/env python

from __future__ import division # allows for real number division
from subprocess import call, PIPE
from time import time

def runProg():
    # Runs the program and tracks the time each run takes
    times = []
    for x in range(1):
        start = time()
        call ("./matmult") # executes the command hiding any output
        end = time()
        times.append(end-start)
    return times

def displayTimes(times):
    # print each time and the mean
    print "Run# \t Time(sec)"
    counter = 1
    for t in times:
        print counter, "\t", t
        counter += 1
    print "Mean:\t", sum(times)/len(times)

if __name__ == "__main__": # if this is being executed from the command line
    times = runProg()
    displayTimes(times)
