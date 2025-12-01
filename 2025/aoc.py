import collections
import datetime
import functools
import heapq
import logging
import math
import networkx
import os
import re
import sympy
import sys
import time
import typing

from absl import app
from absl import flags
from absl.flags import FLAGS

from grid import Pos, Vector, Grid, UP, DOWN, LEFT, RIGHT, DIRECTIONS

def day1_part2(lines):
    # Parse input
    directions = [(1 if line.startswith('R') else -1, int(line[1:])) for line in lines]
    dial = 50

    # Spin the dial and cound the zeros
    zeros = 0
    for direction in directions:
        new_dial = direction[0] * direction[1] + dial
        zeros += abs((new_dial // 100) - (dial // 100))
        log.debug('[zeros=%d] Spun %s from %d -> %d (%d -> %d)', zeros, direction, dial, new_dial, dial % 100, new_dial % 100)
        dial = new_dial
        # new_dial = (direction[0] * direction[1]) + dial
        # log.debug('Direction %s spins %d -> %d', direction, dial, new_dial)

        # # Count the number of times we spin past zero
        # while new_dial >= 100:
        #     new_dial -= 100
        #     zeros += 1
        #     log.debug('  - [zeros=%d] Spun down %d -> %d', zeros, new_dial + 100, new_dial)
        # while new_dial < 0:
        #     new_dial += 100
        #     if dial != 0:
        #         zeros += 1
        #         # Reset dial (we don't use it anymore) so that we can not-count the first negative spin if we started on 0
        #         dial = 1
        #     log.debug('  - [zeros=%d] Spun up %d -> %d', zeros, new_dial - 100, new_dial)

        # if new_dial == 0 and (direction[0] * direction[1]) + dial % 100 == 0:
        #     zeros += 1
        #     log.debug('  - [zeros=%d] Ended on zero', zeros)

        # dial = new_dial

    return zeros


def day1_part1(lines):
    # Parse input
    directions = [(1 if line.startswith('R') else -1, int(line[1:])) for line in lines]
    dial = 50

    # Spin the dial and cound the zeros
    zeros = 0
    for direction in directions:
        dial = (direction[0] * direction[1] + dial) % 100
        log.debug('After %s the wheel is pointing at %d', direction, dial)

        # Count the number of times we end on zero
        if dial == 0:
            zeros += 1

    return zeros


######################################################################
#####               Boilerplate & helper functions               #####
######################################################################

flags.DEFINE_bool('debug', False, 'Enable debug print statements')
flags.DEFINE_integer('day', 1, 'Advent of Code day to execute')
flags.DEFINE_integer('part', 1, 'Part of each day (1 or 2) to execute')
flags.DEFINE_string('infile', '', 'Input file to run the code on')

log = logging.getLogger('aoc')

def main(_):
    if FLAGS.debug:
        log.setLevel(logging.DEBUG)

    with open(FLAGS.infile, 'r') as f:
        contents = [line.strip() for line in f.readlines()]
        func = globals()[f'day{FLAGS.day}_part{FLAGS.part}']

        print(f'----- Executing day {FLAGS.day} part {FLAGS.part} -----')
        start = time.time()
        print(func(contents))
        elapsed = time.time() - start
        print(f'----- Elapsed: {datetime.timedelta(seconds=elapsed)} -----')

if __name__ == "__main__":
    app.run(main, sys.argv)
