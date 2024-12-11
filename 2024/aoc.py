import sys
import os
import functools
import collections
import time
import datetime

######################################################################

def traverse(width, height, boxes, start):
    vector = (0, -1) # up
    visited = set()

    (x, y) = start
    while 0 < x < width and 0 < y < height:
        # TODO: something!
        pass

    # TODO
    return False

def day6_part2(lines):
    # Parse input
    height = len(lines)
    width = len(lines[0])
    boxes = [(x, y) for x in range(width) for y in range(height) if lines[y][x] == '#']
    start = [(x, y) for x in range(width) for y in range(height) if lines[y][x] == '^'][0]
    candidate_boxes = [(x, y) for x in range(width) for y in range(height) if lines[y][x] == '.']

    # Traverse
    for new_box in candidate_boxes:


    return start

def day11_part1(lines): return day11(lines, 25)
def day11_part2(lines): return day11(lines, 75)
def day11(lines, iterations):
    # Parse
    rocks = {int(rock): 1 for rock in lines[0].split()}
    debug(f'{rocks=}')

    for i in range(iterations):
        newrocks = collections.defaultdict(int)
        for rock, count in rocks.items():
            if rock == 0:
                newrocks[1] += count
            elif len(strock := str(rock)) % 2 == 0:
                half = int(len(strock)/2)
                newrocks[int(strock[:half])] += count
                newrocks[int(strock[half:])] += count
            else:
                newrocks[rock * 2024] += count
        rocks = newrocks
        debug(f'After {i+1} iterations, {sum(rocks.values())=}')
    
    return sum(rocks.values())

######################################################################

# Boilerplate & helper functions

DEBUG = False

def debug(msg):
    if DEBUG:
        print(msg)

def main():
    global DEBUG

    [_, day, part, infile, debug] = sys.argv
    DEBUG = (debug == 'True')
    with open(infile, 'r') as f:
        contents = [line.strip() for line in f.readlines()]
        func = globals()[day + '_' + part]

        print(f'----- Executing {day} {part} -----')
        start = time.time()
        print(func(contents))
        elapsed = time.time() - start
        print(f'----- Elapsed: {datetime.timedelta(seconds=elapsed)} -----')

if __name__ == "__main__":
    main()