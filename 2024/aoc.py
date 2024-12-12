import sys
import os
import functools
import collections
import time
import datetime

######################################################################

def day12_part1(lines):
    pass

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

def day6_part2(lines):
    # Parse input
    height = len(lines)
    width = len(lines[0])
    boxes = [(x, y) for x in range(width) for y in range(height) if lines[y][x] == '#']
    start = [(x, y) for x in range(width) for y in range(height) if lines[y][x] == '^'][0]
    candidate_boxes = [(x, y) for x in range(width) for y in range(height) if lines[y][x] == '.']

    boxes_by_y = collections.defaultdict(list)
    boxes_by_x = collections.defaultdict(list)
    for (x, y) in boxes:
        boxes_by_y[y].append(x)
        boxes_by_x[x].append(y)

    # Traverse
    looped = []
    for new_box in candidate_boxes:
        debug(f'Trying with a box at {new_box}...')
        boxes.append(new_box)
        boxes_by_y[new_box[1]].append(new_box[0])
        boxes_by_x[new_box[0]].append(new_box[1])

        if does_it_loop(width, height, boxes, boxes_by_y, boxes_by_x, start):
            debug('  -> Box LOOPED :)')
            looped.append(new_box)
        else:
            debug('  -> Box exited')

        boxes.pop()
        boxes_by_y[new_box[1]].pop()
        boxes_by_x[new_box[0]].pop()


    debug(f'{looped=}')
    return len(looped)

def does_it_loop(width, height, boxes, boxes_by_y, boxes_by_x, start):
    """Returns true if the guard loops, false otherwise."""
    (dx, dy) = (0, -1) # up
    visited = set()
    (x, y) = start

    while True:
        try:
            (boxX, boxY) = find_next_box(boxes_by_y, boxes_by_x, (x, y), (dx, dy))
        except:
            # No next box found --> off the edge of the map
            debug(f'  No box found for {(x, y)} at vector {(dx, dy)} - must have exited the map')
            return False

        (x, y) = (boxX - dx, boxY - dy)
        (dx, dy) = turn_right((dx, dy))
        if ((x, y), (dx, dy)) in visited:
            debug(f'  Already seen {(x,y)} at vector {(dx, dy)}, must have looped')
            return True # Looped
        visited.add(((x, y), (dx, dy)))


def turn_right(vector):
    match vector:
        case (0, dy): return (-dy, 0)
        case (dx, 0): return (0, dx)

def find_next_box(boxes_by_y, boxes_by_x, pos, vector):
    (x, y) = pos
    (dx, dy) = vector
    # debug(f'  ({boxes_by_y=}, {boxes_by_x=})')

    match vector:
        case (0, dy):
            candidate_boxes = boxes_by_x[x]
            (start, dir) = (y, dy)
        case (dx, 0):
            candidate_boxes = boxes_by_y[y]
            (start, dir) = (x, dx)
    
    candidate_boxes = list(filter(lambda box: box < start if dir < 0 else box > start, candidate_boxes))
    next_box = min(candidate_boxes, key=lambda box: abs(box-start))
    debug(f'  Walking from {pos} by {vector}, possible boxes are {list(candidate_boxes)} and I chose {next_box}')
    return (x, next_box) if dx == 0 else (next_box, y)

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