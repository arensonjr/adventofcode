import collections
import datetime
import functools
import networkx
import os
import re
import sys
import time

######################################################################

def day13_part1(lines): return day13(lines, 0)
def day13_part2(lines): return day13(lines, 10000000000000)
# Solve system of equations:
#   let n = num A joysticks, k = num B joysticks
#   n * ax + k * bx = prizex
#     -> k = (-ax/bx)n + (prizex/bx)
#   n * ay + k * by = prizey
#     -> k = (-ay/by)n + (prizey/by)
# Set both definitions of `k` equal to each other:
#   (-ax/bx)n + (prizex/bx) = (-ay/by)n + (prizey/by)
#   (ay/by - ax/bx) n = (prizey/by - prizex/bx)
#   n = (prizey/by - prizex/bx) / (ay/by - ax/bx)
# (and therefore also: k = (prizey/ay - prizex/ax) / (by/ay - bx/ax))
#
# There's only one solution to this system of equations, so "the lowest" is a red herring
def day13(lines, prize_delta):
    _parse = lambda line, delta=0: [int(num)+delta for num in re.compile('.*X(?:\\+|=)(\\d+), Y(?:\\+|=)(\\d+)').match(line).groups()]
    games = [(_parse(a), _parse(b), _parse(prize, prize_delta)) for (a, b, prize) in every_n_lines(lines, 3, gap=True)]

    total_tokens = 0
    for ((ax, ay), (bx, by), (prizex, prizey)) in games:
        n = ((prizey / by) - (prizex / bx)) / ((ay / by) - (ax / bx))
        k = ((prizey / ay) - (prizex / ax)) / ((by / ay) - (bx / ax))
        if abs((int_n := round(n))- n) < 0.0001 and abs((int_k := round(k)) - k) < 0.0001:
            total_tokens += 3 * int_n + int_k

    return total_tokens

######################################################################

def neighbors(x, y):
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

def split_crops(grid):
    debug(f'Splitting crops: {grid=}')

    height = len(grid)
    width = len(grid[0])
    graph = networkx.Graph()
    for x in range(width):
        for y in range(height):
            graph.add_node((x, y))
            for (nx, ny) in neighbors(x, y):
                if 0 <= nx < width and 0 <= ny < height and grid[y][x] == grid[ny][nx]:
                    graph.add_edge((x, y), (nx, ny))

    return list(networkx.connected_components(graph))

def perimeter(region):
    debug(f'Finding perimeter of {region=}')
    # The perimeter is equal to the number of neighbors outside the region (non-unique, since we can border it on more than one side).
    return len(neighbor for (x, y) in region for neighbor in neighbors(x, y) if neighbor not in region)

def num_sides(region):
    debug(f'Finding number of sides of {region=}')
    # Trying YET ANOTHER different approach (with prodding from Girts):
    xs = [x for (x, _) in region]
    ys = [y for (_, y) in region]
    (min_x, max_x) = (min(xs), max(xs))
    (min_y, max_y) = (min(ys), max(ys))

    def _fences(minx, maxx, miny, maxy, region):
        total_fences = 0
        for x in range(minx, maxx + 1):
            prev_fence = (False, False)
            for y in range(miny, maxy + 1):
                if (x, y) not in region:
                    prev_fence = (False, False)
                    continue
                fence = ((x-1, y) not in region, (x+1, y) not in region)
                total_fences += sum(a and not b for (a, b) in zip(fence, prev_fence))
                prev_fence = fence
        return total_fences

    return (
        _fences(min_x, max_x, min_y, max_y, region) +
        _fences(min_y, max_y, min_x, max_x, {(y, x) for (x, y) in region})
    )

def day12_part1(lines): return day12(lines, perimeter)
def day12_part2(lines): return day12(lines, num_sides)
def day12(lines, perim_func):
    # Parse
    height = len(lines)
    width = len(lines[0])
    grid = {(x, y): lines[y][x] for x in range(width) for y in range(height)}

    # Compute
    # regions = split_crops(grid)
    regions = split_crops(lines)
    region_summaries = [(len(region), perim_func(region), grid[region.pop()]) for region in regions]
    debug(f'{region_summaries=}')
    return sum((area * perim) for (area, perim, letter) in region_summaries)

######################################################################

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

def every_n_lines(lines, n, gap=False):
    per_group = n
    if gap:
        per_group += 1
        lines += ['final gap line']
    grouped = zip(*[iter(lines)]*per_group)
    return list(grouped) if not gap else [group[:-1] for group in grouped]

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