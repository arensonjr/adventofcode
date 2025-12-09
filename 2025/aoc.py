# import bidict
import collections
import datetime
import functools
import itertools
import heapq
import logging
import math
# import networkx
# import operator
# import os
# import re
import shapely
# import sympy
import sys
import time
# import typing

from absl import app
from absl import flags
from absl.flags import FLAGS

# from grid import Pos, Vector, Grid, UP, DOWN, LEFT, RIGHT, DIRECTIONS
from grid import Grid, Pos

######################################################################


# 1544362560
def day9_part2(lines):
    coords = [(int(l[0]), int(l[1])) for l in [line.split(',') for line in lines]]

    legal_area = shapely.Polygon(coords)

    max_area = 0
    for i, (x1, y1) in enumerate(coords):
        for (x2, y2) in coords[i+1:]:
            rect = shapely.Polygon([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
            if legal_area.contains(rect):
                area = (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)
                if area >= max_area:
                    max_area = area

    return max_area


# 4771532800
def day9_part1(lines):
    coords = [(int(l[0]), int(l[1])) for l in [line.split(',') for line in lines]]

    max_area = 0
    for i, (x1, y1) in enumerate(coords):
        for (x2, y2) in coords[i+1:]:
            area = (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1)
            if area >= max_area:
                max_area = area

    return max_area


######################################################################

# 8141888143
def day8_part2(lines):
    points = [tuple(map(int, line.split(','))) for line in lines]

    # Calculate distances
    distances = []
    for i, pt1 in enumerate(points):
        for pt2 in points[i+1:]:
            if pt1 == pt2: continue

            distance = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2)
            heapq.heappush(distances, (distance, (pt1, pt2)))

    # Connect the first N
    next_component = len(points)
    components = {pt: i for i, pt in enumerate(points)}
    component_points : dict[int, set[tuple]] = {i: {pt} for pt, i in components.items()}

    while len(component_points) > 1:
        distance, (pt1, pt2) = heapq.heappop(distances)
        log.debug('Next connection: [%s] %s -> %s', distance, pt1, pt2)

        component1 = components[pt1]
        component2 = components[pt2]
        if component1 != component2:
            for pt in component_points[component1]:
                components[pt] = next_component
            for pt in component_points[component2]:
                components[pt] = next_component
            component_points[next_component] = component_points.pop(component1) | component_points.pop(component2)
            next_component += 1

        if len(component_points) == 1:
            log.debug('Finished! Final two points: %s %s', pt1, pt2)
            return pt1[0] * pt2[0]

    raise SystemError('Shouldnt have made it out of the loop!')

# 46398
def day8_part1(lines):
    points = [tuple(map(int, line.split(','))) for line in lines]

    # Calculate distances
    distances = []
    for i, pt1 in enumerate(points):
        for pt2 in points[i+1:]:
            if pt1 == pt2: continue

            distance = math.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2 + (pt1[2] - pt2[2])**2)
            heapq.heappush(distances, (distance, (pt1, pt2)))

    # Connect the first N
    next_component = len(points)
    components = {pt: i for i, pt in enumerate(points)}
    component_points : dict[int, set[tuple]] = {i: {pt} for pt, i in components.items()}
    for _ in range(1000):
        distance, (pt1, pt2) = heapq.heappop(distances)
        log.debug('Next connection: [%s] %s -> %s', distance, pt1, pt2)

        component1 = components[pt1]
        component2 = components[pt2]
        if component1 != component2:
            for pt in component_points[component1]:
                components[pt] = next_component
            for pt in component_points[component2]:
                components[pt] = next_component
            component_points[next_component] = component_points.pop(component1) | component_points.pop(component2)
            next_component += 1

    log.debug('Components: %s', component_points)
    log.debug('Sizes: %s', list(map(len, component_points.values())))
    sizes = map(len, component_points.values())
    top_3 = sorted(sizes, reverse=True)[:3]
    return math.prod(top_3)

######################################################################

# 305999729392659
def day7_part2(lines):
    grid = Grid(lines)

    start = grid.find('S')[0]

    # Split every time we find a '^'
    active_cols = {start.x: 1}
    for y in range(1, grid.height):
        new_cols = collections.defaultdict(int)
        for x, ct in active_cols.items():
            if grid[Pos(x, y)] == '^':
                new_cols[x+1] += ct
                new_cols[x-1] += ct
            else:
                new_cols[x] += ct
        active_cols = new_cols

    return sum(active_cols.values())



# 1660
def day7_part1(lines):
    grid = Grid(lines)

    start = grid.find('S')[0]

    # Split every time we find a '^'
    active_cols = {start.x}
    splits = 0
    for y in range(1, grid.height):
        new_cols = set()
        for x in active_cols:
            if grid[Pos(x, y)] == '^':
                new_cols.update({x - 1, x + 1})
                splits += 1
            else:
                new_cols.add(x)
        active_cols = new_cols

    return splits

######################################################################


# 13807151830618
def day6_part2(lines):
    numbers = lines[:-1]
    operators = lines[-1]

    log.debug('Numbers:')
    log.debug(numbers)

    total = 0
    problem_op = ''
    problem_numbers = []
    problem_fn = lambda _: 0
    for col in range(len(operators)):
        # Problems end with a blank
        op = operators[col]
        n = ''.join(row[col] for row in numbers)
        if op.isspace() and n.isspace():
            # Evaluate the old problem
            log.debug('Evaluating %s %s = %d', problem_op, problem_numbers, problem_fn(problem_numbers))
            total += problem_fn(problem_numbers)
            continue

        # New problems start with an operator
        if not op.isspace():
            problem_op = op
            problem_fn = sum if op == '+' else math.prod
            problem_numbers = []

        # Collect a new number
        n = int(n)
        log.debug('Column %d -> %s -> %d', col, [row[col] for row in numbers], n)
        problem_numbers.append(n)

    # Evaluate the last problem
    total += problem_fn(problem_numbers)

    return total


# 7098065460541
def day6_part1(lines):
    numbers = [[int(n.strip()) for n in line.strip().split()] for line in lines[:-1]]
    operators = [o.strip() for o in lines[-1].strip() if o.strip()]

    total = 0
    for col in range(len(operators)):
        operands = [row[col] for row in numbers]
        op = operators[col]

        fn = sum if op == '+' else math.prod

        total += fn(operands)

    return total


######################################################################


# 347338785050515
def day5_part2(lines):
    blank = lines.index('')
    ranges = [tuple(map(int, line.split('-'))) for line in lines[:blank]]

    sorted_ranges = [x for (lo, hi) in ranges for x in [(lo, '+'), (hi, '-')]]
    sorted_ranges.sort(key=lambda t: t[0])

    sorted_ranges = itertools.groupby(sorted_ranges, key=lambda x: x[0])

    ref_ct = 0
    lo = -1
    valid = 0
    for boundary, items in sorted_ranges:
        items = list(items)
        delta = sum(-1 if item[1] == '-' else 1 for item in items)
        log.debug('Boundary: %d -- delta=%d %s', boundary, delta, items)

        if delta > 0 and ref_ct == 0:
            lo = boundary
            log.debug('  - Resetting lo to %d', boundary)
        elif ref_ct + delta == 0 and ref_ct > 0:
            valid += boundary - lo + 1
            log.debug('  - [valid=%d] Finishing range from %d - %d', valid, lo, boundary)
        elif delta == 0 and ref_ct == 0:
            valid += 1
            log.debug('  - [valid=%d] Starting AND finishing range at %d', valid, boundary)

        ref_ct += delta
        log.debug('  - ref_ct=%d', ref_ct)

    return valid


# 520
def day5_part1(lines):
    blank = lines.index('')
    ranges = [tuple(map(int, line.split('-'))) for line in lines[:blank]]
    ids = list(map(int, lines[blank+1:]))

    valid = 0
    for i in ids:
        for low, high in ranges:
            if low <= i <= high:
                valid += 1
                break

    return valid


######################################################################


def day4_part2(lines):
    grid = Grid(lines=lines)

    # Keep removing accessible rolls until we reach a fixpoint
    removed = 0
    last_removed = -1
    while last_removed != removed:
        # Find accessible rolls
        rolls = set(grid.find('@'))
        accessible = set()
        for roll in rolls:
            if len(roll.surrounding() & rolls) < 4:
                accessible.add(roll)

        # Remove if accessible
        for roll in accessible:
            grid[roll] = '.'

        # Update removal count
        last_removed, removed = removed, removed + len(accessible)

        log.debug('Removed %d rolls (total so far: %d)', len(accessible), removed)

    return removed

def day4_part1(lines):
    grid = Grid(lines=lines)

    # How many rolls of paper are accessible (fewer than 4 adjacent)?
    rolls = set(grid.find('@'))
    accessible = set()
    for roll in rolls:
        if len(roll.surrounding() & rolls) < 4:
            accessible.add(roll)

    return len(accessible)

######################################################################

def day3_part2(lines):
    banks = [tuple(map(int, line)) for line in lines]

    joltages = []
    for bank in banks:
        log.debug('Calculating joltage of bank %s', bank)
        joltages.append(_calc_joltage(bank, 12))

    return sum(joltages)


@functools.cache
def _calc_joltage(remaining_bank, remaining_digits):
    prefix = (15 - len(remaining_bank)) * ' ' + '-'

    if not remaining_bank:
        log.debug('%s Failed to get enough digits; giving up', prefix)
        return -(10**12)

    if remaining_digits == 1:
        log.debug('%s Base case; finding the biggest remaining digit', prefix)
        return max(remaining_bank)

    log.debug('%s Finding joltage(%s, %d)', prefix, remaining_bank, remaining_digits)

    with_joltage = (remaining_bank[0] * 10**(remaining_digits-1)) + _calc_joltage(remaining_bank[1:], remaining_digits - 1)
    without_joltage = _calc_joltage(remaining_bank[1:], remaining_digits)

    log.debug('%s joltage(%s, %d) = %d', prefix, remaining_bank, remaining_digits, max(with_joltage, without_joltage))

    return max(with_joltage, without_joltage)


def day3_part1(lines):
    banks = [list(map(int, line)) for line in lines]

    jolts = []
    for bank in banks:
        log.debug('Processing battery bank %s', bank)

        little_max = bank[-1]
        jolt_max = 0
        for i in range(len(bank) - 2, -1, -1):
            jolt = bank[i] * 10 + little_max
            if jolt > jolt_max:
                jolt_max = jolt
            little_max = max(little_max, bank[i])

        log.debug('  - Biggest jolt is %d', jolt_max)
        jolts.append(jolt_max)

    return sum(jolts)

######################################################################

def day2_part2(lines):
    # Parse input
    ranges = [line.split('-') for line in lines[0].split(',')]

    # Split them into pieces to see what repeats
    multiples = set()
    for r in ranges:
        start, end = int('0' + r[0]), int('0' + r[1])
        log.debug('Processing range %d - %d', start, end)
        for repeat_len in range(1, len(r[1]) // 2 + 1):
            start_rep = r[0][:repeat_len]
            end_rep = r[1][:repeat_len]
            log.debug(f'  - {start_rep=} {end_rep=}')

            if end_rep < start_rep:
                reps = set(range(int(start_rep), int('9'*repeat_len)+1)) | set(range(int('1' + ('0' * (repeat_len-1))), int(end_rep) + 1))
                log.debug('  - Need to wrap around, checking all of %s instead', reps)
            else:
                reps = range(int('0' + start_rep), int('0' + end_rep) + 1)

            for rep in reps:
                # log.debug('  - Trying to repeat %d', rep)
                for rep_ct in range(max(2, len(r[0])//len(str(rep))), len(r[1])//len(str(rep)) + 1, 1):
                    candidate = int(str(rep) * rep_ct)
                    # log.debug('    * Repeated to %d', candidate)
                    if start <= candidate <= end:
                        multiples.add(candidate)
                        log.debug('    * [multiples=%d] Matched! Repeated %d to %d (rep_ct=%d)', len(multiples), rep, candidate, rep_ct)

    return len(multiples), sum(multiples)

# 23039913998
def day2_part1(lines):
    # Parse input
    ranges = [line.split('-') for line in lines[0].split(',')]

    # Split them into halves to see what repeats
    doubles = 0
    doubles_sum = 0
    for r in ranges:
        start_hi, start_lo = int('0' + r[0][len(r[0])//2:]), int('0' + r[0][:len(r[0])//2])
        end_hi, end_lo = int('0' + r[1][len(r[1])//2:]), int('0' + r[1][:(len(r[1])+1)//2]) 
        start, end = int(r[0]), int(r[1])
        log.debug('Processing range %d (%d/%d) - %d (%d/%d)', start, start_lo, start_hi, end, end_lo, end_hi)

        # If both lo are the same, the range is dependent on the hi (there's
        # only one starting number in the range, so there's only one possible
        # double)
        if start_lo == end_lo:
            log.debug('  - Starts match; checking numbers from %d to %d', start_hi, end_hi)
            if start_hi <= start_lo <= end_hi:
                doubles += 1
                doubles_sum += int(str(start_lo)*2)
                log.debug('    * [doubles=%d] Its in the range', doubles)

        # Otherwise, iterate through the possible starting numbers and check if they're in the range.
        else:
            log.debug('  - Starts dont match; searching from %d to %d', start_lo, end_lo)
            for option in range(start_lo, end_lo + 1):
                candidate = int(str(option)*2)
                if start <= candidate <= end:
                    doubles += 1
                    doubles_sum += candidate
                    log.debug('    * [doubles=%d] Found %d', doubles, candidate)

    return doubles_sum

######################################################################

# 5933
def day1_part2(lines):
    # Parse input
    directions = [(1 if line.startswith('R') else -1, int(line[1:])) for line in lines]
    dial = 50

    # Spin the dial and cound the zeros
    zeros = 0
    for direction in directions:
        log.debug('Direction %s spinning from %d', direction, dial)
        move = direction[1]
        bound = 0 if direction[0] == -1 else 100
        while move > 0:
            step = min(move, abs(bound - dial))
            if step == 0:
                step = min(move, 100)
            log.debug(f'{move=} {bound=} {dial=} {step=}')
            move -= step
            dial += direction[0] * step
            dial %= 100
            if dial == 0:
                zeros += 1
            log.debug('  - [zeros=%d] Stepping %d to %d', zeros, step, dial)

    return zeros


# 1021
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
        contents = [line.strip('\n') for line in f.readlines()]
        func = globals()[f'day{FLAGS.day}_part{FLAGS.part}']

        print(f'----- Executing day {FLAGS.day} part {FLAGS.part} -----')
        start = time.time()
        print(func(contents))
        elapsed = time.time() - start
        print(f'----- Elapsed: {datetime.timedelta(seconds=elapsed)} -----')

if __name__ == "__main__":
    app.run(main, sys.argv)
