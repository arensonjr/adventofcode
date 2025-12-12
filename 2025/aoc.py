# import bidict
import collections
import datetime
import functools
import itertools
import heapq
import logging
import math
import networkx
import numpy
# import operator
# import os
import re
import scipy.optimize
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


# 427
def day12_part1(lines):
    # Parse input
    presents, regions = _parse_present_regions(lines)

    # Recursively try to make them fit
    fits = 0
    for i, region in enumerate(regions):
        x, y = region[0][0], region[0][1]
        empty_region = ('.' * x,) * y
        log.debug('========== Region %d: %dx%d ==========', i, x, y)
        if _fill_presents(empty_region, region[1], presents):
            fits += 1
        global recursion_count
        log.debug('%d total fits, %d total recursion count', fits, recursion_count)

    return fits

recursion_count = 0
@functools.cache
def _fill_presents(region, present_counts, presents):
    global recursion_count
    recursion_count += 1
    if recursion_count % 10_000 == 0:
        log.debug(f'{recursion_count=} {sum(present_counts)=}\n{'\n'.join(region)}')
    # log.debug('=== Recursion: Trying to fill region with %s ===\n%s', present_counts, '\n'.join(region))
    # log.debug('=== Recursion: Trying to fill region with %s ===', present_counts)


    # Base case: No more presents to fit
    if all(p == 0 for p in present_counts):
        log.debug('Fits! Final region:\n%s', '\n'.join(region))
        return True

    # first_open = (-1, -1)
    # for y in range(len(region) - 2):
    #     for x in range(len(region[y]) - 2):
    #         if region[y][x] == '.':
    #             first_open = (x, y)
    #             break
    #     if first_open > (-1, -1):
    #         break
    # if first_open == (-1, -1):
    #     log.debug('  - No open spots:\n%s', '\n'.join(region))
    #     return False
    # x, y = first_open
    openings = [(x, y) for x in range(len(region[0])) for y in range(len(region)) if region[y][x] == '.']

    # Prune openings to make sure there's enough room for remaining presents
    remaining = 0
    for ct, present in zip(present_counts, presents):
        blocks = sum(1 if sq == '#' else 0 for row in present[0] for sq in row)
        remaining += ct * blocks
    # log.debug('%d remaining pips, openings = %s -> %s', remaining, openings, openings[:-remaining])
    openings = openings[:-remaining]

    # Inductive case: Try fitting each necessary present into the first unoccupied spot
    for (x, y) in openings:
        for i in range(len(present_counts)):
            if present_counts[i] == 0: continue
            # log.debug('Trying to fit present #%d into region at %d,%d', i, x, y)

            present = presents[i]
            for shape in present:
                # log.debug('Trying to fit shape in region at %d,%d:\n%s\n\n%s', x, y, '\n'.join(shape), '\n'.join(region))
                # Try to fit it in
                if _fits(region, x, y, shape):
                    # log.debug('  - Fits!')
                    new_region = list(region)
                    for j in range(y, y + 3):
                        new_region[j] = new_region[j][0:x] + ''.join(shape[j-y][i-x] if shape[j-y][i-x] == '#' else new_region[j][i] for i in range(x, x + 3)) + new_region[j][x+3:]
                    # log.debug(region)
                    # log.debug(new_region)
                    if _fill_presents(tuple(new_region), tuple(present_counts[k]-1 if k == i else present_counts[k] for k in range(len(present_counts))), presents):
                        return True

    # None of the attempts worked
    return False


def _fits(region, x, y, present):
    for i in range(3):
        for j in range(3):
            try:
                if present[j][i] == '#' and (y + j >= len(region) or x + i >= len(region[y+j]) or region[y + j][x + i] == '#'):
                    return False
            except IndexError as e:
                log.debug('Cant fit index %d %d into %s?', i, j, present)
                raise e
    return True



def _parse_present_regions(lines):
    # Present shapes
    presents = []
    for i in range(6):
        present = tuple(lines[i * 5 + 1 : i * 5 + 4])
        rotations = set()
        # Rotate and flip
        for p in [present, _flip(present)]:
            rotations.add(p)
            for _ in range(3):
                p = _rotate(p)
                rotations.add(p)
        presents.append(tuple(rotations))
    presents = tuple(presents)
    log.debug('Presents: %s', presents)

    # Regions
    regions = []
    num = '(\\d+)'
    region_pattern = re.compile(f'{num}x{num}: {num} {num} {num} {num} {num} {num}')
    for line in lines[30:]:
        parts = region_pattern.match(line)
        if parts is not None:
            parts = parts.groups()
            regions.append((tuple(int(n) for n in parts[0:2]), tuple(int(n) for n in parts[2:])))

    return presents, regions


def _rotate(present):
    """Rotates 90 degrees to the right and returns the resulting shape"""
    return tuple(''.join(l[i] for l in present[::-1]) for i in range(len(present[0])))

def _flip(present):
    return tuple(l[::-1] for l in present)



######################################################################


# 603
def day11_part1(lines):
    # Parse input
    digraph = collections.defaultdict(list)
    for line in lines:
        pieces = line.split()
        source = pieces[0][:-1]
        dests = pieces[1:]
        digraph[source] = dests

    # Graph search for numpaths
    paths = collections.defaultdict(int, {'you': 1})
    queue = collections.deque(['you'])
    visited = set()
    while queue:
        source = queue.popleft()
        if source in visited:
            continue
        visited.add(source)

        for nbr in digraph[source]:
            log.debug('Adding %d paths to %s->%s', paths[source], source, nbr)
            paths[nbr] += paths[source]
            queue.append(nbr)

    log.debug(paths)


    return paths['out']


# 380961604031372
def day11_part2(lines):
    # Parse input
    digraph = collections.defaultdict(set)
    digraph = networkx.digraph.DiGraph()
    for line in lines:
        pieces = line.split()
        source = pieces[0][:-1]
        for dest in pieces[1:]:
            digraph.add_edge(source, dest)

    # Graph search for numpaths through dac/fft
    paths = {'svr': (1, 0, 0, 0)}
    for source in networkx.topological_sort(digraph):
        invalid, dac, fft, both = paths[source]

        for nbr in digraph[source]:

            nbr_i, nbr_d, nbr_f, nbr_b = paths.get(nbr, (0, 0, 0, 0))
            if nbr == 'dac':
                log.debug('Found dac!')
                i, d, f, b = nbr_i, nbr_d + invalid + dac, nbr_f, nbr_b + fft + both
                log.debug('(%d, %d, %d, %d) + (%d, %d, %d, %d) -> (%d, %d, %d, %d)', nbr_i, nbr_d, nbr_f, nbr_b, invalid, dac, fft, both, i, d, f, b)
            elif nbr == 'fft':
                log.debug('Found fft!')
                i, d, f, b = nbr_i, nbr_d, nbr_f + invalid, nbr_b + dac + both
            else:
                i, d, f, b = nbr_i + invalid, nbr_d + dac, nbr_f + fft, nbr_b + both

            log.debug('Setting (%d, %d, %d, %d) paths to %s->%s', i, d, f, b, source, nbr)
            paths[nbr] = (i, d, f, b)

    log.debug(paths)

    return paths['out'][3]


######################################################################


# 17133
def day10_part2(lines):
    # Parse input (ignoring the lights)
    boards = []
    for line in lines:
        parts = line.split()
        joltages = tuple(int(x) for x in parts[-1][1:-1].split(','))
        # joltages = numpy.array(list(int(parts[-1][i+1]) for i in range(len(parts[-1])-2)))
        buttons = []
        for button in parts[1:-1]:
            numbers = set(int(x) for x in button[1:-1].split(','))
            # buttons.append(numpy.array(list(1 if i in numbers else 0 for i in range(joltages.size))))
            buttons.append(tuple(1 if i in numbers else 0 for i in range(len(joltages))))

        boards.append((joltages, buttons))

    log.debug('Boards: %s', boards)

    total_moves = 0
    maxsize_moves = 0
    for joltages, buttons in boards:
        result = scipy.optimize.milp(
                # Minimize the total number of button presses (equal weights)
                [1] * len(buttons),
                # Integer number of button presses
                integrality=[1] * len(buttons),
                # Minimize this system of equations:
                #   e.g. (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
                #    --> 0a + 0b + 0c + 0d + 1e + 1f = 3
                #        0a + 1b + 0c + 0d + 0e + 1f = 5
                #        0a + 0b + 1c + 1d + 1e + 0f = 4
                #        1a + 1b + 0c + 1d + 0e + 0f = 7
                # So, the nth coefficient in the kth equation is whether or not the nth
                # button increments the kth joltage.
                constraints=scipy.optimize.LinearConstraint(
                    A=[[button[i] for button in buttons] for i in range(len(joltages))],
                    # (equal bounds for an equality constraint)
                    lb=joltages,
                    ub=joltages,
                ))

        moves = int(sum(result.x))
        total_moves += moves
        log.debug(f'{result.success=} -- {moves} ({result.x=})')

        # log.debug('===== Starting: %s ===== %s', joltages, buttons)
        # moves = _fill_joltages2(joltages, buttons)
        # log.debug('%d moves to solve %s %s', moves, joltages, buttons)
        # if moves >= sys.maxsize:
        #     maxsize_moves += 1
        # total_moves += moves

    # log.debug('(%d maxsize moves)', maxsize_moves)
    return total_moves


def _fill_joltages2(joltage, buttons):
    # Base case: already done
    if sum(joltage) == 0:
        return 0

    # Pick the joltage with the fewest satisfying buttons to solve first
    i = min(range(len(joltage)), key=lambda x: sum(button[x] for button in buttons) + (0 if joltage[x] else sys.maxsize))

    # Otherwise: satisfy the smallest nonzero joltage, then satisfy the rest
    # i = min(range(len(joltage)), key=lambda x: joltage[x] or sys.maxsize)
    log.debug('  - Solving joltage[%d] for %s / %s', i, joltage, buttons)
    if any(joltage[i] > 0 and sum(b[i] for b in buttons) == 0 for i in range(len(joltage))):
        log.debug('    - Illegal state; buttons cant satisfy this')
        return sys.maxsize

    moves = [sys.maxsize]
    for button in buttons:
        if button[i] == 0: continue

        move = sys.maxsize
        presses = joltage[i]
        while move >= sys.maxsize and presses > 0:
            log.debug('    - Trying to press %s %d times for %s', button, presses, joltage)
            new_joltage = tuple(joltage[x] - presses * button[x] for x in range(len(joltage)))
            presses //= 2 # just in case we need to loop again
            if any(j < 0 for j in new_joltage):
                continue
            move = joltage[i] + _fill_joltages2(new_joltage, buttons - {button})

        moves.append(move)

    if moves == [sys.maxsize]:
        # log.debug('  - Unable to solve %s! Falling back', joltage)
        # return _fill_joltages(joltage, tuple(buttons))
        return sys.maxsize

    return min(moves)




@functools.cache
def _fill_joltages(joltage, buttons):
    """Returns the min number of moves to fill the remaining joltages."""
    if all(j == 0 for j in joltage):
        return 0

    if not buttons:
        # log.debug('Need to fill %s but buttons is %s! Impossible', joltage, buttons)
        return sys.maxsize

    moves = []
    # Try the best buttons first
    options = list(buttons)
    target = numpy.array(joltage) / max(joltage)
    options.sort(key=lambda b: sum(abs(target - b)))
    # log.debug('Sorted buttons for %s: %s', joltage, options)
    for button in options:
        npbutton = numpy.array(button)
        # try:
        #     presses = math.floor(min(joltage / npbutton))
        # except ValueError:
        #     presses = 0
        if any(joltage - npbutton < 0):
            continue

        moves.append(1 + _fill_joltages(tuple(joltage - npbutton), buttons))

    # log.debug('  - Solving %s took %s moves -> %d', joltage, moves, min([sys.maxsize] + moves))
    return min([sys.maxsize] + moves)



# 498
def day10_part1(lines):
    # Parse input (ignoring the joltages)
    boards = []
    for line in lines:
        parts = line.split()
        lights = {i if parts[0][i+1] == '#' else -1 for i in range(len(parts[0])-2)} - {-1}
        buttons = []
        for button in parts[1:-1]:
            buttons.append(frozenset(int(x) for x in button[1:-1].split(',')))

        boards.append((lights, buttons))

    log.debug('Boards: %s', boards)

    # Find matching combos
    total_moves = 0
    for lights, buttons in boards:
        best_moves = math.inf
        log.debug('Searching for %s', lights)
        for button_set in itertools.chain.from_iterable(itertools.combinations(buttons, i) for i in range(len(buttons)+1)):
            effective_lights = functools.reduce(set.symmetric_difference, button_set, set())
            log.debug('  - Trying: %s (%s)', effective_lights, list(button_set))
            if effective_lights == lights:
                moves = len(button_set)
                log.debug('  - IT WORKED! %d moves', moves)
                if moves < best_moves:
                    best_moves = moves

        total_moves += best_moves

    return total_moves





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
