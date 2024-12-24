import collections
import datetime
import functools
import heapq
import math
import networkx
import os
import re
import sys
import time
import typing

from grid import Pos, Vector, Grid, UP, DOWN, LEFT, RIGHT, DIRECTIONS

######################################################################

def day23_part1(lines):
    # Parse input
    g = networkx.Graph()
    for line in lines:
        one, two = line.split('-')
        g.add_edge(one, two)

    triplets = set()
    for node1 in [n for n in g.nodes if n.startswith('t')]:
        nbrs1 = set(g.neighbors(node1))
        for node2 in nbrs1:
            nbrs2 = set(g.neighbors(node2))

            triplets |= set(frozenset((node1, node2, n)) for n in nbrs1.intersection(nbrs2))

    debug(f'{triplets}')
    return len(triplets)

def day23_part2(lines):
    # Parse input
    g = networkx.Graph()
    for line in lines:
        one, two = line.split('-')
        g.add_edge(one, two)

    # Find complete subgraphs
    clique, size = networkx.max_weight_clique(g, weight=None)
    debug(f'{clique}')
    return ','.join(sorted(clique))

######################################################################

def day21_bfs(grid:Grid, start:Pos, end:Pos) -> set[str]:
    paths = collections.defaultdict(set)
    costs = collections.defaultdict(lambda: math.inf)
    queue = collections.deque([(start, 0, ())])
    while queue:
        next, cost, path = queue.popleft()
        if cost > costs[end]:
            break
        next_path = path + (next,)

        if not paths[next] == 0 or costs[next] == cost:
            costs[next] = cost
            paths[next] = paths[next] | {next_path}
        else:
            continue

        for _, nbr in grid.neighbors(next):
            queue.append((nbr, cost + 1, next_path))

    return {
        ''.join((p2-p1).to_arrow() for (p1, p2) in zip(path, path[1:])) + 'A'
        for path in paths[end]
    }

def cache_paths(grid:Grid) -> dict[tuple[str, str], set[str]]:
    cache = {}
    for start in grid.grid.nodes:
        for end in grid.grid.nodes:
            if grid[start] == '#' or grid[end] == '#':
                continue
            if grid[start] == grid[end]:
                cache[(grid[start], grid[end])] = {'A'}
            else:
                cache[(grid[start], grid[end])] = day21_bfs(grid, start, end)
    return cache

def day21(lines:list[str], steps:int):
    arrow_pad = GridWithWalls(['#^A', '<v>'])
    number_pad = GridWithWalls(['789', '456', '123', '#0A'])
    cache = cache_paths(arrow_pad) | cache_paths(number_pad) 

    # Defining an inner function so that `cache` isn't a cache key to the recursive fn
    @functools.lru_cache
    def _num_solutions(sequence:str, steps:int):
        if steps == 0:
            return len(sequence)

        sequence = 'A' + sequence
        cost = 0
        for i in range(len(sequence) - 1):
            options = cache[(sequence[i], sequence[i+1])]
            cost += min(_num_solutions(opt, steps-1) for opt in options)
        debug(f'num_solutions({sequence=}, {steps=}) = {cost}')
        return cost

    # return sum(_num_solutions(line, steps) for line in lines)
    total = 0
    for line in lines:
        cost = _num_solutions(line, steps)
        debug(f'===== cost({line=}): {cost} =====')
        total += cost
    return sum(_num_solutions(line, steps) * int(line[:-1]) for line in lines)

def day21_part1(lines): return day21(lines, steps=3)
def day21_part2(lines): return day21(lines, steps=26)

######################################################################

def day20_dijkstra(grid:Grid, start:Pos, end:Pos) -> tuple[dict[Pos, int], list[Pos]]:
    queue = collections.deque()
    queue.append((0, start, [start]))
    costs = {}
    while queue:
        cost, pos, path = queue.popleft()
        if pos in costs:
            continue
        costs[pos] = cost

        if pos == end:
            return cost, path

        # If not at the end, add all neighbors to the queue
        neighbors = grid.neighbors(pos)
        for (_, nbr) in neighbors:
            queue.append((cost + 1, nbr, path + [nbr]))

def day20(lines, max_jump=2, min_savings=1):
    # Parse
    grid = GridWithWalls(lines)
    start = grid.find('S')[0]
    end = grid.find('E')[0]
    
    debug(f'Starting grid:\n{grid.pretty()}\n')

    num_steps, path = day20_dijkstra(grid, start, end) 
    debug(f'Running fairly takes {num_steps} steps')

    # Sadly, making this more concise only helped optimize ~5-10%
    return len([
        savings
        for (i, jump_from) in enumerate(path)
        for (j, jump_to) in enumerate(path[i:])
        if (dist := jump_from.taxicab(jump_to)) <= max_jump and (savings := j - dist) >= min_savings
    ])

def day20_part1(lines): return day20(lines, max_jump=2, min_savings=100)
def day20_part2(lines): return day20(lines, max_jump=20, min_savings=100)

# Test version
# def day20_part1(lines): return day20(lines, max_jump=2, min_savings=1)
# def day20_part2(lines): return day20(lines, max_jump=20, min_savings=50)

######################################################################

def day19(lines):
    # Parse
    towels = set(map(str.strip, lines[0].split(',')))
    debug(f'{towels=}')
    problems = lines[2:]

    # Count solutions
    solvable = []
    for problem in problems:
        debug(f'{problem}:')
        possible_indices = collections.defaultdict(int)
        possible_indices[0] += 1
        for target in range(len(problem)):
            for prev in set(possible_indices.keys()):
                if problem[prev:target + 1] in towels:
                    possible_indices[target+1] += possible_indices[prev]
        if (num_solutions := possible_indices[len(problem)]) > 0:
            debug(f"  \\-> CAN make {problem} ({num_solutions})")
            solvable.append(num_solutions)
        else:
            debug(f"  \\-> cannot make {problem} ({possible_indices=})")

    return solvable

def day19_part1(lines): return len(day19(lines))
def day19_part2(lines): return sum(day19(lines))

######################################################################

class GridWithWalls(Grid):
    def is_neighbor(self, x, y):
        return super().is_neighbor(x, y) and self[x] != '#' and self[y] != '#'

def day16_dijkstra(grid, start, end):
    queue = []
    costs = collections.defaultdict(lambda: math.inf)
    all_paths = collections.defaultdict(set)
    heapq.heappush(queue, (0, (start, RIGHT, {start})))
    while queue:
        (cost, (pos, dir, path)) = heapq.heappop(queue)
        debug(f'POLLED {(cost, pos, dir)=}')

        # Update the cost of facing each direction at this spot
        all_worse_costs = True
        new_cost = {}
        for new_dir, delta in [(dir, 0), (dir.turn(LEFT), 1000), (dir.turn(RIGHT), 1000), (dir.reverse(), 2000)]:
            new_cost[new_dir] = cost + delta
        for new_dir in new_cost:
            if new_cost[new_dir] <= costs[(pos, new_dir)]:
                all_worse_costs = False
                debug(f'  |- updated cost of {pos} facing {new_dir} to {costs[(pos, new_dir)]} -> {new_cost[new_dir]}')
                costs[(pos, new_dir)] = new_cost[new_dir]
                all_paths[(pos, new_dir)].update(path)
            else:
                debug(f'  |- cost of walking to {pos} facing {new_dir} is {new_cost[new_dir]} > {costs[(pos, new_dir)]}')

        if all_worse_costs:
             debug(f'  \\--> (discarding {pos}, new costs are all greater than existing costs')
             continue

        if pos == end:
            # Find all other possible paths that would've gotten here
            while (other := heapq.heappop(queue))[0] == cost:
                (_, (opos, _, opath)) = other
                if opos == end:
                    path.update(opath)
            return (cost, path)


        # If not at the end, add all neighbors to the queue
        for (_, nbr) in grid.neighbors(pos):
            new_dir = nbr - pos
            debug(f'  |- enqueueing {nbr} with new cost {new_cost[new_dir] + 1}')
            next_cost = new_cost[new_dir] + 1
            new_paths = all_paths[(pos, dir)] | {nbr}
            heapq.heappush(queue, (next_cost, (nbr, new_dir, new_paths)))

        # if DEBUG:
        #     sys.stdin.readline()

def day16_part1(lines):
    # Parse input
    grid = GridWithWalls(lines)
    start = grid.find('S')[0]
    end = grid.find('E')[0]

    # Dijkstra from start to end
    cost, _ = day16_dijkstra(grid, start, end)
    return cost

def day16_part2(lines):
    # Parse input
    grid = GridWithWalls(lines)
    start = grid.find('S')[0]
    end = grid.find('E')[0]

    # Dijkstra from start to end
    cost, visited = day16_dijkstra(grid, start, end)

    if DEBUG:
        debug(f'\nFinal visited ({len(visited)}):\n{visited}')
        for pos in visited:
            grid[pos] = 'O'
        debug(f'\nOn the grid:\n{grid.pretty()}\n')
    return len(visited)

######################################################################

def day15_part1(lines:list[str]):
    # Parse
    empty = lines.index("")
    room = Grid(lines[:empty])
    moves = list(map(Vector.from_arrow, ''.join(lines[empty+1:])))

    # Simulate
    robot = room.find('@')[0]
    for move in moves:
        whats_up = room.in_front_of(robot, move, until='#')
        # Is there any space to move boxes? (if not, happily no-op into a wall)
        empties = [pos for (pos, val) in whats_up if val == '.']
        debug(f'{robot=} facing {move}: {whats_up=} -> {empties=}')
        if empties:
            # If so, everything shifts to the next empty space in order
            for distance in range(robot.taxicab(empties[0]), 0, -1):
                room[robot + move*distance] = room[robot + move*(distance-1)]
            # And finally, upkeep on the robot's old position
            room[robot] = '.'
            robot += move
        else:
            debug(f'No empty space found')

    # Compute
    debug(f'After all moves complete:\n{room.pretty()}')
    return sum([100*p.y + p.x for p in room.find('O')])

def day15_part2(lines:list[str]):
    # Parse
    empty = lines.index("")
    double = {'O': '[]', '@': '@.', '#': '##', '.': '..'}
    room = Grid([''.join(map(double.get, line)) for line in lines[:empty]])
    moves = list(map(Vector.from_arrow, ''.join(lines[empty+1:])))

    # Simulate
    robot = room.find('@')[0]
    for move_num, move in enumerate(moves):
        whats_up = room.in_front_of(robot, move, until='#')

        # If we're moving vertically, any boxes in the way also need to be
        # pushed along BOTH columns, which makes this way more complicated than
        # a horizontal move would be.
        if move in {UP, DOWN}:
            move_cols = {robot.x}
            y = robot.y

            # Try pushing in this direction until we reach a thing we can't push.
            # If it works, replace our current room with the hypothetical room.
            # If not, discard it and continue.
            # next_room = room.copy()
            next_room = {}
            next_room[robot] = '.'
            found_all_empty = False
            found_wall = False
            while not (found_all_empty or found_wall):
                next_y = y + move.dy
                next_row = {Pos(x, next_y) for x in move_cols}

                # Augment with all of the box-halves that we collect
                unmatched_left = {
                    right for pos in next_row
                    if room[pos] == '[' and (right := pos + RIGHT) not in next_row
                }
                unmatched_right = {
                    left for pos in next_row
                    if room[pos] == ']' and (left := pos + LEFT) not in next_row
                }
                next_row = next_row | unmatched_left | unmatched_right

                # Stop if any of the boxes we have to move would hit any wall
                if any(room[pos] == '#' for pos in next_row):
                    found_wall = True
                    break

                # We've established there's space for everything, so move the
                # current row into the next row
                for pos in next_row:
                    # Move the old thing into the new space, but if it's a box
                    # we newly picked up, then there was no "moved from" -- it
                    # just creates a vacuum where it used to be.
                    if pos.x in move_cols:
                        next_room[pos] = room[pos - move]
                    else:
                        next_room[pos] = '.'

                # Stop if everything moved into empty space
                if all(room[pos] == '.' for pos in next_row):
                    found_all_empty = True
                    break

                # Update which columns we have to move next time
                move_cols = {pos.x for pos in next_row if room[pos] != '.'}
                y = next_y

            if found_all_empty:
                robot = robot + move
                for pos, val in next_room.items():
                    room[pos] = val
            elif found_wall:
                # This means we failed to move, so don't use the temp locations
                # that we calculated earlier
                pass
            else:
                raise AssertionError('Somehow I did not find an empty row NOR walls!')

        else:
            # Otherwise, just do a normal move.
            empties = [pos for (pos, val) in whats_up if val == '.']
            if empties:
                for distance in range(robot.taxicab(empties[0]), 0, -1):
                    room[robot + move*distance] = room[robot + move*(distance-1)]
                # And finally, upkeep on the robot's old position
                room[robot] = '.'
                robot += move

        # As it turns out, printing out the whole room is REALLY slow,
        # and f-strings eagerly evaluate so `room.pretty()` runs every
        # time, even if debugging is off.
        if DEBUG:
            debug(f'After move {move_num}:\n{room.pretty()}\n')

    # Compute
    debug(f'After all moves complete:\n{room.pretty()}\n')
    return sum([100*p.y + p.x for p in room.find('[')])

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
    return len([neighbor for (x, y) in region for neighbor in neighbors(x, y) if neighbor not in region])

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
    grid = Grid(lines)
    start = grid.find('^')[0]

    # Preprocess box locations
    boxes_by_x = collections.defaultdict(list)
    boxes_by_y = collections.defaultdict(list)
    for box in grid.find('#'):
        boxes_by_x[box.x].append(box)
        boxes_by_y[box.y].append(box)

    looped = []
    for new_box in grid.find('.'):
        grid[new_box] = '#'
        boxes_by_x[new_box.x].append(new_box)
        boxes_by_y[new_box.y].append(new_box)

        if does_it_loop2(grid, start, boxes_by_x, boxes_by_y):
            looped.append(new_box)

        grid[new_box] = '.'
        boxes_by_x[new_box.x].pop()
        boxes_by_y[new_box.y].pop()

    return len(looped)

def does_it_loop2(grid:Grid, start:Pos, boxes_by_x:dict[int, Pos], boxes_by_y:dict[int, Pos]) -> bool:
    visited = set()
    vector = UP
    pos = start

    while True:
        # Attempt to find the closest box
        try:
            if vector is UP:
                next_box = max([box for box in boxes_by_x[pos.x] if box.y < pos.y], key=lambda pos: pos.y)
            elif vector is DOWN:
                next_box = min([box for box in boxes_by_x[pos.x] if box.y > pos.y], key=lambda pos: pos.y)
            elif vector is LEFT:
                next_box = max([box for box in boxes_by_y[pos.y] if box.x < pos.x], key=lambda pos: pos.x)
            elif vector is RIGHT:
                next_box = min([box for box in boxes_by_y[pos.y] if box.x > pos.x], key=lambda pos: pos.x)
            else:
                raise Exception(f'WTF {vector=}')
        except ValueError as e:
            # No next box -> we've reached the edge of the grid
            debug(f'  -> No loop')
            debug(e)
            return False # (a.k.a. no loop)

        # Otherwise, move forward
        pos = next_box - vector

        vector = vector.turn(RIGHT)
        # Loop check!
        if (pos, vector) in visited:
            debug(f'  -> Loop')
            return True
        visited.add((pos, vector))

######################################################################

def arbitrary(s: set[typing.Any]):
    for elem in s: return elem

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