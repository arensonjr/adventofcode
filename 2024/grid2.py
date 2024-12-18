from __future__ import annotations

import abc
import networkx

# Set up the type aliases for our usage
type Pos = tuple[int, int]
type Vector = tuple[int, int]

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]

def arrow_to_vector(arrow:str) -> Vector:
    return {
        '^': UP,
        '>': RIGHT,
        '<': LEFT,
        'v': DOWN,
    }[arrow]

def x(pos:Pos) -> int:
    return pos[0]

def y(pos:Pos) -> int:
    return pos[1]

def vector_to(start:Pos, end:Pos) -> Vector:
    """Not to scale. Just returns positive/negative 1 in the correct direction for each axis (or 0)."""
    (sx, sy) = start
    (ex, ey) = end
    def _sign(x):
        return (x > 0) - (x < 0)
    return (_sign(ex - sx), _sign(ey - sy))

def adjacents(pos:Pos) -> list[Pos]:
    (x, y) = pos
    return [(x+dx, y+dy) for (dx, dy) in DIRECTIONS]

def shift(pos:Pos, vec:Vector) -> Pos:
    (x, y) = pos
    (dx, dy) = vec
    return (x+dx, y+dy)

def sub(pos:Pos, vec:Vector) -> Pos:
    (dx, dy) = vec
    return shift(pos, (-dx, -dy))

def scale(vec:Vector, factor:int) -> Vector:
    (dx, dy) = vec
    return (dx*factor, dy*factor)

def turn(vec:Vector, dir:Vector) -> Vector:
    (dx, dy) = vec
    if dir == LEFT: return (dy, -dx)
    if dir == RIGHT: return (-dy, dx)
    else: raise ValueError("Can't turn {dir}")

def reverse(vec:Vector) -> Vector:
    if vec == UP: return DOWN
    elif vec == DOWN: return UP
    elif vec == LEFT: return RIGHT
    elif vec == RIGHT: return LEFT
    else:
        (dx, dy) = vec
        return (-1*dx, -1*dy)

def taxicab(start:Pos, end:Pos):
    (sx, sy) = start
    (ex, ey) = end
    return abs(ex-sx) + abs(ey-sy)

class Grid(abc.ABC):
    """Grid of space for Advent of Code.
    
    This is an abstract class, and is designed to be overridden """
    def __init__(self, lines:list[int]=None, _height=None, _width=None, _grid=None):
        # Secret copy constructor!
        if _height and _width and _grid:
            self._height = _height
            self._width = _width
            self.grid = _grid
            return

        self._height = len(lines)
        self._width = len(lines[0])

        self.grid = networkx.Graph()
        for x in range(self.width):
            for y in range(self.height):
                self.grid.add_node((x, y), val=lines[y][x])

        for x in range(self.width):
            for y in range(self.height):
                p = (x, y)
                for other in [o for o in adjacents((x, y)) if self.is_neighbor(p, o)]:
                    self.grid.add_edge(p, other)

    @property
    def height(self): return self._height

    @property 
    def width(self): return self._width

    def __getitem__(self, pos:Pos) -> str:
        return self.grid.nodes[pos]['val']

    def __setitem__(self, pos:Pos, val:str):
        self.grid.nodes[pos]['val'] = val

    def find(self, val:str) -> list[Pos]:
        return [pos for (pos, attrs) in self.grid.nodes.items() if attrs['val'] == val]

    @abc.abstractmethod
    def is_neighbor(self, pos1:Pos, pos2:Pos):
        """Determines if pos1 and pos2 are legal neighbors.
        
        Override this to change behavior (e.g. exclude "walls" in the grid).
        """
        return pos1 in self and pos2 in self

    def neighbors(self, pos:Pos) -> list[Pos]:
        return self.grid.edges(pos)

    def __contains__(self, pos:Pos) -> bool:
        (x, y) = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def regions(self):
        return list(networkx.connected_components(self.grid))

    def in_front_of(self, pos:Pos, vec:Vector, until=None) -> list[(Pos, str)]:
        """Returns the grid values if you were to stand at `pos` and face in the `vec` direction.
        
        If `until` is provided, this stops before finding one of those values.
        """
        line = []
        while (pos := shift(pos, vec)) in self and self[pos] != until:
            line.append((pos, self[pos]))
        return line

    def pretty(self):
        return '\n'.join(
            ''.join(self[(x, y)] for x in range(self.width)) + f' {y}'
            for y in range(self.height)
        ) + '\n' + ''.join(str(x % 10) for x in range(self.width))