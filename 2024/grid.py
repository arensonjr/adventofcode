from __future__ import annotations

import abc
import dataclasses
import networkx
import unpackable

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
                self.grid.add_node(Pos(x, y), val=lines[y][x])

        for x in range(self.width):
            for y in range(self.height):
                p = Pos(x, y)
                for other in [o for o in p.adjacents() if self.is_neighbor(p, o)]:
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

    def is_neighbor(self, pos1, pos2):
        return pos1 in self and pos2 in self and pos1.taxicab(pos2) == 1

    def neighbors(self, pos):
        return self.grid.edges(pos)

    def __contains__(self, pos:Pos) -> bool:
        return 0 <= pos.x < self.width and 0 <= pos.y < self.height

    def regions(self):
        return list(networkx.connected_components(self.grid))

    def in_front_of(self, pos:Pos, vec:Vector, until=None) -> list[(Pos, str)]:
        """Returns the grid values if you were to stand at `pos` and face in the `vec` direction.
        
        If `until` is provided, this stops before finding one of those values."""
        line = []
        while (pos := pos + vec) in self and self[pos] != until:
            line.append((pos, self[pos]))
        return line

    def pretty(self):
        return '\n'.join(
            ''.join(self[Pos(x, y)] for x in range(self.width)) + f' {y}'
            for y in range(self.height)
        ) + '\n' + ''.join(str(x % 10) for x in range(self.width))

    def copy(self):
        return Grid(_height=self.height, _width=self.width, _grid=self.grid.copy())

@dataclasses.dataclass(frozen=True)
class Vector(unpackable.Unpackable):
    """Vector represented by {-1, 0, 1} for movement along each axis."""
    dx : int
    dy : int

    @staticmethod
    def from_arrow(arrow:str) -> Vector:
        return {
            '^': UP,
            '>': RIGHT,
            '<': LEFT,
            'v': DOWN,
        }[arrow]

    @staticmethod
    def of(dx:int, dy:int) -> Vector:
        match dx, dy:
            case -1, 0: return LEFT
            case 1, 0: return RIGHT
            case 0, -1: return UP
            case 0, 1: return DOWN
            case _, _: return Vector(dx, dy) 

    def scale(self, factor:int) -> Vector:
        return Vector(dx=self.dx*factor, dy=self.dy*factor)

    def __mul__(self, factor:int) -> Vector:
        return self.scale(factor)

    def __floordiv__(self, factor:int) -> Vector:
        return Vector(self.dx // factor, self.dy // factor)

    def __neg__(self) -> Vector:
        return self.reverse()

    def __add__(self, other) -> Pos:
        match other:
            case Pos(x, y): return Pos(self.x + x, self.y + y)
            case Vector(dx, dy): return Pos(self.x + dx, self.y + dy)

    def __sub__(self, other) -> Vector:
        return self + (-other)

    def __lt__(self, other:Vector):
        return (self.dx, self.dy) < (other.dx, other.dy)

    def reverse(self):
        return Vector.of(dx=-1*self.dx, dy=-1*self.dy)

    def turn(self, dir):
        if dir == LEFT: return Vector.of(self.dy, -self.dx)
        if dir == RIGHT: return Vector.of(-self.dy, self.dx)
        else: raise ValueError("Can't turn {dir}")

    def __repr__(self):
        return f'Vector({self.dx},{self.dy})'

@dataclasses.dataclass(frozen=True)
class Pos(unpackable.Unpackable):
    """Individual position in a grid."""
    x : int
    y : int

    def adjacents(self) -> list[int]:
        return [self.shift(vec) for vec in DIRECTIONS]

    def shift(self, vec:Vector) -> Pos:
        return Pos(self.x + vec.dx, self.y + vec.dy) 

    def __neg__(self) -> Pos:
        return Pos(-self.x, -self.y)

    def __add__(self, other) -> Pos:
        match other:
            case Pos(x, y): return Pos(self.x + x, self.y + y)
            case Vector(dx, dy): return Pos(self.x + dx, self.y + dy)

    def __sub__(self, other):
        match other:
            case Pos(x, y): return Vector(self.x - x, self.y - y)
            case Vector(dx, dy): return Pos(self.x - dx, self.y - dy)
            case otherwise: raise ValueError(f"Can't subtract {self} and {other}")

    def __lt__(self, other:Pos):
        return (self.x, self.y) < (other.x, other.y)

    def vector_to(self, other:Pos) -> Vector:
        def _sign(x): return (x > 0) - (x < 0)
        return Vector(_sign(other.x - self.x), _sign(other.y - self.y))

    def taxicab(self, other:Pos) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __repr__(self):
        return f'Pos({self.x},{self.y})'

    def through(self, other:Pos) -> list[Pos]:
        """Lists all points between self and other, inclusive.

        e.g.:
          Pos(1, 2).between(Pos(5, 2)) -> [(1,2), (2,2), (3,2), (4,2), (5,2)]
        """
        if not (self.x == other.x or self.y == other.y):
            raise ValueError(f'Points {self} and {other} are not in line with each other')

        distance = self.taxicab(other)
        direction = self.vector_to(other)
        pt = self
        points = []
        for _ in range(distance+1):
            points.append(pt)
            pt += direction

UP = Vector(dx=0, dy=-1)
DOWN = Vector(dx=0, dy=1)
LEFT = Vector(dx=-1, dy=0)
RIGHT = Vector(dx=1, dy=0)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT]