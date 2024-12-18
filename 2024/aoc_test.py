import unittest

import aoc

def run_func(name, infile):
    with open(infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        return getattr(aoc, name)(lines)

class AocTestInputs(unittest.TestCase):
    def test_day6_pt2(self):
        self.assertEqual(run_func("day6_part2", "day6_test.txt"), 6)

    def test_day11_pt1(self):
        self.assertEqual(run_func("day11_part1", "day11_test.txt"), 55312)

    def test_day12_pt1(self):
        self.assertEqual(run_func("day12_part1", "day12_test.txt"), 1930)

    def test_day12_pt2(self):
        self.assertEqual(run_func("day12_part2", "day12_test.txt"), 1206)

    def test_day13_pt1(self):
        self.assertEqual(run_func("day13_part1", "day13_test.txt"), 480)

    def test_day15_pt1(self):
        self.assertEqual(run_func("day15_part1", "day15_test.txt"), 10092)

    def test_day15_pt2(self):
        self.assertEqual(run_func("day15_part2", "day15_test.txt"), 9021)

    def test_day16_pt1(self):
        self.assertEqual(run_func("day16_part1", "day16_test.txt"), 7036)

    def test_day16_pt2(self):
        self.assertEqual(run_func("day16_part2", "day16_test.txt"), 45)

if __name__ == "__main__":
    unittest.main()