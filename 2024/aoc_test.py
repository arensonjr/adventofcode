import unittest

import aoc

def run_func(func, infile):
    with open(infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        return func(lines)

class AocTestInputs(unittest.TestCase):
    def test_day6_pt2(self):
        self.assertEqual(run_func(aoc.day6_part2, "day6_test.txt"), 6)

    def test_day11_pt1(self):
        self.assertEqual(run_func(aoc.day11_part1, "day11_test.txt"), 55312)

    def test_day12_pt1(self):
        self.assertEqual(run_func(aoc.day12_part1, "day12_test.txt"), 1930)

    def test_day12_pt2(self):
        self.assertEqual(run_func(aoc.day12_part2, "day12_test.txt"), 1206)

    def test_day13_pt1(self):
        self.assertEqual(run_func(aoc.day13_part1, "day13_test.txt"), 480)

    def test_day15_pt1(self):
        self.assertEqual(run_func(aoc.day15_part1, "day15_test.txt"), 10092)

    def test_day15_pt2(self):
        self.assertEqual(run_func(aoc.day15_part2, "day15_test.txt"), 9021)

    def test_day16_pt1(self):
        self.assertEqual(run_func(aoc.day16_part1, "day16_test.txt"), 7036)

    def test_day16_pt2(self):
        self.assertEqual(run_func(aoc.day16_part2, "day16_test.txt"), 45)

    def test_day19_pt1(self):
        self.assertEqual(run_func(aoc.day19_part1, "day19_test.txt"), 6)

    def test_day19_pt1_input(self):
        self.assertEqual(run_func(aoc.day19_part1, "day19_input.txt"), 265)

    def test_day19_pt2(self):
        self.assertEqual(run_func(aoc.day19_part2, "day19_test.txt"), 16)

    def test_day19_pt2_input(self):
        self.assertEqual(run_func(aoc.day19_part2, "day19_input.txt"), 752461716635602)

if __name__ == "__main__":
    unittest.main()