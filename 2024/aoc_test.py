import unittest

import aoc

def lines(infile):
    with open(infile, 'r') as f:
        return [line.strip() for line in f.readlines()]

class AocTestInputs(unittest.TestCase):
    def test_day6_pt2(self):
        self.assertEqual(aoc.day6_part2(lines("day6_test.txt")), 6)

    def test_day11_pt1(self):
        self.assertEqual(aoc.day11_part1(lines("day11_test.txt")), 55312)

    def test_day12_pt1(self):
        self.assertEqual(aoc.day12_part1(lines("day12_test.txt")), 1930)

    def test_day12_pt2(self):
        self.assertEqual(aoc.day12_part2(lines("day12_test.txt")), 1206)

    def test_day13_pt1(self):
        self.assertEqual(aoc.day13_part1(lines("day13_test.txt")), 480)

    def test_day15_pt1(self):
        self.assertEqual(aoc.day15_part1(lines("day15_test.txt")), 10092)

    def test_day15_pt2(self):
        self.assertEqual(aoc.day15_part2(lines("day15_test.txt")), 9021)

    def test_day16_pt1(self):
        self.assertEqual(aoc.day16_part1(lines("day16_test.txt")), 7036)

    def test_day16_pt2(self):
        self.assertEqual(aoc.day16_part2(lines("day16_test.txt")), 45)

    def test_day19_pt1(self):
        self.assertEqual(aoc.day19_part1(lines("day19_test.txt")), 6)

    def test_day19_pt1_input(self):
        self.assertEqual(aoc.day19_part1(lines("day19_input.txt")), 265)

    def test_day19_pt2(self):
        self.assertEqual(aoc.day19_part2(lines("day19_test.txt")), 16)

    def test_day19_pt2_input(self):
        self.assertEqual(aoc.day19_part2(lines("day19_input.txt")), 752461716635602)

    def test_day20_pt1(self):
        self.assertEqual(aoc.day20(lines("day20_test.txt"), max_jump=2, min_savings=2), 44)

    def test_day20_pt2(self):
        self.assertEqual(aoc.day20(lines("day20_test.txt"), max_jump=20, min_savings=50), 285)

    def test_day21_pt1(self):
        self.assertEqual(aoc.day21_part1(lines("day21_test.txt")), 126384)

    def test_day21_pt2(self):
        self.assertEqual(aoc.day21_part2(lines("day21_test.txt")), 0)

if __name__ == "__main__":
    unittest.main()