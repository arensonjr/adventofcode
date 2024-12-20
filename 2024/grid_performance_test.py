import datetime
import time
import unittest

import aoc

def timed(func):
    start = time.time()
    func()
    end = time.time()
    return datetime.timedelta(seconds=end - start).total_seconds()

def run_func(name, infile):
    with open(infile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        return getattr(aoc, name)(lines)

class AocGridPerformanceTests(unittest.TestCase):
    def test_day16_pt2(self):
        # 19s with class-based grid
        # 16s with type aliases
        #
        # I feel like it's easier to do better here.
        self.assertLessEqual(
            timed(lambda: run_func("day16_part2", "day16_input.txt")),
            10
        )

    def test_day6_pt2(self):
        # ~4m20s with class-based grid
        # ~2m5s with type aliases
        #
        # I feel like it's easier to do better here.
        self.assertLessEqual(
            timed(lambda: run_func("day6_part2", "day6_input.txt")),
            10
        )

    def test_day20_pt2(self):
        # ~8s with class-based grid
        self.assertLessEqual(
            timed(lambda: run_func("day20_part2", "day20_input.txt")),
            10
        )

if __name__ == "__main__":
    unittest.main()