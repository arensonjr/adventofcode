import sys
import os

def parse_input(lines):
    left = []
    right = []
    for line in lines:
        nums = tuple(map(int, line.split()))
        left.append(nums[0])
        right.append(nums[1])
    return left, right


def main():
    # Part 1
    left, right = parse_input(sys.stdin.readlines())
    left, right = sorted(left), sorted(right)
    print(sum(map(lambda pair: max(pair) - min(pair), zip(left, right))))

    # Part 2
    score = 0
    for i in left:
        score += i * right.count(i)
    print(score)

if __name__ == "__main__":
    main()