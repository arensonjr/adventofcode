package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

func day18_part1(lines []string) int {
	// Parse input
	heightstr, widthstr, _ := strings.Cut(lines[0], ",")
	height, _ := strconv.ParseInt(heightstr, 10, 64)
	width, _ := strconv.ParseInt(widthstr, 10, 64)
	firstN, _ := strconv.ParseInt(lines[1], 10, 64)
	boxes := make(map[Pos]bool)
	for i, line := range lines[2:] {
		// Problem description says to only take the first N boxes for some reason
		if int64(i) >= firstN {
			break
		}

		xstr, ystr, _ := strings.Cut(line, ",")
		x, _ := strconv.ParseInt(xstr, 10, 64)
		y, _ := strconv.ParseInt(ystr, 10, 64)
		debug("BOX: ", x, ", ", y)
		boxes[Pos{x: x, y: y}] = true
	}

	debug("After ", firstN, " boxes:")
	for y := range height {
		row := ""
		for x := range width {
			if boxes[Pos{x: x, y: y}] {
				row = row + "#"
			} else {
				row = row + "."
			}
		}
		debug(row)
	}
	debug()

	// And shortest path out of it with BFS
	return day18_bfs(width, height, Pos{x: 0, y: 0}, Pos{x: width - 1, y: height - 1}, boxes)
}

func day18_part2(lines []string) (int64, int64) {
	// Parse input
	heightstr, widthstr, _ := strings.Cut(lines[0], ",")
	height, _ := strconv.ParseInt(heightstr, 10, 64)
	width, _ := strconv.ParseInt(widthstr, 10, 64)

	// We're not using firstN in this part, we're finding our own
	// firstN, _ := strconv.ParseInt(lines[1], 10, 64)

	boxes := make(map[Pos]bool)
	for i, line := range lines[2:] {
		xstr, ystr, _ := strings.Cut(line, ",")
		x, _ := strconv.ParseInt(xstr, 10, 64)
		y, _ := strconv.ParseInt(ystr, 10, 64)
		debug("BOX: ", x, ", ", y)
		boxes[Pos{x: x, y: y}] = true

		if day18_bfs(width, height, Pos{x: 0, y: 0}, Pos{x: width - 1, y: height - 1}, boxes) < 0 {
			// Exploration failed! This is the first box.
			debug("  |- Exploration failed for ", i, "th box: ", x, ", ", y)
			return x, y
		}

	}

	// If we get through every box, something is wrong
	debug("Somehow every box worked?!")
	return -1, -1
}

func day18_bfs(width int64, height int64, start Pos, end Pos, boxes map[Pos]bool) int {
	visited := make(map[Pos]bool)
	queue := []Path{{pos: start, path: []Pos{}}}
	var next Path
	for {
		// If we've failed, return early
		if len(queue) == 0 {
			debug("FAILED: No path found")
			return -1
		}

		next, queue = queue[0], queue[1:]
		debug("Exploring ", next.pos, " and the rest of the queue is ", len(queue), " elements")
		debug("  |- Path to here is ", next.path)

		if next.pos == end {
			debug("  |- Fully explored! Path is ", append(next.path, next.pos))
			return len(next.path) // + 1
		}
		if visited[next.pos] {
			debug("  |- Already visited, leaving early")
			continue
		}
		visited[next.pos] = true

		nextPath := append(next.path, next.pos)
		for _, nbr := range next.pos.neighbors() {
			if !boxes[nbr] {
				if nbr.x >= 0 && nbr.x < width && nbr.y >= 0 && nbr.y < height {
					debug("  |- Neighbor ", nbr, " is a legal neighbor")
					queue = append(queue, Path{pos: nbr, path: nextPath})
				} else {
					debug("  |- Neighbor ", nbr, " is illegal because it's out of bounds")
				}
			} else {
				debug("  |- Neighbor ", nbr, " is an illegal neighbor because it's a box")
			}
		}
	}
}

// -----------------------------------------------

type Path struct {
	pos  Pos
	path []Pos
}

type Pos struct {
	x int64
	y int64
}

func (p Pos) neighbors() []Pos {
	return []Pos{
		{x: p.x + 1, y: p.y},
		{x: p.x - 1, y: p.y},
		{x: p.x, y: p.y + 1},
		{x: p.x, y: p.y - 1},
	}
}

func main() {
	funcname := flag.String("func", "", "Name of the function to run")
	infile := flag.String("input", "", "Name of the input file to pipe to --func")
	flag.BoolVar(&DEBUG, "debug", false, "Whether to include debug messages")
	flag.Parse()

	debug("Running ", *funcname, " on ", *infile, " in debug mode: ", DEBUG)

	lines := readfile(*infile)
	switch *funcname {
	case "day18_part1":
		fmt.Println(day18_part1(lines))
	case "day18_part2":
		fmt.Println(day18_part2(lines))
	}
}

func readfile(filename string) []string {
	out := []string{}

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		out = append(out, scanner.Text())
	}

	return out
}

func debug(args ...any) {
	if DEBUG {
		fmt.Println(fmt.Sprint(args...))
	}
}

var DEBUG bool
