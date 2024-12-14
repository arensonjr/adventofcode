import java.io.File
import java.util.Arrays
import java.nio.file.Path

fun day14_part1(lines : List<String>) : Int {
    debug("${lines}")
    // Parse input
    val (height, width) = lines[0].split(' ').map(String::toInt)
    val pattern = """p=([0-9-]+),([0-9-]+) v=([0-9-]+),([0-9-]+)""".toRegex();
    val data = lines.drop(1).map(pattern::find).map {
        val (px, py, vx, vy) = it?.destructured ?: error("Non-matching input?! ${it}")
        listOf(px.toInt(), py.toInt(), vx.toInt(), vy.toInt())
    }

    // Figure out where they end up
    fun toQuadrant(x : Int, size : Int) : Int {
        val half = size / 2
        return when {
            x < half  -> -1
            x == half -> 0
            else      -> 1
        }
    }
    val endPositions = data.map {(px, py, vx, vy) -> listOf(Math.floorMod(px + vx * 100, width), Math.floorMod(py + vy * 100, height))}
    val endQuadrantCounts = endPositions
        .map { (x, y) -> listOf(toQuadrant(x, width), toQuadrant(y, height))}
        .filter { (x, y) -> x != 0 && y != 0 }
        .groupingBy { it }
        .eachCount()
    
    // Multiply them all together
    return endQuadrantCounts.values.reduce(Int::times)
}

fun day14_part2(lines : List<String>) : Int {
    debug("${lines}")
    // Parse input
    val (height, width) = lines[0].split(' ').map(String::toInt)
    val pattern = """p=([0-9-]+),([0-9-]+) v=([0-9-]+),([0-9-]+)""".toRegex();
    val data = lines.drop(1).map(pattern::find).map {
        val (px, py, vx, vy) = it?.destructured ?: error("Non-matching input?! ${it}")
        listOf(px.toInt(), py.toInt(), vx.toInt(), vy.toInt())
    }

    for (step in 0..(height*width)) {
        val endPositions = data.map {(px, py, vx, vy) -> listOf(Math.floorMod(px + vx * step, width), Math.floorMod(py + vy * step, height))}
        // Guess: If they make a christmas tree, each bot is in a unique position
        if (endPositions.toSet().size == endPositions.size) {
            // DEBUG: Print out the tree so I can see what it looks like?
            if (DEBUG) {
                val endSet = endPositions.toSet()
                for (y in 0..height-1) {
                    for (x in 0..width-1) {
                        print(if (endSet.contains(listOf(x, y))) 'X' else '.')
                    }
                    println()
                }
            }
            return step
        }
    }
    // If we didn't find one, we failed
    error("Fully-unique positions not found")
}

// ----------- Boilerplate & helper functions -----------

private var DEBUG = false

fun debug(msg : String) {
    if (DEBUG) {
        println(msg)
    }
}

private val fns = listOf(
    ::day14_part1,
    ::day14_part2,
).associateBy { it.name }

fun main(args: Array<String>) {
    val day = args[0]
    val part = args[1]
    val lines = File(args[2]).absoluteFile.readLines()

    DEBUG = args[3] != "False"

    var result = fns["${day}_${part}"]?.invoke(lines) ?: error("Unknown: ${day}_${part}")
    println(result)
}