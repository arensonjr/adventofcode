use std::collections::HashMap;
use std::env;
use std::fmt::Debug;
use std::fs::read_to_string;
use std::time::Instant;

/* ---------------------------------------------------- */

fn day22_part1(lines: &Vec<String>) -> String {
    let secret_numbers: Vec<i64> = lines
        .iter()
        .map(String::as_str)
        .map(str::parse::<i64>)
        .map(Result::unwrap)
        .collect();

    let mut total = 0;
    for mut secret_num in secret_numbers {
        debug(format!("Secret number: {}", secret_num));
        for _step in 0..2000 {
            secret_num = ((secret_num * 64) ^ secret_num) % 16777216; 
            secret_num = ((secret_num / 32) ^ secret_num) % 16777216; 
            secret_num = ((secret_num * 2048) ^ secret_num) % 16777216; 
            // debug(format!("  |- After {} steps: {}", step + 1, secret_num));
        }
        debug(format!("  |- After 2000 steps: {}", secret_num));
        total += secret_num;
    }

    return total.to_string();
}

fn day22_part2(lines: &Vec<String>) -> String {
    let secret_numbers: Vec<i64> = lines
        .iter()
        .map(String::as_str)
        .map(str::parse::<i64>)
        .map(Result::unwrap)
        .collect();

    let mut overall_delta_results: HashMap<Vec<i64>, i64> = HashMap::new();
    for mut secret_num in secret_numbers {
        let mut deltas = Vec::with_capacity(4);
        let mut delta_results: HashMap<Vec<i64>, i64> = HashMap::new();
        debug(format!("Secret number: {}", secret_num));
        for _step in 0..2000 {
            let mut next = secret_num;
            next = ((next * 64) ^ next) % 16777216; 
            next = ((next / 32) ^ next) % 16777216; 
            next = ((next * 2048) ^ next) % 16777216; 
            let delta = (next % 10) - (secret_num % 10);
            secret_num = next;
            // debug(format!("  |- After {} steps: {} -> {} ({})", step + 1, next, next % 10, delta));

            deltas.push(delta);
            if deltas.len() > 4 {
                deltas.remove(0);
            }
            if let Some(end) = deltas.rchunks(4.min(deltas.len())).next() {
                let last_four = end.to_vec();
                if !delta_results.contains_key(&last_four) {
                    delta_results.insert(last_four.clone(), secret_num % 10);
                }
            } 
        }
        for (k, v) in delta_results.iter() {
            let prior = overall_delta_results.get(k).unwrap_or(&0);
            overall_delta_results.insert(k.to_vec(), prior + v);
        }
    }

    return overall_delta_results.values().max().unwrap().to_string();
}

/* ---------------------------------------------------- */

fn combo(operand: i64, a: i64, b: i64, c: i64) -> i64 {
    return match operand {
        0..=3 => operand,
        4 => a,
        5 => b,
        6 => c,
        _ => unimplemented!("Unknown operand"),
    };
}

fn run_program(init_a: &i64, init_b: &i64, init_c: &i64, program: &Vec<i64>) -> Vec<i64> {
    let mut a: i64 = *init_a;
    let mut b: i64 = *init_b;
    let mut c: i64 = *init_c;

    let mut i = 0;
    let mut output: Vec<i64> = vec![];
    while i < program.len() {
        let opcode = program[i];
        let operand = program[i + 1];
        i += 2;

        if opcode == 0 {
            // ADV - A divided by 2^combo into A
            let exp = combo(operand, a, b, c);
            a = a / 2_i64.pow(exp as u32);
            debug(format!("ADV({}) -> combo={} a={}", operand, exp, a));
        } else if opcode == 1 {
            // BXL - bitwise xor of B with literal into B
            b = b ^ operand;
            debug(format!("BXL({}) -> b={}", operand, b));
        } else if opcode == 2 {
            // BST - combo mod 8 into B
            let combo = combo(operand, a, b, c);
            b = combo % 8;
            debug(format!("BST({}) -> combo={}, b={}", operand, combo, b));
        } else if opcode == 3 {
            // JNZ - A == 0 || jump to literal A
            if a > 0 {
                i = operand as usize;
            }
            debug(format!("JNZ({}) -> a={}, i={}", operand, a, i));
        } else if opcode == 4 {
            // BXC - bitwise or B ^ C into B (operand ignored)
            b = b ^ c;
            debug(format!("BXC({}) -> b={}", operand, b));
        } else if opcode == 5 {
            // OUT - output combo operand mod 8
            let combo = combo(operand, a, b, c);
            output.push(combo % 8);
            // debug(format!("OUT({}) -> combo={} combo%8={}", operand, combo, combo % 8));
            debug(format!(
                "OUT({}) -> combo={} combo%8={} output={}",
                operand,
                combo,
                combo % 8,
                output
                    .iter()
                    .map(|x| { x.to_string() })
                    .collect::<Vec<String>>()
                    .join(",")
            ));
        } else if opcode == 6 {
            // BDV - A divided by 2^combo into B
            let exp = combo(operand, a, b, c);
            b = a / 2_i64.pow(exp as u32);
            debug(format!("BDV({}) -> combo={} b={}", operand, exp, b));
        } else if opcode == 7 {
            // CDV - A divided by 2^combo into C
            let exp = combo(operand, a, b, c);
            c = a / 2_i64.pow(exp as u32);
            debug(format!("CDV({}) -> combo={} c={}", operand, exp, c));
        }
    }

    return output;
}

fn day17_part1(lines: &Vec<String>) -> String {
    // Parse input
    let a = str::parse::<i64>(lines[0].split(" ").last().expect("Bad register input?")).unwrap();
    let b = str::parse::<i64>(lines[1].split(" ").last().expect("Bad register input?")).unwrap();
    let c = str::parse::<i64>(lines[2].split(" ").last().expect("Bad register input?")).unwrap();

    let program: Vec<i64> = lines[4]
        .split(" ")
        .last()
        .expect("Bad program input?")
        .split(",")
        .map(str::parse::<i64>)
        .map(Result::unwrap)
        .collect();

    debug("Program input:");
    debug(a);
    debug(b);
    debug(c);
    debug(&program);

    let output: Vec<i64> = run_program(&a, &b, &c, &program);
    return output
        .iter()
        .map(|x| x.to_string())
        .collect::<Vec<String>>()
        .join(",");
}

fn day17_part2(lines: &Vec<String>) -> String {
    // We don't care about `a`, we're replacing it anyway:
    // let a = str::parse::<i64>(lines[0].split(" ").last().expect("Bad register input?")).unwrap();
    let b = str::parse::<i64>(lines[1].split(" ").last().expect("Bad register input?")).unwrap();
    let c = str::parse::<i64>(lines[2].split(" ").last().expect("Bad register input?")).unwrap();

    let program: Vec<i64> = lines[4]
        .split(" ")
        .last()
        .expect("Bad program input?")
        .split(",")
        .map(str::parse::<i64>)
        .map(Result::unwrap)
        .collect();

    // It looks like it's spitting out base-8 numbers, with some per-place-value mapping from what we put in.
    // So...
    // Let's see if we can iteratively build it back up?

    let mut potential_a = 0;
    let mut target: Vec<i64> = program.clone();
    while target.len() > 0 {
        let mut found = false;
        for i in 0..9 {
            let trial_a = (potential_a * 8) + i;
            let out = run_program(&trial_a, &b, &c, &program);
            if out[0] == target[target.len() - 1] {
                dbg!(i, trial_a, "==", out[0], target[target.len() - 1]);
                potential_a = trial_a;
                target.pop();
                found = true;
                break;
            } else {
                dbg!(i, trial_a, "!=", out, &target);
            }
        }
        if !found {
            dbg!("Nothing found for:", target[target.len() - 1]);
            break;
        } else {
            dbg!("Found and incremented to:", potential_a);
        }
    }
    return potential_a.to_string();

    //////////////// Looking for patterns...
    // let mut potential_a = 0;
    // // let mut potential_a = 184766955921675;
    // // smallest 16 digit base 8 number?
    // // let mut potential_a = 35184372088832;
    // loop {
    //     let output: Vec<i64> = run_program(&potential_a, &b, &c, &program);
    //     if output == program {
    //         return potential_a.to_string();
    //     }

    //     dbg!(format!("a={} --> out={}", potential_a, output.iter().map(|x| { x.to_string() }).collect::<Vec<String>>().join(",")));

    //     if potential_a % 10_000 == 0 {
    //         dbg!(format!("Tried {}...", potential_a));
    //     }
    //     if potential_a > 200 {
    //         // dbg!(program);
    //         // dbg!(output);
    //         std::process::exit(999);
    //     }

    //     potential_a += 1;
    // }
}

/* ---------------------------------------------------- */

fn main() {
    let args: Vec<String> = env::args().collect();

    let func_name = &args[1];
    let lines: Vec<String> = read_to_string(&args[2])
        .unwrap()
        .lines()
        .map(String::from)
        .collect();

    debug(&args);
    debug(&lines);

    let func = match func_name.as_str() {
        "day17_part1" => day17_part1,
        "day17_part2" => day17_part2,
        "day22_part1" => day22_part1,
        "day22_part2" => day22_part2,
        _ => unimplemented!("{}", func_name),
    };

    println!("---------- Running {}({}) ----------", &args[1], &args[2]);
    let now = Instant::now();
    println!("{}", func(&lines));
    let elapsed = now.elapsed();
    println!("---------- Execution complete in {:.2?} ----------", elapsed);

}

#[cfg(debug_assertions)]
fn debug<T: Debug>(obj: T) -> () {
    dbg!(obj);
}
#[cfg(not(debug_assertions))]
fn debug<T: Debug>(_: T) -> () {}

/* ---------------------------------------------------- */

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct Position {
    x: usize,
    y: usize,
}

impl std::ops::Add<Position> for Position {
    type Output = Position;
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl std::ops::Add<Vector> for Position {
    type Output = Position;
    fn add(self, other: Vector) -> Self {
        Self {
            x: self.x + other.dx,
            y: self.y + other.dy,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
struct Vector {
    dx: usize,
    dy: usize,
}

impl std::ops::Add<Position> for Vector {
    type Output = Position;
    fn add(self, other: Position) -> Position {
        Position {
            x: self.dx + other.x,
            y: self.dy + other.y,
        }
    }
}

impl std::ops::Add<Vector> for Vector {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            dx: self.dx + other.dx,
            dy: self.dy + other.dy,
        }
    }
}

#[derive(Debug)]
struct Grid {
    height: usize,
    width: usize,
    grid: Vec<Vec<String>>,
}

trait Graph {
    fn is_neighbor(&self, pos1: &Position, pos2: &Position) -> bool;
    fn neighbors(pos: &Position) -> Vec<Position>;
}

impl Graph for Grid {
    fn is_neighbor(
        &self,
        Position { x: x1, y: y1 }: &Position,
        Position { x: x2, y: y2 }: &Position,
    ) -> bool {
        return (0..self.height).contains(y1)
            && (0..self.height).contains(y2)
            && (0..self.width).contains(x1)
            && (0..self.width).contains(x2)
            && self.grid[*y1][*x1] != "#"
            && self.grid[*y2][*x2] != "#";
    }

    fn neighbors(Position { x, y }: &Position) -> Vec<Position> {
        vec![
            Position { x: *x + 1, y: *y },
            Position { x: *x - 1, y: *y },
            Position { x: *x, y: *y + 1 },
            Position { x: *x, y: *y - 1 },
        ]
    }
}
