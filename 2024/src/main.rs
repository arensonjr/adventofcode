use std::env;
use std::fmt::Debug;
use std::fs::read_to_string;

fn combo(operand: i64, a: i64, b: i64, c: i64) -> i64 {
    return match operand {
        0..=3 => operand,
        4 => a,
        5 => b,
        6 => c,
        _ => unimplemented!("Unknown operand"),
    }
}

fn run_program(init_a :&i64, init_b: &i64, init_c: &i64, program: &Vec<i64>) -> Vec<i64> {
    let mut a: i64 = *init_a;
    let mut b: i64 = *init_b;
    let mut c: i64 = *init_c;

    let mut i = 0;
    let mut output : Vec<i64> = vec![];
    while i < program.len() {
        let opcode = program[i];
        let operand = program[i+1];
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
            debug(format!("OUT({}) -> combo={} combo%8={} output={}", operand, combo, combo % 8, output.iter().map(|x| { x.to_string() }).collect::<Vec<String>>().join(",")));
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
    return output.iter().map(|x| { x.to_string() }).collect::<Vec<String>>().join(",");
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
        _ => unimplemented!("{}", func_name),
    };
    println!("{}", func(&lines));
}

#[cfg(debug_assertions)]
fn debug<T: Debug>(obj: T) -> () {
    dbg!(obj);
}
#[cfg(not(debug_assertions))]
fn debug<T: Debug>(_: T) -> () {}
