use std::env;
use std::cmp::Ordering;
use log::debug;

fn main() {
    let args: Vec<String> = env::args().collect();

    let day = &args[1];
    let part = &args[2];
    let infile = &args[3];
    let debug = &args[4];

    stderrlog::new().module(module_path!())
        .verbosity(if debug == "True" { log::Level::Debug } else { log::Level::Warn })
        .init().unwrap();

    debug!(args);
}

#[cfg(debug_assertions)]
fn debug(obj: ()) {
    dbg!(obj)
}

#[cfg(debug_assertions)]
fn debug(obj: ()) {
    dbg!(obj)
}