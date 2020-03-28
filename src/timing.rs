use std::time::Instant;

pub fn time_milliseconds<N, F: FnMut() -> N>(mut f: F) -> (N, u128) {
    let start = Instant::now();
    let data = f();
    let millis = Instant::now().duration_since(start).as_millis();
    (data, millis)
}

pub fn print_time_milliseconds<N, F: FnMut() -> N>(label: &str, f: F) -> N {
    println!("Started {}...", label);
    let result = time_milliseconds(f);
    let seconds = (result.1 / 1000) as f64;
    let milliseconds = ((result.1 % 1000) as f64) / 1000.0;
    println!("Finished {} after {} seconds", label, seconds + milliseconds);
    result.0
}