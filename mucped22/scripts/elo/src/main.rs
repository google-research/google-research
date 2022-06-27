use std::env;
use std::fs;

use rand::seq::SliceRandom;
use rand::thread_rng;
use serde_derive::Deserialize;
use serde_json;
use std::collections::HashMap;

const INITIAL_ELO: f32 = 2000.0;
const K: f32 = 30.0;
const N: f32 = 400.0;
const NUM_SIMULATIONS: u32 = 10000;
const MIN_TIME: u64 = 1000;
const MIN_FLIPS: u32 = 3;

#[derive(Deserialize, Debug, Clone)]
struct Comparison {
    greater: String,
    lesser: String,
    random_choice: bool,
    rater_flips: u32,
    rater_time_ms: u64,
    image: String,
}

fn main() {
    let args: Vec<String> = env::args().collect();

    let contents = fs::read_to_string(&args[1]).expect("Something went wrong reading the file");

    let img = args.get(2).filter(|&s| !s.is_empty());

    let data: Vec<Comparison> = serde_json::from_str(&contents).unwrap();

    let mut filtered_data: Vec<Comparison> = data
        .iter()
        .filter(|x| x.rater_flips >= MIN_FLIPS && x.rater_time_ms >= MIN_TIME)
        .filter(|x| img.is_none() || img == Some(&x.image))
        .cloned()
        .collect();

    let mut rng = thread_rng();

    let mut elos = HashMap::<String, f64>::new();

    for _ in 0..NUM_SIMULATIONS {
        filtered_data.shuffle(&mut rng);
        let mut current_elos = HashMap::<String, f32>::new();
        for cmp in filtered_data.iter().cycle().take(data.len()) {
            let (ma, mb) = (&cmp.greater, &cmp.lesser);
            let (sa, sb) = if cmp.random_choice {
                (0.5, 0.5)
            } else {
                (1.0, 0.0)
            };
            let eloa = current_elos.get(ma).unwrap_or(&INITIAL_ELO);
            let elob = current_elos.get(mb).unwrap_or(&INITIAL_ELO);
            let qa = 10f32.powf(eloa / N);
            let qb = 10f32.powf(elob / N);
            let ea = qa / (qa + qb);
            let eb = qb / (qa + qb);
            *current_elos.entry(ma.to_owned()).or_insert(INITIAL_ELO) += K * (sa - ea);
            *current_elos.entry(mb.to_owned()).or_insert(INITIAL_ELO) += K * (sb - eb);
        }
        for (k, v) in current_elos {
            *elos.entry(k).or_insert(0.0) += v as f64;
        }
    }

    let mut ratings: Vec<_> = elos
        .iter()
        .map(|(k, v)| (k, v / NUM_SIMULATIONS as f64))
        .collect();
    ratings.sort_by(|(_, a), (_, b)| b.partial_cmp(&a).unwrap());
    for (m, r) in ratings {
        println!("{} {}", m, r);
    }
}
