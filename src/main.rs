use noisy_float::prelude::*;
use rand::prelude::*;
use rand_distr::Exp;

struct Job {
    arrival_time: f64,
    original_size: f64,
    remaining_size: f64,
    id: u64,
}

enum Dist {
    Exp(f64),
    Discrete(Vec<(f64, f64)>),
}
impl Dist {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        todo!()
    }
    fn mean_size(&self) -> f64 {
        todo!()
    }
}

enum Policy {
    Global_NP_FCFS,
    Local_FCFS,
    Global_P_FCFS,
    Global_LPS(usize),
    PS,
}
impl Policy {
    // Set of indices to share amongst
    fn serve(&self, queue: &[Job]) -> Vec<usize> {
        todo!()
    }
}

struct Results {
    bucket_width: f64,
    bucket_counts: Vec<u64>,
}
impl Results {
    fn new(bucket_width: f64) -> Self {
        todo!()
    }
    fn add_response(&mut self, response: f64) {
        todo!()
    }
}

fn simulate(
    policy: &Policy,
    dist: &Dist,
    num_servers_per_stage: usize,
    num_stages: usize,
    rho: f64,
    num_jobs: u64,
    seed: u64,
) -> Results {
    todo!()
}
fn main() {
    let num_servers_per_stage = 3;
    let num_stages = 3;
    let policy = Policy::Global_LPS(6);
    let dist = Dist::Exp(1.0);
    let rho = 0.5;
    let num_jobs = 1_000_000;
    let seed = 0;
    let results = simulate(
        &policy,
        &dist,
        num_servers_per_stage,
        num_stages,
        rho,
        num_jobs,
        seed,
    );
    todo!()
}
