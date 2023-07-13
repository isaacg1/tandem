use noisy_float::prelude::*;
use rand::prelude::*;
use rand_distr::Exp;

use std::collections::VecDeque;

const EPSILON: f64 = 1e-8;
const INFINITY: f64 = f64::INFINITY;

struct Job {
    arrival_time: f64,
    original_size: f64,
    remaining_size: f64,
    id: u64,
}

#[derive(Debug)]
enum Dist {
    Exp,
    Discrete(Vec<(f64, f64)>),
}
impl Dist {
    fn sample<R: Rng>(&self, rng: &mut R) -> f64 {
        match self {
            Dist::Exp => {
                let exp = Exp::new(1.0).expect("valid");
                exp.sample(rng)
            }
            Dist::Discrete(cases) => {
                let mut num: f64 = rng.gen();
                for (prob, value) in cases {
                    num -= prob;
                    if num <= 0.0 {
                        return *value;
                    }
                }
                unreachable!("{:?}", cases)
            }
        }
    }
    fn mean_size(&self) -> f64 {
        match self {
            Dist::Exp => 1.0,
            Dist::Discrete(cases) => cases.iter().map(|(prob, value)| prob * value).sum(),
        }
    }
}

#[derive(Debug)]
enum Policy {
    Global_NP_FCFS,
    Local_FCFS,
    Global_P_FCFS,
    Global_LPS(usize),
    PS,
}
impl Policy {
    // Set of indices to share amongst
    fn serve(&self, queue: &[Job], num_servers: usize) -> Vec<usize> {
        match self {
            Policy::Global_NP_FCFS => {
                let mut indices: Vec<usize> = (0..queue.len()).collect();
                if indices.len() <= num_servers {
                    return indices;
                }
                indices.sort_by_key(|i| {
                    let job = &queue[*i];
                    let key = if job.original_size > job.remaining_size {
                        -1.0
                    } else {
                        job.arrival_time
                    };
                    n64(key)
                });
                indices[..num_servers].to_vec()
            }
            Policy::Local_FCFS => (0..queue.len()).take(num_servers).collect(),
            Policy::Global_P_FCFS => {
                let mut indices: Vec<usize> = (0..queue.len()).collect();
                if indices.len() <= num_servers {
                    return indices;
                }
                indices.sort_by_key(|i| {
                    let job = &queue[*i];
                    let key = job.arrival_time;
                    n64(key)
                });
                indices[..num_servers].to_vec()
            }
            Policy::Global_LPS(jobs_in_service) => {
                let mut indices: Vec<usize> = (0..queue.len()).collect();
                if indices.len() <= *jobs_in_service {
                    return indices;
                }
                indices.sort_by_key(|i| {
                    let job = &queue[*i];
                    let key = job.arrival_time;
                    n64(key)
                });
                indices[..*jobs_in_service].to_vec()
            }
            Policy::PS => (0..queue.len()).collect(),
        }
    }
}

struct Results {
    bucket_width: f64,
    bucket_counts: Vec<u64>,
}
impl Results {
    fn new(bucket_width: f64) -> Self {
        Results {
            bucket_width,
            bucket_counts: vec![],
        }
    }
    fn add_response(&mut self, response: f64) {
        let target_bucket = (response / self.bucket_width).floor() as usize;
        if self.bucket_counts.len() <= target_bucket {
            self.bucket_counts
                .extend((self.bucket_counts.len()..=target_bucket).map(|_| 0));
        }
        self.bucket_counts[target_bucket] += 1;
    }
    fn percentile(&self, perc: f64) -> f64 {
        assert!(perc >= 0.0);
        assert!(perc <= 1.0);
        let total: u64 = self.bucket_counts.iter().sum();
        let target = (total as f64 * perc) as u64;
        let mut running_count = 0;
        for (i, count) in self.bucket_counts.iter().enumerate() {
            running_count += count;
            if running_count >= target {
                return i as f64 * self.bucket_width;
            }
        }
        unreachable!("Perc: {}", perc);
    }
}

fn simulate(
    policy: &Policy,
    dist: &Dist,
    num_servers_per_stage: usize,
    num_stages: usize,
    rho: f64,
    num_jobs: u64,
    bucket_width: f64,
    thread_pool_limit: usize,
    seed: u64,
) -> Results {
    assert!((dist.mean_size() - 1.0).abs() < EPSILON);
    let arrival_rate = rho;
    let arrival_dist = Exp::new(arrival_rate).expect("Valid rate");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut num_completions = 0;
    let mut num_arrivals = 0;
    let mut time = 0.0;
    // Stored in LocalFCFS order
    let mut admission_pool: VecDeque<Job> = VecDeque::new();
    let mut queues: Vec<Vec<Job>> = (0..num_stages).map(|_| vec![]).collect();
    let mut results = Results::new(bucket_width);
    let mut to_append = vec![];
    while num_completions < num_jobs {
        let mut next_comp_diff = INFINITY;
        let mut services = vec![];
        for queue in &queues {
            let mut service = policy.serve(&queue, num_servers_per_stage);
            service.sort();
            assert!(service.len() >= num_servers_per_stage.min(queue.len()));
            for &s in &service {
                // Each job runs at speed inversely proportional
                // to the number of jobs in service if >= num_servers in service,
                // otherwise at speed 1/num_servers
                let time_to_completion =
                    queue[s].remaining_size * service.len().max(num_servers_per_stage) as f64;
                next_comp_diff = next_comp_diff.min(time_to_completion);
            }
            services.push(service);
        }
        // Valid to resample because Poisson
        let arrival_diff = arrival_dist.sample(&mut rng);
        let is_arrival = arrival_diff < next_comp_diff;
        let event_diff = arrival_diff.min(next_comp_diff);
        time += event_diff;
        let mut num_completion_occured = 0;
        for stage in 0..num_stages {
            let queue = &mut queues[stage];
            let service = &services[stage];
            // Reverse sorted order, safe to delete
            for &s in service.iter().rev() {
                // Each job runs at speed inversely proportional
                // to the number of jobs in service if >= num_servers in service,
                // otherwise at speed 1/num_servers
                queue[s].remaining_size -=
                    event_diff / service.len().max(num_servers_per_stage) as f64;
                if queue[s].remaining_size < EPSILON {
                    let job = queue.remove(s);
                    if stage == num_stages - 1 {
                        num_completions += 1;
                        results.add_response(time - job.arrival_time);
                        num_completion_occured += 1;
                    } else {
                        let new_size = dist.sample(&mut rng);
                        let new_job = Job {
                            arrival_time: job.arrival_time,
                            remaining_size: new_size,
                            original_size: new_size,
                            id: job.id,
                        };
                        to_append.push(new_job);
                    }
                }
            }
            if stage < num_stages - 1 {
                queues[stage + 1].append(&mut to_append);
            }
            assert!(to_append.is_empty());
        }
        for _i in 0..num_completion_occured {
            if let Some(new_job) = admission_pool.pop_front() {
                queues[0].push(new_job);
            }
        }
        if is_arrival {
            let new_size = dist.sample(&mut rng);
            let new_job = Job {
                arrival_time: time,
                remaining_size: new_size,
                original_size: new_size,
                id: num_arrivals,
            };
            let num_jobs_in_flight: usize = queues.iter().map(|q| q.len()).sum();
            if num_jobs_in_flight < thread_pool_limit {
                queues[0].push(new_job);
            } else {
                admission_pool.push_back(new_job);
            }
            num_arrivals += 1;
        }
    }
    results
}
fn main() {
    let num_servers_per_stage = 3;
    let num_stages = 3;
    let policies = vec![
        Policy::Global_NP_FCFS,
        Policy::Global_P_FCFS,
        Policy::Global_LPS(6),
        Policy::Local_FCFS,
        Policy::PS,
    ];
    let dist_num = 0;
    let dist = match dist_num {
        0 => Dist::Exp,
        1 => {
            let p = 1.0 / 40.0;
            Dist::Discrete(vec![
                (p, 1.0 / (2.0 * p)),
                (1.0 - p, 1.0 / (2.0 * (1.0 - p))),
            ])
        }
        _ => unimplemented!()
    };
    let rhos = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95,
    ];
    //let rhos = vec![0.96, 0.97, 0.98, 0.99];
    let num_jobs = 1_000_000;
    let seed = 0;
    let bucket_width = 0.1;
    let perc = 0.99;
    let thread_pool_limit = 100;
    println!(
        "servers {}; stages {}; dist {:?}; jobs {}; bucket {}; seed {}; thread_pool_limit {}",
        num_servers_per_stage, num_stages, dist, num_jobs, bucket_width, seed, thread_pool_limit
    );
    println!("{}th percentile of response time", perc);
    print!("rho;");
    for policy in &policies {
        print!("{:?};", policy);
    }
    println!();
    for rho in rhos {
        print!("{};", rho);
        for policy in &policies {
            let results = simulate(
                policy,
                &dist,
                num_servers_per_stage,
                num_stages,
                rho,
                num_jobs,
                bucket_width,
                thread_pool_limit,
                seed,
            );
            print!("{};", results.percentile(perc));
        }
        println!();
    }
}
