//! Bounded CPU work over Rust-owned buffers, detached from the interpreter.
//!
//! GPU shards share one pool rather than creating a pool per calling thread.
//! A call submits at most its captured worker limit in independent partitions;
//! even concurrent calls cannot exceed the pool's 32-thread hardware cap.
use pyo3::{exceptions::PyRuntimeError, prelude::*};
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use std::{cell::Cell, sync::OnceLock};

pub(crate) const MAX_WORKERS: usize = 32;
const MIN_PARALLEL_ITEMS: usize = 4096;
thread_local! { static WORKER_LIMIT: Cell<i32> = const { Cell::new(0) }; }
static POOL: OnceLock<Result<ThreadPool, String>> = OnceLock::new();

pub fn set_worker_limit(limit: i32) -> i32 {
    WORKER_LIMIT.with(|value| value.replace(limit.max(0)))
}

fn hardware_workers() -> usize {
    std::thread::available_parallelism()
        .map_or(4, usize::from)
        .min(MAX_WORKERS)
}

/// Captured on the calling Python thread before entering a Rayon worker.
#[derive(Clone, Copy)]
pub struct Parallelism {
    workers: usize,
    use_pool: bool,
}

impl Parallelism {
    fn capture(items: usize) -> Self {
        let limit = WORKER_LIMIT.with(Cell::get);
        let workers = if items < MIN_PARALLEL_ITEMS {
            1
        } else if limit > 0 {
            hardware_workers().min(limit as usize)
        } else {
            hardware_workers()
        };
        Self {
            workers,
            use_pool: items >= MIN_PARALLEL_ITEMS,
        }
    }

    pub fn workers(self) -> usize {
        self.workers
    }

    /// The function receives each disjoint partition and its element offset.
    /// There are at most `workers` tasks, including when another shard is busy.
    pub fn for_each_chunk<T: Send>(
        self,
        output: &mut [T],
        operation: impl Fn(usize, &mut [T]) + Sync + Send,
    ) {
        if output.is_empty() {
            return;
        }
        if self.workers == 1 {
            operation(0, output);
        } else {
            let chunk = output.len().div_ceil(self.workers);
            output
                .par_chunks_mut(chunk)
                .enumerate()
                .for_each(|(i, part)| operation(i * chunk, part));
        }
    }
}

fn execute<R: Send>(
    plan: Parallelism,
    operation: impl FnOnce(Parallelism) -> R + Send,
) -> Result<R, String> {
    if !plan.use_pool {
        return Ok(operation(plan));
    }
    let pool = POOL.get_or_init(|| {
        ThreadPoolBuilder::new()
            .num_threads(hardware_workers())
            .thread_name(|index| format!("rsc-host-{index}"))
            .build()
            .map_err(|error| format!("unable to create host worker pool: {error}"))
    });
    match pool {
        Ok(pool) => Ok(pool.install(|| operation(plan))),
        Err(error) => Err(error.clone()),
    }
}

/// Only owned Rust inputs and disjoint owned outputs may enter `operation`.
/// Python buffers must first be copied while attached to the interpreter.
pub fn run<R: Send>(
    py: Python<'_>,
    items: usize,
    operation: impl FnOnce(Parallelism) -> R + Send,
) -> PyResult<R> {
    let plan = Parallelism::capture(items);
    py.detach(|| execute(plan, operation))
        .map_err(PyRuntimeError::new_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    #[test]
    fn small_work_stays_on_the_caller() {
        let caller = std::thread::current().id();
        let plan = Parallelism::capture(MIN_PARALLEL_ITEMS - 1);
        assert_eq!(plan.workers(), 1);
        execute(plan, |_| assert_eq!(std::thread::current().id(), caller)).unwrap();
    }

    #[test]
    fn limits_are_captured_per_caller_and_the_pool_is_shared() {
        let barrier = Arc::new(Barrier::new(4));
        let callers: Vec<_> = (1..=4)
            .map(|limit| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    assert_eq!(set_worker_limit(limit), 0);
                    let plan = Parallelism::capture(100_000);
                    assert_eq!(plan.workers(), hardware_workers().min(limit as usize));
                    barrier.wait();
                    execute(plan, |plan| {
                        let tasks = AtomicUsize::new(0);
                        let mut output = vec![0; 100_000];
                        plan.for_each_chunk(&mut output, |offset, part| {
                            tasks.fetch_add(1, Ordering::Relaxed);
                            for (i, value) in part.iter_mut().enumerate() {
                                *value = offset + i;
                            }
                        });
                        assert!(tasks.load(Ordering::Relaxed) <= plan.workers());
                        assert!(output.iter().copied().eq(0..100_000));
                    })
                    .unwrap();
                    assert_eq!(set_worker_limit(-1), limit);
                })
            })
            .collect();
        for caller in callers {
            caller.join().unwrap();
        }
        if let Some(Ok(pool)) = POOL.get() {
            assert!(pool.current_num_threads() <= MAX_WORKERS);
        }
    }
}
