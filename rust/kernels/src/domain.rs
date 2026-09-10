//! Bounded-buffer CUDA implementations of the domain-specific backend.
#![allow(clippy::too_many_arguments, non_snake_case, unused_variables)]
use cuda_device::atomic::{AtomicOrdering, DeviceAtomicF64, DeviceAtomicI32, DeviceAtomicU64};
use cuda_device::{device, kernel, thread, warp};

/// Borrowed device buffer. The host validates allocation extent and alignment.
/// Indirect accesses are checked here before dereferencing sparse metadata.
#[derive(Clone, Copy)]
struct Buffer {
    pointer: u64,
    len: u64,
    kind: u64,
    order: u64,
    rows: u64,
    cols: u64,
}
impl Buffer {
    #[inline(always)]
    fn f(self, i: u64) -> f64 {
        if i >= self.len {
            return 0.0;
        }
        unsafe {
            match self.kind {
                0 => *(self.pointer as *const f32).add(i as usize) as f64,
                1 => *(self.pointer as *const f64).add(i as usize),
                _ => self.i(i) as f64,
            }
        }
    }
    // These typed accesses are used only inside float32 dispatch branches;
    // host validation ensures all corresponding values share that dtype.
    #[inline(always)]
    fn single(self, i: u64) -> f32 {
        if i >= self.len {
            return 0.0;
        }
        unsafe { *(self.pointer as *const f32).add(i as usize) }
    }
    #[inline(always)]
    fn put_single(self, i: u64, v: f32) {
        if i < self.len {
            unsafe {
                *(self.pointer as *mut f32).add(i as usize) = v;
            }
        }
    }
    #[inline(always)]
    fn i(self, i: u64) -> u64 {
        if i >= self.len {
            return 0;
        }
        unsafe {
            match self.kind {
                2 => *(self.pointer as *const i32).add(i as usize) as u64,
                3 => *(self.pointer as *const i64).add(i as usize) as u64,
                4 => *(self.pointer as *const u8).add(i as usize) as u64,
                5 => *(self.pointer as *const u32).add(i as usize) as u64,
                6 => *(self.pointer as *const u64).add(i as usize),
                _ => 0,
            }
        }
    }
    #[inline(always)]
    fn put(self, i: u64, v: f64) {
        if i >= self.len {
            return;
        }
        unsafe {
            match self.kind {
                0 => *(self.pointer as *mut f32).add(i as usize) = v as f32,
                1 => *(self.pointer as *mut f64).add(i as usize) = v,
                _ => self.put_i(i, v as u64),
            }
        }
    }
    #[inline(always)]
    fn put_i(self, i: u64, v: u64) {
        if i >= self.len {
            return;
        }
        unsafe {
            match self.kind {
                2 => *(self.pointer as *mut i32).add(i as usize) = v as i32,
                3 => *(self.pointer as *mut i64).add(i as usize) = v as i64,
                4 => *(self.pointer as *mut u8).add(i as usize) = u8::from(v != 0),
                5 => *(self.pointer as *mut u32).add(i as usize) = v as u32,
                6 => *(self.pointer as *mut u64).add(i as usize) = v,
                _ => {}
            }
        }
    }
    #[inline(always)]
    fn add(self, i: u64, v: f64) {
        if i >= self.len {
            return;
        }
        unsafe {
            match self.kind {
                0 => {
                    crate::atomics::add_f32((self.pointer as *mut f32).add(i as usize), v as f32);
                }
                1 => {
                    DeviceAtomicF64::from_ptr((self.pointer as *mut f64).add(i as usize))
                        .fetch_add(v, AtomicOrdering::Relaxed);
                }
                2 => {
                    DeviceAtomicI32::from_ptr((self.pointer as *mut i32).add(i as usize))
                        .fetch_add(v as i32, AtomicOrdering::Relaxed);
                }
                6 => {
                    DeviceAtomicU64::from_ptr((self.pointer as *mut u64).add(i as usize))
                        .fetch_add(v as u64, AtomicOrdering::Relaxed);
                }
                _ => {}
            }
        }
    }
    #[inline(always)]
    fn single_at(self, row: u64, col: u64) -> f32 {
        if row >= self.rows || col >= self.cols {
            return 0.0;
        }
        self.single(if self.order == 0 {
            row * self.cols + col
        } else {
            col * self.rows + row
        })
    }
    #[inline(always)]
    fn round(self, v: f64) -> f64 {
        if self.kind == 0 { v as f32 as f64 } else { v }
    }
}
#[device]
fn tid() -> u64 {
    thread::index_1d().get() as u64
}
#[device]
fn stride() -> u64 {
    thread::blockDim_x() as u64 * thread::gridDim_x() as u64
}
#[device]
fn sum_warp(mut v: f64) -> f64 {
    let mut d = 16;
    while d > 0 {
        v += warp::shuffle_down_f64_sync(u32::MAX, v, d);
        d /= 2;
    }
    v
}
#[device]
fn domain_warp_total(value: f64) -> f64 {
    warp::shuffle_f64_sync(u32::MAX, sum_warp(value), 0)
}
// Block reduction for the 128- and 256-thread domain kernels. Every thread
// participates in each call.
#[device]
fn domain_block_total(value: f64) -> f64 {
    static mut PARTIAL: cuda_device::SharedArray<f64, 8> = cuda_device::SharedArray::UNINIT;
    let block = thread::blockDim_x() as u64;
    let lane = thread::threadIdx_x() as u64;
    let value = sum_warp(value);
    if lane.is_multiple_of(32) {
        unsafe {
            PARTIAL[(lane / 32) as usize] = value;
        }
    }
    thread::sync_threads();
    if lane == 0 {
        unsafe {
            let mut warp = 1;
            while warp < block / 32 {
                PARTIAL[0] += PARTIAL[warp as usize];
                warp += 1;
            }
        }
    }
    thread::sync_threads();
    let total = unsafe { PARTIAL[0] };
    thread::sync_threads();
    total
}

// Domain module registry.
mod aggr;
mod aucell;
mod autocorr;
mod bbknn;
mod cooc;
mod edistance;
mod guide_assignment;
mod kde;
mod ligrec;
mod mixscale;
mod nn_descent;
mod pseudobulk;
mod pv;
mod sinkhorn;
mod spca;
