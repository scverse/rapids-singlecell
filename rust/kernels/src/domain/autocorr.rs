//! Native autocorr CUDA kernels.
use super::*;

/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_autocorr_morans_dense(
    data_centered: u64,
    adj_row_ptr: u64,
    adj_col_ind: u64,
    adj_data: u64,
    num: u64,
    n_samples: u64,
    n_features: u64,
    data_centered_len: u64,
    data_centered_kind: u64,
    data_centered_order: u64,
    data_centered_rows: u64,
    data_centered_cols: u64,
    adj_row_ptr_len: u64,
    adj_row_ptr_kind: u64,
    adj_row_ptr_order: u64,
    adj_row_ptr_rows: u64,
    adj_row_ptr_cols: u64,
    adj_col_ind_len: u64,
    adj_col_ind_kind: u64,
    adj_col_ind_order: u64,
    adj_col_ind_rows: u64,
    adj_col_ind_cols: u64,
    adj_data_len: u64,
    adj_data_kind: u64,
    adj_data_order: u64,
    adj_data_rows: u64,
    adj_data_cols: u64,
    num_len: u64,
    num_kind: u64,
    num_order: u64,
    num_rows: u64,
    num_cols: u64,
) {
    let data_centered = Buffer {
        pointer: data_centered,
        len: data_centered_len,
        kind: data_centered_kind,
        order: data_centered_order,
        rows: data_centered_rows,
        cols: data_centered_cols,
    };
    let adj_row_ptr = Buffer {
        pointer: adj_row_ptr,
        len: adj_row_ptr_len,
        kind: adj_row_ptr_kind,
        order: adj_row_ptr_order,
        rows: adj_row_ptr_rows,
        cols: adj_row_ptr_cols,
    };
    let adj_col_ind = Buffer {
        pointer: adj_col_ind,
        len: adj_col_ind_len,
        kind: adj_col_ind_kind,
        order: adj_col_ind_order,
        rows: adj_col_ind_rows,
        cols: adj_col_ind_cols,
    };
    let adj_data = Buffer {
        pointer: adj_data,
        len: adj_data_len,
        kind: adj_data_kind,
        order: adj_data_order,
        rows: adj_data_rows,
        cols: adj_data_cols,
    };
    let num = Buffer {
        pointer: num,
        len: num_len,
        kind: num_kind,
        order: num_order,
        rows: num_rows,
        cols: num_cols,
    };
    let mut p = tid();
    while p < n_samples * n_features {
        let row = p / n_features;
        let feature = p % n_features;
        let x = data_centered.f(row * n_features + feature);
        let mut q = adj_row_ptr.i(row);
        let end = adj_row_ptr
            .i(row + 1)
            .min(adj_col_ind.len)
            .min(adj_data.len);
        while q < end {
            let other = adj_col_ind.i(q);
            if other < n_samples {
                let y = data_centered.f(other * n_features + feature);
                let value = if data_centered.kind == 0 {
                    (adj_data.f(q) as f32 * ((x as f32) * (y as f32))) as f64
                } else {
                    adj_data.f(q) * x * y
                };
                num.add(feature, value);
            }
            q += 1;
        }
        p += stride();
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_autocorr_morans_sparse(
    adj_row_ptr: u64,
    adj_col_ind: u64,
    adj_data: u64,
    data_row_ptr: u64,
    data_col_ind: u64,
    data_values: u64,
    n_samples: u64,
    n_features: u64,
    mean_array: u64,
    num: u64,
    adj_row_ptr_len: u64,
    adj_row_ptr_kind: u64,
    adj_row_ptr_order: u64,
    adj_row_ptr_rows: u64,
    adj_row_ptr_cols: u64,
    adj_col_ind_len: u64,
    adj_col_ind_kind: u64,
    adj_col_ind_order: u64,
    adj_col_ind_rows: u64,
    adj_col_ind_cols: u64,
    adj_data_len: u64,
    adj_data_kind: u64,
    adj_data_order: u64,
    adj_data_rows: u64,
    adj_data_cols: u64,
    data_row_ptr_len: u64,
    data_row_ptr_kind: u64,
    data_row_ptr_order: u64,
    data_row_ptr_rows: u64,
    data_row_ptr_cols: u64,
    data_col_ind_len: u64,
    data_col_ind_kind: u64,
    data_col_ind_order: u64,
    data_col_ind_rows: u64,
    data_col_ind_cols: u64,
    data_values_len: u64,
    data_values_kind: u64,
    data_values_order: u64,
    data_values_rows: u64,
    data_values_cols: u64,
    mean_array_len: u64,
    mean_array_kind: u64,
    mean_array_order: u64,
    mean_array_rows: u64,
    mean_array_cols: u64,
    num_len: u64,
    num_kind: u64,
    num_order: u64,
    num_rows: u64,
    num_cols: u64,
) {
    let adj_row_ptr = Buffer {
        pointer: adj_row_ptr,
        len: adj_row_ptr_len,
        kind: adj_row_ptr_kind,
        order: adj_row_ptr_order,
        rows: adj_row_ptr_rows,
        cols: adj_row_ptr_cols,
    };
    let adj_col_ind = Buffer {
        pointer: adj_col_ind,
        len: adj_col_ind_len,
        kind: adj_col_ind_kind,
        order: adj_col_ind_order,
        rows: adj_col_ind_rows,
        cols: adj_col_ind_cols,
    };
    let adj_data = Buffer {
        pointer: adj_data,
        len: adj_data_len,
        kind: adj_data_kind,
        order: adj_data_order,
        rows: adj_data_rows,
        cols: adj_data_cols,
    };
    let data_row_ptr = Buffer {
        pointer: data_row_ptr,
        len: data_row_ptr_len,
        kind: data_row_ptr_kind,
        order: data_row_ptr_order,
        rows: data_row_ptr_rows,
        cols: data_row_ptr_cols,
    };
    let data_col_ind = Buffer {
        pointer: data_col_ind,
        len: data_col_ind_len,
        kind: data_col_ind_kind,
        order: data_col_ind_order,
        rows: data_col_ind_rows,
        cols: data_col_ind_cols,
    };
    let data_values = Buffer {
        pointer: data_values,
        len: data_values_len,
        kind: data_values_kind,
        order: data_values_order,
        rows: data_values_rows,
        cols: data_values_cols,
    };
    let mean_array = Buffer {
        pointer: mean_array,
        len: mean_array_len,
        kind: mean_array_kind,
        order: mean_array_order,
        rows: mean_array_rows,
        cols: mean_array_cols,
    };
    let num = Buffer {
        pointer: num,
        len: num_len,
        kind: num_kind,
        order: num_order,
        rows: num_rows,
        cols: num_cols,
    };
    static mut LEFT: cuda_device::SharedArray<f64, 1024> = cuda_device::SharedArray::UNINIT;
    static mut RIGHT: cuda_device::SharedArray<f64, 1024> = cuda_device::SharedArray::UNINIT;
    let mut row = tid() / 128;
    let lane = tid() % 128;
    while row < n_samples {
        let mut base = 0;
        while base < n_features {
            let mut j = lane;
            while j < 1024 {
                unsafe {
                    LEFT[j as usize] = 0.0;
                }
                j += 128;
            }
            thread::sync_threads();
            let mut p = data_row_ptr.i(row) + lane;
            let end = data_row_ptr
                .i(row + 1)
                .min(data_col_ind.len)
                .min(data_values.len);
            while p < end {
                let g = data_col_ind.i(p);
                if g >= base && g < base + 1024 {
                    unsafe {
                        LEFT[(g - base) as usize] = data_values.f(p);
                    }
                }
                p += 128;
            }
            thread::sync_threads();
            let mut q = adj_row_ptr.i(row);
            let stop = adj_row_ptr
                .i(row + 1)
                .min(adj_col_ind.len)
                .min(adj_data.len);
            while q < stop {
                let other = adj_col_ind.i(q);
                if other < n_samples {
                    j = lane;
                    while j < 1024 {
                        unsafe {
                            RIGHT[j as usize] = 0.0;
                        }
                        j += 128;
                    }
                    thread::sync_threads();
                    p = data_row_ptr.i(other) + lane;
                    let end = data_row_ptr
                        .i(other + 1)
                        .min(data_col_ind.len)
                        .min(data_values.len);
                    while p < end {
                        let g = data_col_ind.i(p);
                        if g >= base && g < base + 1024 {
                            unsafe {
                                RIGHT[(g - base) as usize] = data_values.f(p);
                            }
                        }
                        p += 128;
                    }
                    thread::sync_threads();
                    j = lane;
                    while j < 1024 && base + j < n_features {
                        let feature = base + j;
                        let x = unsafe { LEFT[j as usize] };
                        let y = unsafe { RIGHT[j as usize] };
                        let mean = mean_array.f(feature);
                        let a = data_values.round(x - mean);
                        let b = data_values.round(y - mean);
                        let value = if data_values.kind == 0 {
                            (adj_data.f(q) as f32
                                * ((x as f32 - mean_array.f(feature) as f32)
                                    * (y as f32 - mean_array.f(feature) as f32)))
                                as f64
                        } else {
                            adj_data.f(q) * a * b
                        };
                        num.add(feature, value);
                        j += 128;
                    }
                    thread::sync_threads();
                }
                q += 1;
            }
            base += 1024;
        }
        row += stride() / 128;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_autocorr_gearys_dense(
    data: u64,
    adj_row_ptr: u64,
    adj_col_ind: u64,
    adj_data: u64,
    num: u64,
    n_samples: u64,
    n_features: u64,
    data_len: u64,
    data_kind: u64,
    data_order: u64,
    data_rows: u64,
    data_cols: u64,
    adj_row_ptr_len: u64,
    adj_row_ptr_kind: u64,
    adj_row_ptr_order: u64,
    adj_row_ptr_rows: u64,
    adj_row_ptr_cols: u64,
    adj_col_ind_len: u64,
    adj_col_ind_kind: u64,
    adj_col_ind_order: u64,
    adj_col_ind_rows: u64,
    adj_col_ind_cols: u64,
    adj_data_len: u64,
    adj_data_kind: u64,
    adj_data_order: u64,
    adj_data_rows: u64,
    adj_data_cols: u64,
    num_len: u64,
    num_kind: u64,
    num_order: u64,
    num_rows: u64,
    num_cols: u64,
) {
    let data = Buffer {
        pointer: data,
        len: data_len,
        kind: data_kind,
        order: data_order,
        rows: data_rows,
        cols: data_cols,
    };
    let adj_row_ptr = Buffer {
        pointer: adj_row_ptr,
        len: adj_row_ptr_len,
        kind: adj_row_ptr_kind,
        order: adj_row_ptr_order,
        rows: adj_row_ptr_rows,
        cols: adj_row_ptr_cols,
    };
    let adj_col_ind = Buffer {
        pointer: adj_col_ind,
        len: adj_col_ind_len,
        kind: adj_col_ind_kind,
        order: adj_col_ind_order,
        rows: adj_col_ind_rows,
        cols: adj_col_ind_cols,
    };
    let adj_data = Buffer {
        pointer: adj_data,
        len: adj_data_len,
        kind: adj_data_kind,
        order: adj_data_order,
        rows: adj_data_rows,
        cols: adj_data_cols,
    };
    let num = Buffer {
        pointer: num,
        len: num_len,
        kind: num_kind,
        order: num_order,
        rows: num_rows,
        cols: num_cols,
    };
    let mut p = tid();
    while p < n_samples * n_features {
        let row = p / n_features;
        let feature = p % n_features;
        let x = data.f(row * n_features + feature);
        let mut q = adj_row_ptr.i(row);
        let end = adj_row_ptr
            .i(row + 1)
            .min(adj_col_ind.len)
            .min(adj_data.len);
        while q < end {
            let other = adj_col_ind.i(q);
            if other < n_samples {
                let y = data.f(other * n_features + feature);
                let diff = data.round(x - y);
                let value = if data.kind == 0 {
                    (adj_data.f(q) as f32 * (x as f32 - y as f32) * (x as f32 - y as f32)) as f64
                } else {
                    adj_data.f(q) * diff * diff
                };
                num.add(feature, value);
            }
            q += 1;
        }
        p += stride();
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_autocorr_gearys_sparse(
    adj_row_ptr: u64,
    adj_col_ind: u64,
    adj_data: u64,
    data_row_ptr: u64,
    data_col_ind: u64,
    data_values: u64,
    n_samples: u64,
    n_features: u64,
    num: u64,
    adj_row_ptr_len: u64,
    adj_row_ptr_kind: u64,
    adj_row_ptr_order: u64,
    adj_row_ptr_rows: u64,
    adj_row_ptr_cols: u64,
    adj_col_ind_len: u64,
    adj_col_ind_kind: u64,
    adj_col_ind_order: u64,
    adj_col_ind_rows: u64,
    adj_col_ind_cols: u64,
    adj_data_len: u64,
    adj_data_kind: u64,
    adj_data_order: u64,
    adj_data_rows: u64,
    adj_data_cols: u64,
    data_row_ptr_len: u64,
    data_row_ptr_kind: u64,
    data_row_ptr_order: u64,
    data_row_ptr_rows: u64,
    data_row_ptr_cols: u64,
    data_col_ind_len: u64,
    data_col_ind_kind: u64,
    data_col_ind_order: u64,
    data_col_ind_rows: u64,
    data_col_ind_cols: u64,
    data_values_len: u64,
    data_values_kind: u64,
    data_values_order: u64,
    data_values_rows: u64,
    data_values_cols: u64,
    num_len: u64,
    num_kind: u64,
    num_order: u64,
    num_rows: u64,
    num_cols: u64,
) {
    let adj_row_ptr = Buffer {
        pointer: adj_row_ptr,
        len: adj_row_ptr_len,
        kind: adj_row_ptr_kind,
        order: adj_row_ptr_order,
        rows: adj_row_ptr_rows,
        cols: adj_row_ptr_cols,
    };
    let adj_col_ind = Buffer {
        pointer: adj_col_ind,
        len: adj_col_ind_len,
        kind: adj_col_ind_kind,
        order: adj_col_ind_order,
        rows: adj_col_ind_rows,
        cols: adj_col_ind_cols,
    };
    let adj_data = Buffer {
        pointer: adj_data,
        len: adj_data_len,
        kind: adj_data_kind,
        order: adj_data_order,
        rows: adj_data_rows,
        cols: adj_data_cols,
    };
    let data_row_ptr = Buffer {
        pointer: data_row_ptr,
        len: data_row_ptr_len,
        kind: data_row_ptr_kind,
        order: data_row_ptr_order,
        rows: data_row_ptr_rows,
        cols: data_row_ptr_cols,
    };
    let data_col_ind = Buffer {
        pointer: data_col_ind,
        len: data_col_ind_len,
        kind: data_col_ind_kind,
        order: data_col_ind_order,
        rows: data_col_ind_rows,
        cols: data_col_ind_cols,
    };
    let data_values = Buffer {
        pointer: data_values,
        len: data_values_len,
        kind: data_values_kind,
        order: data_values_order,
        rows: data_values_rows,
        cols: data_values_cols,
    };
    let num = Buffer {
        pointer: num,
        len: num_len,
        kind: num_kind,
        order: num_order,
        rows: num_rows,
        cols: num_cols,
    };
    static mut LEFT: cuda_device::SharedArray<f64, 1024> = cuda_device::SharedArray::UNINIT;
    static mut RIGHT: cuda_device::SharedArray<f64, 1024> = cuda_device::SharedArray::UNINIT;
    let mut row = tid() / 128;
    let lane = tid() % 128;
    while row < n_samples {
        let mut base = 0;
        while base < n_features {
            let mut j = lane;
            while j < 1024 {
                unsafe {
                    LEFT[j as usize] = 0.0;
                }
                j += 128;
            }
            thread::sync_threads();
            let mut p = data_row_ptr.i(row) + lane;
            let end = data_row_ptr
                .i(row + 1)
                .min(data_col_ind.len)
                .min(data_values.len);
            while p < end {
                let g = data_col_ind.i(p);
                if g >= base && g < base + 1024 {
                    unsafe {
                        LEFT[(g - base) as usize] = data_values.f(p);
                    }
                }
                p += 128;
            }
            thread::sync_threads();
            let mut q = adj_row_ptr.i(row);
            let stop = adj_row_ptr
                .i(row + 1)
                .min(adj_col_ind.len)
                .min(adj_data.len);
            while q < stop {
                let other = adj_col_ind.i(q);
                if other < n_samples {
                    j = lane;
                    while j < 1024 {
                        unsafe {
                            RIGHT[j as usize] = 0.0;
                        }
                        j += 128;
                    }
                    thread::sync_threads();
                    p = data_row_ptr.i(other) + lane;
                    let end = data_row_ptr
                        .i(other + 1)
                        .min(data_col_ind.len)
                        .min(data_values.len);
                    while p < end {
                        let g = data_col_ind.i(p);
                        if g >= base && g < base + 1024 {
                            unsafe {
                                RIGHT[(g - base) as usize] = data_values.f(p);
                            }
                        }
                        p += 128;
                    }
                    thread::sync_threads();
                    j = lane;
                    while j < 1024 && base + j < n_features {
                        let feature = base + j;
                        let x = unsafe { LEFT[j as usize] };
                        let y = unsafe { RIGHT[j as usize] };
                        let diff = data_values.round(x - y);
                        let value = if data_values.kind == 0 {
                            (adj_data.f(q) as f32 * (x as f32 - y as f32) * (x as f32 - y as f32))
                                as f64
                        } else {
                            adj_data.f(q) * diff * diff
                        };
                        num.add(feature, value);
                        j += 128;
                    }
                    thread::sync_threads();
                }
                q += 1;
            }
            base += 1024;
        }
        row += stride() / 128;
    }
}
/// # Safety
/// The caller must supply validated, aligned device allocations matching each
/// descriptor and keep them alive until the borrowed stream completes.
#[kernel]
pub unsafe fn domain_autocorr_pre_den_sparse(
    data_col_ind: u64,
    data_values: u64,
    nnz: u64,
    mean_array: u64,
    den: u64,
    counter: u64,
    data_col_ind_len: u64,
    data_col_ind_kind: u64,
    data_col_ind_order: u64,
    data_col_ind_rows: u64,
    data_col_ind_cols: u64,
    data_values_len: u64,
    data_values_kind: u64,
    data_values_order: u64,
    data_values_rows: u64,
    data_values_cols: u64,
    mean_array_len: u64,
    mean_array_kind: u64,
    mean_array_order: u64,
    mean_array_rows: u64,
    mean_array_cols: u64,
    den_len: u64,
    den_kind: u64,
    den_order: u64,
    den_rows: u64,
    den_cols: u64,
    counter_len: u64,
    counter_kind: u64,
    counter_order: u64,
    counter_rows: u64,
    counter_cols: u64,
) {
    let data_col_ind = Buffer {
        pointer: data_col_ind,
        len: data_col_ind_len,
        kind: data_col_ind_kind,
        order: data_col_ind_order,
        rows: data_col_ind_rows,
        cols: data_col_ind_cols,
    };
    let data_values = Buffer {
        pointer: data_values,
        len: data_values_len,
        kind: data_values_kind,
        order: data_values_order,
        rows: data_values_rows,
        cols: data_values_cols,
    };
    let mean_array = Buffer {
        pointer: mean_array,
        len: mean_array_len,
        kind: mean_array_kind,
        order: mean_array_order,
        rows: mean_array_rows,
        cols: mean_array_cols,
    };
    let den = Buffer {
        pointer: den,
        len: den_len,
        kind: den_kind,
        order: den_order,
        rows: den_rows,
        cols: den_cols,
    };
    let counter = Buffer {
        pointer: counter,
        len: counter_len,
        kind: counter_kind,
        order: counter_order,
        rows: counter_rows,
        cols: counter_cols,
    };
    let mut p = tid();
    while p < nnz {
        let gene = data_col_ind.i(p);
        let square = if data_values.kind == 0 {
            let delta = data_values.single(p) - mean_array.single(gene);
            (delta * delta) as f64
        } else {
            let delta = data_values.f(p) - mean_array.f(gene);
            delta * delta
        };
        counter.add(gene, 1.0);
        den.add(gene, square);
        p += stride();
    }
}
