use atomic_float::AtomicF64;
use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis};
use std::sync::atomic::Ordering;
use std::time::SystemTime;

pub mod util;

use util::{from_atomic_to_matrix, from_matrix_to_atomic};
use util::{init_matrix_parallel, init_matrix_sequential};

// 矩阵的每一维的长度。较大时更能体现并行的加速效果
const LENGTH: usize = 4096;

// 迭代轮数，在循环中取代LENGTH，来减少循环的次数.ITEARTION应该小于等于LENGTH
const ITERATION: usize = 20;

/// 每个元素为f64的二维矩阵
type Matrix = Array2<f64>;
/// 每个元素为AtomicF64的二维矩阵
type MatrixWithAtomicCell = Array2<AtomicF64>;

fn demo() {
    let mut matrix_vj = init_matrix_parallel();
    let mut matrix_dm = init_matrix_parallel();
    let mut reduce_ij = init_matrix_parallel();

    let mut cloned_matrix_vj = matrix_vj.clone();
    let mut matrix_vj_with_atomic_cell = from_matrix_to_atomic(&matrix_vj);
    assert_eq!(matrix_vj, cloned_matrix_vj);

    // 串行计算
    let sys_time = SystemTime::now();
    sequential_calculate(&mut matrix_vj, &matrix_dm, &reduce_ij);
    let run_time_sequential = sys_time.elapsed().unwrap().as_millis();

    // 并行计算:行级别的
    let sys_time = SystemTime::now();
    parallel_calculate_low_level(&mut cloned_matrix_vj, &mut matrix_dm, &mut reduce_ij);
    let run_time_parallel_low_level = sys_time.elapsed().unwrap().as_millis();

    // 并行计算：更高级别的
    let sys_time = SystemTime::now();
    parallel_calculate_high_level(&mut matrix_vj_with_atomic_cell, &matrix_dm, &reduce_ij);
    let run_time_parallel_high_level = sys_time.elapsed().unwrap().as_millis();
    let cloned_matrix_vj_2 = from_atomic_to_matrix(&matrix_vj_with_atomic_cell);

    // 判断计算结果是否相等
    assert_eq!(matrix_vj, cloned_matrix_vj);
    // 浮点数由于计算精度会产生一些误差，我们只要求近似相等即可
    matrix_vj
        .iter()
        .zip(cloned_matrix_vj_2.iter())
        .for_each(|(iterm1, iterm2)| {
            assert!((*iterm1 - *iterm2).abs() < 1.0f64);
        });

    println!("sequential run time for demo: {} ms", run_time_sequential);
    println!(
        "parallel(low level) run time for demo: {} ms",
        run_time_parallel_low_level
    );
    println!(
        "parallel(high level) run time for demo: {} ms",
        run_time_parallel_high_level
    );
}

/// 串行计算
fn sequential_calculate(matrix_vj: &mut Matrix, matrix_dm: &Matrix, reduce_ij: &Matrix) {
    for jc in 0..ITERATION {
        for ic in 0..ITERATION {
            let dm_ij = matrix_dm.iter().nth(ic * LENGTH + jc).unwrap().to_owned()
                + matrix_dm.iter().nth(jc * LENGTH + ic).unwrap().to_owned();
            matrix_vj
                .iter_mut()
                .zip(reduce_ij.iter())
                .for_each(|(vj_ij, eri_ij)| {
                    *vj_ij += eri_ij * dm_ij;
                });
        }
    }
}

// 注意到按照行切分的话本身就会提升代码的局部性，从而使得运算效率提高
// 要像只衡量并行带来的效率提升，需要对于vj_line_iter分别在调用和不调用into_par_iter的情况下进行测试
/// 行级别的并行计算
fn parallel_calculate_low_level(matrix_vj: &mut Matrix, matrix_dm: &Matrix, reduce_ij: &Matrix) {
    for jc in 0..ITERATION {
        for ic in 0..ITERATION {
            let dm_ij = matrix_dm.iter().nth(ic * LENGTH + jc).unwrap().to_owned()
                + matrix_dm.iter().nth(jc * LENGTH + ic).unwrap().to_owned();
            // 以行为单位切分任务,每个iter将会遍历矩阵的行。并且允许并行
            let vj_line_iter = matrix_vj.axis_iter_mut(Axis(0)).into_par_iter();
            // 以行为单位切分任务,每个iter将会遍历矩阵的行。并且不允许并行
            //let vj_line_iter = matrix_vj.axis_iter_mut(Axis(0));

            // 从本地测试结果来看，对reduce_line_iter继续调用into_par_iter不会对结果产生影响。
            // let reduce_line_iter = reduce_ij.axis_iter(Axis(0)).into_par_iter();
            let reduce_line_iter = reduce_ij.axis_iter(Axis(0));

            vj_line_iter
                .zip(reduce_line_iter)
                .for_each(|(mut vj_line, reduce_line)| {
                    vj_line
                        .iter_mut()
                        .zip(reduce_line.iter())
                        .for_each(|(vj_ij, eri_ij)| {
                            *vj_ij += eri_ij * dm_ij;
                        })
                });
        }
    }
}

// 这个实现可能会导致数据竞争
// struct ThreadSafeBox(*mut f64);
// unsafe impl Send for ThreadSafeBox {}
// unsafe impl Sync for ThreadSafeBox {}

// fn parallel_calculate_high_level(matrix_vj: &mut Matrix, matrix_dm: &Matrix, reduce_ij: &Matrix) {
//     let vj_ptr = ThreadSafeBox(matrix_vj.as_mut_ptr());
//     (0..ITERATION).into_iter().into_par_iter().for_each(|jc| {
//         (0..ITERATION).into_iter().for_each(|ic| {
//             let dm_ij = matrix_dm.iter().nth(ic * LENGTH + jc).unwrap().to_owned()
//                 + matrix_dm.iter().nth(jc * LENGTH + ic).unwrap().to_owned();
//             let matrix_len = matrix_vj.len();
//             for (offset, eri_ij) in (0..matrix_len).zip(reduce_ij.iter()) {
//                 unsafe {
//                     let vj_ij_ptr = vj_ptr.0.offset(offset as isize);
//                     std::ptr::write(vj_ij_ptr, *vj_ij_ptr + *eri_ij * dm_ij);
//                 }
//             }
//         })
//     });
// }

struct ThreadSafeBox(*mut AtomicF64);
unsafe impl Send for ThreadSafeBox {}
unsafe impl Sync for ThreadSafeBox {}

/// 避免了数据竞争；上层循环开展的并行计算
fn parallel_calculate_high_level(
    matrix_vj: &mut MatrixWithAtomicCell,
    matrix_dm: &Matrix,
    reduce_ij: &Matrix,
) {
    let vj_ptr = ThreadSafeBox(matrix_vj.as_mut_ptr());
    (0..ITERATION).into_iter().into_par_iter().for_each(|jc| {
        (0..ITERATION).into_iter().into_par_iter().for_each(|ic| {
            let dm_ij = matrix_dm.iter().nth(ic * LENGTH + jc).unwrap().to_owned()
                + matrix_dm.iter().nth(jc * LENGTH + ic).unwrap().to_owned();
            let matrix_len = matrix_vj.len();
            for (offset, eri_ij) in (0..matrix_len).zip(reduce_ij.iter()) {
                unsafe {
                    let vj_ij_ptr = vj_ptr.0.offset(offset as isize);
                    (*vj_ij_ptr).fetch_add(*eri_ij * dm_ij, Ordering::Relaxed);
                }
            }
        })
    });
}

fn test_init_matrix_sequential() {
    for _ in 0..ITERATION {
        init_matrix_sequential();
    }
}

fn test_init_matrix_parallel() {
    for _ in 0..ITERATION {
        init_matrix_parallel();
    }
}

/// 执行f，并获取f的执行时间
fn get_run_time<F>(f: F) -> u128
where
    F: FnOnce(),
{
    let sys_time = SystemTime::now();
    f();
    let run_time = sys_time.elapsed().unwrap().as_millis();
    run_time
}

fn main() {
    // init matrix sequentially
    let run_time = get_run_time(test_init_matrix_sequential);
    println!("init matrix sequentially: {} ms", run_time);

    // init matrix parallel
    let run_time = get_run_time(test_init_matrix_parallel);
    println!("init matrix parallel: {} ms", run_time);

    // run demo
    let run_time = get_run_time(demo);
    println!("Total demo time: {} ms", run_time);
}
