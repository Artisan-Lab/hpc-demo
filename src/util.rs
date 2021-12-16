use crate::{Matrix, MatrixWithAtomicCell, LENGTH};
use atomic_float::AtomicF64;
use ndarray::parallel::prelude::*;
use ndarray::Axis;
use rand::prelude::*;
use std::sync::atomic::Ordering;

/// 串行的用随机数初始化一个矩阵
pub fn init_matrix_sequential() -> Matrix {
    let mut matrix = Matrix::uninit((LENGTH, LENGTH));
    let mut rng = rand::thread_rng();
    for elem in matrix.iter_mut() {
        elem.write(rng.gen_range(0.0..1000.0f64));
    }
    unsafe { matrix.assume_init() }
}

/// 并行的用随机数初始化一个矩阵。以行为单位，每行为一个子任务
pub fn init_matrix_parallel() -> Matrix {
    let mut matrix = Matrix::uninit((LENGTH, LENGTH));
    matrix
        .axis_iter_mut(Axis(0)) //Every Line
        .into_par_iter()
        .for_each(|row| {
            // Thread Rng is not thread safe. We have to init a thread rng for each task
            let mut rng = rand::thread_rng();
            for elem in row {
                elem.write(rng.gen_range(0.0..1000.0f64));
            }
        });
    unsafe { matrix.assume_init() }
}

// 由于孤儿规则，没有办法为MatrixWithAtomicCell实现From,为了方便起见，直接定义新的函数
/// 将一个Matrix转换成MatrixWithAtomicCell
pub fn from_matrix_to_atomic(matrix: &Matrix) -> MatrixWithAtomicCell {
    let mut res = MatrixWithAtomicCell::uninit((LENGTH, LENGTH));
    res.iter_mut()
        .zip(matrix.iter())
        .for_each(|(res_item, matrix_item)| {
            let val = AtomicF64::new(*matrix_item);
            res_item.write(val);
        });
    unsafe { res.assume_init() }
}

/// 将一个MatrixWithAtomicCell转换成一个Matrix
pub fn from_atomic_to_matrix(matrix: &MatrixWithAtomicCell) -> Matrix {
    let mut res = Matrix::uninit((LENGTH, LENGTH));
    res.iter_mut()
        .zip(matrix.iter())
        .for_each(|(res_item, matrix_item)| {
            let val = matrix_item.load(Ordering::Relaxed);
            res_item.write(val);
        });
    unsafe { res.assume_init() }
}

/// 引入耗时的浮点操作来占用当前线程的fpu
pub fn consume_fpu_for_a_long_time() {
    let mut tmp = 1.000000001f64;
    for _ in 0..1000000000 {
        tmp = tmp * 1.000000001f64;
    }
    // 使用tmp的结果来强制运算，防止被优化
    consume(tmp);
}

fn consume(input: f64) {
    let mut v = Vec::new();
    v.push(input);
}