use ndarray::parallel::prelude::*;
use ndarray::{Array2, Axis};
use rand::prelude::*;
use std::time::SystemTime;

// 矩阵的每一维的长度。较大时更能体现并行的加速效果
const LENGTH: usize = 4096;

// 迭代轮数，在循环中取代LENGTH，来减少循环的次数
const ITERATION: usize = 5;

type Matrix = Array2<f64>;

fn init_matrix_serial(matrix: &mut Matrix) {
    let mut rng = rand::thread_rng();
    for elem in matrix.iter_mut() {
        *elem = rng.gen_range(0.0..1000.0f64);
    }
}

fn init_matrix_parallel(matrix: &mut Matrix) {
    matrix
        .axis_iter_mut(Axis(0)) //Every Line
        .into_par_iter()
        .for_each(|row| {
            // Thread Rng is not thread safe. We have to init a thread rng for each task
            let mut rng = rand::thread_rng();
            for element in row {
                *element = rng.gen_range(0.0..1000.0f64);
            }
        });
}

fn demo() {
    let mut matrix_vj = Matrix::zeros((LENGTH, LENGTH));
    let mut matrix_dm = Matrix::zeros((LENGTH, LENGTH));
    let mut reduce_ij = Matrix::zeros((LENGTH, LENGTH));

    init_matrix_serial(&mut matrix_vj);
    init_matrix_serial(&mut matrix_dm);
    init_matrix_serial(&mut reduce_ij);

    let mut clone_matrix_vj = matrix_vj.clone();

    // 串行计算
    let sys_time = SystemTime::now();
    serial_calculate(&mut matrix_vj, &mut matrix_dm, &mut reduce_ij);
    let run_time_serial = sys_time.elapsed().unwrap().as_millis();

    // 并行计算
    let sys_time = SystemTime::now();
    parallel_calculate(&mut clone_matrix_vj, &mut matrix_dm, &mut reduce_ij);
    let run_time_parallel = sys_time.elapsed().unwrap().as_millis();

    assert_eq!(matrix_vj, clone_matrix_vj);

    println!("serial run time for demo: {} ms", run_time_serial);
    println!("parallel run time for demo: {} ms", run_time_parallel);
}

fn serial_calculate(matrix_vj: &mut Matrix, matrix_dm: &mut Matrix, reduce_ij: &mut Matrix) {
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
fn parallel_calculate(matrix_vj: &mut Matrix, matrix_dm: &mut Matrix, reduce_ij: &mut Matrix) {
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

fn test_init_matrix_serial() {
    let mut matrix = Matrix::zeros((LENGTH, LENGTH));
    for _ in 0..ITERATION {
        init_matrix_serial(&mut matrix);
    }
}

fn test_init_matrix_parallel() {
    let mut matrix = Matrix::zeros((LENGTH, LENGTH));
    for _ in 0..ITERATION {
        init_matrix_parallel(&mut matrix);
    }
}

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
    // init matrix serially
    let run_time = get_run_time(test_init_matrix_serial);
    println!("init matrix serially: {} ms", run_time);

    // init matrix parallel
    let run_time = get_run_time(test_init_matrix_parallel);
    println!("init matrix parallel: {} ms", run_time);

    // run demo
    let run_time = get_run_time(demo);
    println!("Total demo time: {} ms", run_time);
}
