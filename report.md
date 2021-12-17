# Rust高性能计算

首先我们想要回答的是，Rust是否适合进行高性能计算，经过调研之后，我认为总的来说，结果是肯定的。论文\[1\]比较了Rust和C/C++在各种不同程序上的性能对比，结果显示都能取得相近的性能。论文\[2\]比较了rayon和Open MP（rayon和Open MP分别是Rust和C++上最流行的并行计算框架)在求解N体问题（传统的并行计算的问题）上的影响，其结果是Open MP性能要略好于rayon(大概在1-1.1倍之间)，但是Rust可以显著降低开发的复杂度（代码行数等）。除了这两篇论文之外，也有一些网络上的讨论，比如\[3\]在实验中发现Rust and Rayon在多核机器上的可扩展性不高，在核数较少时，两者的性能相仿，但是在多核机器（36核）上比起 C++ and OpenMP慢两到三倍。\[4\]中对同一个并行程序的C++实现和Rust实现进行不同层次的优化，结果表明，C++程序在单核和多核所能达到的峰值性能上都与Rust相差不大，Rust程序同样显现了优秀的多核扩展性（但这个实验用的核数比较少，更大规模的核则效果未知）。其中一个有意思的现象是，有些对于C++程序有效的优化方法，对于Rust程序反而是反作用。(ps.\[4\]中的实验设计非常严谨，对我来说是非常值得一读的代码。)。\[5\]对Open MP(C++)和Rayon(Rust)的多线程性能在一组benchmark上进行了对比，我直接引用其结论：在编译器或者底层算法可以提供足够的优化的情况下，rayon和Open MP的表现差不多（我个人觉得这个结论比较弱）。Rayon在较大矩阵的排序和乘法方面表现更好，在其他测试上，都是Open MP表现更好。Rayon的优势在于，不安全的共享变量会导致编译错误，所以程序总是正确的。

综合以上材料，我觉得大概可以得出结论：用Rust+rayon编写高性能程序基本是可行的，rayon提供的原语，足够编写出性能接近于C++的并行程序。在核数较少的多核系统上，同样可以表现出良好的扩展性。但是当核的数量更多的时候，则需要进一步的验证。

接下来的内容将包含三部分内容。
1. 结合demo探讨Rust并行编程的基本用法
2. rayon的底层机制   
3. 并行编程可能的优化策略

### Rust并行编程的常见用法

我们通过一个简单的随机初始化矩阵的例子来演示下用rayon并行编程。我们使用ndarray来作为矩阵的数据表示。首先是最简单的串行的版本。我们使用一个随机数生成器来为矩阵中的每个元素生成一个随机值。这里我们使用了unsafe，因为我们一开始创建了一个未初始化的矩阵，需要用assume_init来告诉编译器我们初始化完成了。很容易验证这个unsafe是不会造成影响的，因为我们确实初始化了每一个元素。
```rust
type Matrix = Array2<f64>;
const LENGTH: usize = 4096;
fn init_matrix_sequential() -> Matrix {
    let mut matrix = Matrix::uninit((LENGTH, LENGTH));
    let mut rng = rand::thread_rng();
    for elem in matrix.iter_mut() {
        elem.write(rng.gen_range(0.0..1000.0f64));
    }
    unsafe { matrix.assume_init() }
}
```

下面是一个并行的版本。我们将矩阵按行进行划分，将初始化每一行作为一个单独的任务来执行。核心是这个into_par_iter()的函数,这个迭代器将会把每一个单独的任务分配给不同的线程来执行，底层由rayon来进行调度。需要注意的是，我们为每一行都使用了一个单独的随机数生成器，这是因为ThreadRng不是Send的，没法安全的在线程间共享。

```rust
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
```

这个简单的并行化的程序，相比于串行的版本，在我的电脑上取得了接近4倍的性能的提升，当然这还不是理论极限，仍有可以优化的空间。

由于into_par_iter会将不同的任务发给不同的线程，所以仍然需要遵循好Rust对于并发安全的要求，避免数据竞争的出现。一个可行的方法是预先做好任务的切分，避免防止共享的变量。如果一定要访问共享变量的话，则需要原子操作或者加锁。目前并不确定是否在某些场景下，unsafe是必须使用的，但在确保正确性的前提下适当使用unsafe可以简化编程的过程。

### rayon的底层机制

以下内容主要参考\[6\]。rayon是一个轻量级的并行计算框架，其底层主要实现了一个原语`join`，into_par_iter等都是基于`join`提供的上层的抽象。当然能使用迭代器等上层抽象的话还是应该尽量使用，这也是Rust推荐的使用方式。

`join`大概提供了如下的接口。`join`会接受两个闭包，这两个闭包应该是可以并行运行的。然后`join`将会负责调度这两个闭包的执行，在这两个闭包都计算结束后，返回计算结果。`join`并不保证这两个闭包一定会被并行的执行，这是根据程序运行时的状态所决定的。如果当前空闲的核只有一个，`join`可能会调度两个任务到同一个空闲的核上去执行，否则的话会调度到多个核上去执行。
```rust
join(|| do_something(), || do_something());
```
在底层的实现上，`join`使用了叫做work stealing的技术。具体的来说，每个线程（还是只是其中一个线程线程？）会维护一个本线程的等待执行任务的队列。当调用`join`时，会往当前线程的队列里压入两个闭包。当一个线程空闲时，首先会从自己本线程的队列里尝试取出任务来执行，如果没有待执行的任务的话，会尝试从别的线程的队列里面去取出待执行的任务到本地来执行。work stealing是一种线程间协作推进任务完成的技术，也使得rayon的实现较为轻量级。

### 可能的优化策略

1. 程序设计。影响并行的程序运行的效率最重要的因素在于程序的设计。具体来说，是需要找到程序中耗时较多的计算部分，将这部分内容进行任务拆分成可以并行完成的部分。根据Amadahl定律,优化的程序运行时间占原程序的比重越大，优化带来的收益也会越大。从程序设计的角度来说，大致有两种思路，一种是在程序的顶层将程序的任务进行切分，但这个对整个程序的并行度有一些要求（即可拆分为多个并行的任务），用来测试并行框架的任务可能符合这个需求，但实际的情况下可能不一定能够做到。另一种思路是对每一个耗时的计算任务进行拆分，从而达到程序整体性能的优化，这对于大多数程序都是可以实现的。
2. 编译优化。编译优化指的是通过一些编程的技巧，使得编译器能够编译出最优的代码，比如更好的数据局部性，向量化执行，循环展开，指令预取等等。对于Rust这样一个提供了高层抽象的语言来说，大部分优化工作都可以由编译器自动完成，只需要遵循Rust推荐的编程规范（比如尽量用迭代器来取代循环）。\[6\]中的结果表示，在允许部分牺牲浮点数计算精度的情况下，可以大幅提升性能。
3. 硬件优化。硬件同样对程序的执行效率有很大的影响。\[6\]中的工作表明，对于支持AVX512的CPU来说，相比于不支持该指令的CPU，性能可以提升68%。


### References
- \[1\] Energy Efficiency across Programming Languages: How Do Energy, Time, and Memory Relate?(SLE17)
- \[2\] Performance vs Programming Effort between Rust and C on Multicore Architectures: Case Study in N-Body（CLEI21）  
- \[3\] https://www.reddit.com/r/rust/comments/brre8o/a_scaling_comparison_of_rust_rayon_and_c_and/
- \[4\] https://parallel-rust-cpp.github.io/introduction.html
- \[5\] https://github.com/trsupradeep/15618-project
- \[6\] https://smallcultfollowing.com/babysteps/blog/2015/12/18/rayon-data-parallelism-in-rust/