# Segment Exclusive Prefix Scan

A algorithm adapted from:

Lecture 24: Fun With Parallel Algorithms Segmented Scan Neutral territory methods(CMU 15-418: Parallel Computer Architecture and Programming (Spring 2012)), which illuminated a work-efficient segmented scan.

Adaption still works without paddings if n is not power of 2.

Demo ver.

Advanced CUDA optimizations not implemented.

## Possible Optimizations

* In warp data exchange, do 10 iterations(block size 1024) for one pass and use shared memory.
* Use bitmap instead of uint8_t.
* Reverse data(n - 1 - (x)) in only first and final stage.
* Move ([d_value[n - 1] = 0) to the last uphill pass.

## More 
* Other scan operations/data types are welcomed.
