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

## Result

```
h_value, 0, 0
h_value, 1, 10
---
h_value, 2, 20
h_value, 3, 30
h_value, 4, 40
---
h_value, 5, 50
h_value, 6, 60
---
h_value, 7, 70
h_value, 8, 80
h_value, 9, 90
h_value, 10, 100
h_value, 11, 110
---
h_value, 12, 120
h_value, 13, 130
h_value, 14, 140
h_value, 15, 150
h_value, 16, 160
cuda:Error:no error
h_value, 0, 0
h_value, 1, 0
h_value, 2, 0
h_value, 3, 20
h_value, 4, 50
h_value, 5, 0
h_value, 6, 50
h_value, 7, 0
h_value, 8, 70
h_value, 9, 150
h_value, 10, 240
h_value, 11, 340
h_value, 12, 0
h_value, 13, 120
h_value, 14, 250
h_value, 15, 390
h_value, 16, 540
```

