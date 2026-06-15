CUDA Webcam Filter Pipeline Implementation

Overview

This project extends the CUDA webcam filter application by implementing a simple filter pipeline capable of applying multiple filters sequentially in real time.

The objective was to demonstrate filter chaining, transition effects, and performance monitoring using GPU-accelerated image processing.

--------------------------------------------------

Part 1 – Filter Pipeline Architecture

A basic filter pipeline was implemented using two stages:

1. Blur Filter
2. Sharpen Filter

Pipeline flow:

Input Frame
    ↓
Blur Filter
    ↓
Sharpen Filter
    ↓
Output Frame

Intermediate image buffers were used to store the result of each stage before passing it to the next filter.

Memory Management

Separate image buffers were allocated for:

- Input frame
- Stage 1 output
- Stage 2 output
- Final transition output

This prevents data corruption between pipeline stages and simplifies filter chaining.

--------------------------------------------------

Part 2 – Wipe Transition

A wipe transition effect was implemented.

The transition gradually replaces the original image with the processed pipeline output from left to right.

The wipe position increases continuously during execution and resets after reaching the end of the frame.

This creates a continuous visual demonstration of the pipeline effect.

--------------------------------------------------

Part 3 – Performance Analysis

The execution time of the entire filter pipeline is measured using OpenCV timing functions.

The following information is displayed in real time:

- Pipeline FPS
- Pipeline execution time (milliseconds)

Example results:

Single Filter: 1.2 ms

Blur + Sharpen Pipeline: 2.4 ms

Actual values depend on hardware and image resolution.

Bottlenecks

The main bottlenecks identified are:

- Multiple GPU kernel launches
- Memory transfers between processing stages
- Additional operations required for transition rendering

--------------------------------------------------

CUDA Streams

The current implementation uses a simple sequential pipeline.

CUDA streams were investigated as a future optimization method to allow concurrent execution of pipeline stages and improve throughput.

--------------------------------------------------

Results

Successfully implemented:

- Multi-stage filter pipeline
- Blur → Sharpen filter chain
- Wipe transition effect
- Real-time FPS monitoring
- Pipeline execution timing

--------------------------------------------------

Future Improvements

Potential future optimizations include:

- Dynamic filter addition and removal
- Multiple CUDA streams
- More transition types
- Shared memory optimization
- Real-time performance graphs
- User-configurable filter chains

--------------------------------------------------

Conclusion

The project demonstrates a working CUDA filter pipeline architecture with sequential filter processing and transition effects. The implementation provides a foundation for more advanced real-time image processing systems while maintaining a simple and understandable design.