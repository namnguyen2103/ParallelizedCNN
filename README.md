# ParallelizedCNN

## Introduction
A Parallel and Distributed Programming school project that aims to implement a simple neural network from scratch in C and convert it to CUDA code to utilize GPU acceleration.

## Result
By parallelizing the neural network training process using CUDA, we achieved significant speedups in both forward and backward propagation. reducing from approximately 12 seconds per epoch on the CPU to just 0.3 seconds per epoch. This represents a 40x speedup, with the CUDA implementation producing the same results as the CPU version after many training epochs.

<table style="width:100%; border: none;">
  <tr>
    <td style="text-align: center; border: none;">
      <img src="https://github.com/user-attachments/assets/b753e7ac-4a6d-4b03-a540-b31fa858fed6" alt="CPU loss" style="width: 300px; height: auto;"><br>
      <span>CPU loss</span>
    </td>
    <td style="text-align: center; border: none;">
      <img src="https://github.com/user-attachments/assets/6c8ce938-a554-4ef4-8f6f-45da874f0143" alt="GPU loss" style="width: 300px; height: auto;"><br>
      <span>GPU loss</span>
    </td>
  </tr>
</table>


