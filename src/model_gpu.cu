#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <string.h>

#define NUM_TRAIN_IMAGES 200
#define NUM_TEST_IMAGES 50
#define IMG_HEIGHT 64
#define IMG_WIDTH 64
#define IMG_DEPTH 3
#define FLATTENED_SIZE (IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)
#define NUM_CLASSES 1
#define BATCH_SIZE 100

#define EPSILON 1e-10

// Activation functions
__device__ double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

__device__ double sigmoid_derivative(double x) {
    double sx = sigmoid(x);
    return sx * (1 - sx);
}

__device__ double relu(double x) {
    return x > 0 ? x : 0;
}

__device__ double relu_derivative(double x) {
    return x > 0 ? 1 : 0;
}

__device__ double stable_sigmoid(double x) {
    double sig = sigmoid(x);
    return fmax(fmin(sig, 1.0 - EPSILON), EPSILON);
}

// Compute loss
__device__ double compute_loss(double y_true, double y_pred) {
    y_pred = fmin(fmax(y_pred, EPSILON), 1.0 - EPSILON);
    double term1 = y_true * log(y_pred);
    double term2 = (1 - y_true) * log(1 - y_pred);
    return -(term1 + term2);
}

__device__ double compute_derivative_loss(double y_true, double y_pred) {
    double term1 = y_true / y_pred;
    double term2 = - (1 - y_true) / (1 - y_pred);
    return - (term1 + term2);
}

// Host version of compute_loss
double host_compute_loss(double y_true, double y_pred) {
    y_pred = fmin(fmax(y_pred, EPSILON), 1.0 - EPSILON);
    double term1 = y_true * log(y_pred);
    double term2 = (1 - y_true) * log(1 - y_pred);
    return -(term1 + term2);
}

double random_normal() {
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

void he_init(double *weights, int fan_in, int fan_out) {
    double stddev = sqrt(2.0 / fan_in);
    for (int i = 0; i < fan_in * fan_out; i++) {
        weights[i] = random_normal() * stddev;
    }
}

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    double *w1;
    double *b1;
    double *w2;
    double *b2;
    double *z1;
    double *a1;
    double *z2;
    double *a2;
} NeuralNetwork;

// Initialize the neural network
void init_nn(NeuralNetwork *nn, int input_size, int hidden_size, int output_size) {
    nn->input_size = input_size;
    nn->hidden_size = hidden_size;
    nn->output_size = output_size;

    nn->w1 = (double*)malloc(input_size * hidden_size * sizeof(double));
    nn->b1 = (double*)malloc(hidden_size * sizeof(double));
    nn->w2 = (double*)malloc(hidden_size * output_size * sizeof(double));
    nn->b2 = (double*)malloc(output_size * sizeof(double));

    he_init(nn->w1, input_size, hidden_size);
    he_init(nn->w2, hidden_size, output_size);
}

void free_nn(NeuralNetwork *nn) {
    free(nn->w1);
    free(nn->b1);
    free(nn->w2);
    free(nn->b2);
}



// Forward propagation
__global__ void forward_layer1(double *gpu_input, double *gpu_z1, double *gpu_a1, double *gpu_w1, double *gpu_b1, int batch_size, int input_size, int hidden_size) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < batch_size && Col < hidden_size) {
        double sum1 = 0.0;
        for (int k = 0; k < input_size; k++) {
            sum1 += gpu_input[Row * input_size + k] * gpu_w1[k * hidden_size + Col];
        }
        gpu_z1[Row * hidden_size + Col] = sum1 + gpu_b1[Col];
        
        gpu_a1[Row * hidden_size + Col] = relu(gpu_z1[Row * hidden_size + Col]);
    }
}

__global__ void forward_layer2(double *gpu_a1, double *gpu_z2, double *gpu_a2, double *gpu_w2, double *gpu_b2, int batch_size, int hidden_size, int output_size) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < batch_size && Col < output_size) {
        double sum2 = 0.0;
        for (int k = 0; k < hidden_size; k++) {
            sum2 += gpu_a1[Row * hidden_size + k] * gpu_w2[k * output_size + Col];
        }
        gpu_z2[Row * output_size + Col] = sum2 + gpu_b2[Col];
        gpu_a2[Row * output_size + Col] = stable_sigmoid(gpu_z2[Row * output_size + Col]);

    }
}

// Backward propagation
__global__ void backward1(double *gpu_loss, double *gpu_dLa2, double *gpu_da2z2, double *gpu_db2, double *gpu_y_true, double *gpu_z2, double *gpu_a2, int batch_size, int output_size) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    if (Row < batch_size && Col < output_size) {
        gpu_loss[Row * output_size + Col] = compute_loss(gpu_y_true[Row * output_size + Col], gpu_a2[Row * output_size + Col]);
        gpu_dLa2[Row * output_size + Col] = compute_derivative_loss(gpu_y_true[Row * output_size + Col], gpu_a2[Row * output_size + Col]);
        gpu_da2z2[Row * output_size + Col] = sigmoid_derivative(gpu_z2[Row * output_size + Col] );
        gpu_db2[Row * output_size + Col] = gpu_dLa2[Row * output_size + Col] * gpu_da2z2[Row * output_size + Col];
    }
}

__global__ void backward2(double *gpu_dw2, double *gpu_dLa1, double *gpu_da1z1, double *gpu_db1, double *gpu_dw1, double *gpu_input, double *gpu_z1, double *gpu_a1, double *gpu_w2, double *gpu_db2, int batch_size, int input_size, int hidden_size) {
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;

    
       if (Row < batch_size && Col < hidden_size) {
            gpu_dw2[Row * hidden_size + Col] = gpu_db2[Row] * gpu_a1[Row * hidden_size + Col];
            
            gpu_dLa1[Row * hidden_size + Col] = gpu_db2[Row] * gpu_w2[Col];
            gpu_da1z1[Row * hidden_size + Col] = relu_derivative(gpu_z1[Row * hidden_size + Col]);
            gpu_db1[Row * hidden_size + Col] = gpu_dLa1[Row * hidden_size + Col] * gpu_da1z1[Row * hidden_size + Col];

         for (int i = 0; i < input_size; i++) {
                gpu_dw1[Row * input_size * hidden_size + i * hidden_size + Col] = gpu_input[Row * input_size + i]  * gpu_db1[Row * hidden_size + Col] ;
            } 
       }
}

// Compute loss and gradient for updating
__global__ void reduction(
    double *gpu_dw2, double *gpu_db1, double *gpu_dw1, 
    double *gpu_w2, double *gpu_b1, double *gpu_w1, 
    double *gpu_loss, double *gpu_db2, double *gpu_epoch_loss, double *gpu_b2,
    int batch_size, int input_size, int hidden_size, double learning_rate) {
    
    int tid = threadIdx.x;
    int node = blockIdx.x;

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (tid + s >= 100) {
                if (node == 0) {
                    gpu_loss[tid] += 0;
                    gpu_db2[tid] += 0;
                }

                gpu_dw2[tid * hidden_size + node] += 0;
                gpu_db1[tid * hidden_size + node] += 0;
                for (int i = 0; i < input_size; i++) {
                    gpu_dw1[tid * input_size * hidden_size + i * hidden_size + node] += 0;
                }
            } else {
                if (node == 0) {
                    gpu_loss[tid] += gpu_loss[tid + s];
                    gpu_db2[tid] += gpu_db2[tid + s];
                }
                gpu_dw2[tid * hidden_size + node] += gpu_dw2[(tid + s) * hidden_size + node];
                gpu_db1[tid * hidden_size + node] += gpu_db1[(tid + s) * hidden_size + node];
                for (int i = 0; i < input_size; i++) {
                    gpu_dw1[tid * input_size * hidden_size + i * hidden_size + node] += gpu_dw1[(tid + s) * input_size * hidden_size + i * hidden_size + node];
                }
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (node == 0) {
            gpu_db2[0] /= batch_size;
            *gpu_epoch_loss += gpu_loss[0];
            gpu_b2[0] -= learning_rate * gpu_db2[0];
        }
        gpu_dw2[node] /= batch_size;
        gpu_db1[node] /= batch_size;
        gpu_w2[node] -= learning_rate * gpu_dw2[node];
        gpu_b1[node] -= learning_rate * gpu_db1[node];
        for (int i = 0; i < input_size; i++) {
            gpu_dw1[i * hidden_size + node] /= batch_size;
            gpu_w1[i * hidden_size + node] -= learning_rate * gpu_dw1[i * hidden_size + node];
        }
    }
}

// Predict function
void predict(NeuralNetwork *nn, double *input, double *output, double *gpu_input, double *gpu_z1, double *gpu_a1, double *gpu_z2, double *gpu_a2, double *gpu_w1, double *gpu_b1, double *gpu_w2, double *gpu_b2) {
    dim3 gridSize(16, 16);
    dim3 blockSize(16, 16);
    forward_layer1<<<gridSize, blockSize>>>(gpu_input, gpu_z1, gpu_a1, gpu_w1, gpu_b1, BATCH_SIZE, nn->input_size, nn->hidden_size);
    cudaDeviceSynchronize();
    forward_layer2<<<gridSize, blockSize>>>(gpu_a1, gpu_z2, gpu_a2, gpu_w2, gpu_b2, BATCH_SIZE, nn->hidden_size, nn->output_size);
    cudaDeviceSynchronize();
    cudaMemcpy(output, gpu_a2, nn->output_size * sizeof(double), cudaMemcpyDeviceToHost);
}

// Test accuracy
double test_accuracy(NeuralNetwork *nn, double *x_test, double *y_test, int data_size, double *gpu_input, double *gpu_z1, double *gpu_a1, double *gpu_z2, double *gpu_a2, double *gpu_w1, double *gpu_b1, double *gpu_w2, double *gpu_b2) {
    int correct = 0;
    double *output = (double*)malloc(nn->output_size * sizeof(double));

    for (int i = 0; i < data_size; i++) {
        double *input = &x_test[i * nn->input_size];
        double y_true = y_test[i];

        cudaMemcpy(gpu_input, input, nn->input_size * sizeof(double), cudaMemcpyHostToDevice);
        predict(nn, input, output, gpu_input, gpu_z1, gpu_a1, gpu_z2, gpu_a2, gpu_w1, gpu_b1, gpu_w2, gpu_b2);

        if ((output[0] > 0.5) == y_true) {
            correct++;
        }
    }

    free(output);
    return (double)correct / data_size;
}

// Train the network
void train(NeuralNetwork *nn, double *x_train, double* y_train, int epochs, double learning_rate, int batch_size, int train_data_size, double *x_test, double *y_test, int test_data_size) {
    double *gpu_input, *gpu_z1, *gpu_a1, *gpu_z2, *gpu_a2, *gpu_w1, *gpu_b1, *gpu_w2, *gpu_b2, *gpu_dw1, *gpu_db1, *gpu_dw2, *gpu_db2;
    double *gpu_epoch_loss, *gpu_loss, *gpu_dLa2, *gpu_da2z2, *gpu_dLa1, *gpu_da1z1, *gpu_y_true;

    cudaMalloc(&gpu_input, BATCH_SIZE * nn->input_size * sizeof(double));
    cudaMalloc(&gpu_z1, BATCH_SIZE * nn->hidden_size * sizeof(double));
    cudaMalloc(&gpu_a1, BATCH_SIZE * nn->hidden_size * sizeof(double));
    cudaMalloc(&gpu_z2, BATCH_SIZE * nn->output_size * sizeof(double));
    cudaMalloc(&gpu_a2, BATCH_SIZE * nn->output_size * sizeof(double));
    cudaMalloc(&gpu_w1, nn->input_size * nn->hidden_size * sizeof(double));
    cudaMalloc(&gpu_b1, nn->hidden_size * sizeof(double));
    cudaMalloc(&gpu_w2, nn->hidden_size * nn->output_size * sizeof(double));
    cudaMalloc(&gpu_b2, nn->output_size * sizeof(double));
    cudaMalloc(&gpu_dw1, BATCH_SIZE * nn->input_size * nn->hidden_size * sizeof(double));
    cudaMalloc(&gpu_db1, BATCH_SIZE * nn->hidden_size * sizeof(double));
    cudaMalloc(&gpu_dw2, BATCH_SIZE * nn->hidden_size * nn->output_size * sizeof(double));
    cudaMalloc(&gpu_db2, BATCH_SIZE * nn->output_size * sizeof(double));
    
    
    cudaMemcpy(gpu_w1, nn->w1, nn->input_size * nn->hidden_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b1, nn->b1, nn->hidden_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_w2, nn->w2, nn->hidden_size * nn->output_size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_b2, nn->b2, nn->output_size * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&gpu_loss, BATCH_SIZE * nn->output_size * sizeof(double));
    cudaMalloc(&gpu_dLa2, BATCH_SIZE * nn->output_size * sizeof(double));
    cudaMalloc(&gpu_da2z2, BATCH_SIZE * nn->output_size * sizeof(double));
    cudaMalloc(&gpu_dLa1, BATCH_SIZE * nn->hidden_size * sizeof(double));
    cudaMalloc(&gpu_da1z1, BATCH_SIZE * nn->hidden_size * sizeof(double));
    cudaMalloc(&gpu_y_true, BATCH_SIZE * sizeof(double));
    cudaMalloc(&gpu_epoch_loss, sizeof(double));
    
    double epoch_loss;

    //int no = (NUM_TRAIN_IMAGES + BATCH_SIZE - 1) / BATCH_SIZE;
    
    dim3 gridSize(16, 16);
    dim3 blockSize(16, 16);

    for (int epoch = 0; epoch < epochs; epoch++) {
        epoch_loss = 0.0;
        cudaMemcpy(gpu_epoch_loss, &epoch_loss, sizeof(double), cudaMemcpyHostToDevice);
        
        int num_batches = (train_data_size + batch_size - 1) / batch_size;  // Ceiling division

        for (int batch = 0; batch < num_batches; batch++) {
            int start = batch * batch_size;
            int end = start + batch_size < train_data_size ? start + batch_size : train_data_size;
            int current_batch_size = end - start;

            double *input = x_train + start * nn->input_size;
            double *y_true = y_train + start;

            cudaMemcpy(gpu_input, input, current_batch_size * nn->input_size * sizeof(double), cudaMemcpyHostToDevice);
            cudaMemcpy(gpu_y_true, y_true, current_batch_size * sizeof(double), cudaMemcpyHostToDevice);

            forward_layer1<<<gridSize, blockSize>>>(gpu_input, gpu_z1, gpu_a1, gpu_w1, gpu_b1, current_batch_size, nn->input_size, nn->hidden_size);
            cudaDeviceSynchronize();

            forward_layer2<<<gridSize, blockSize>>>(gpu_a1, gpu_z2, gpu_a2, gpu_w2, gpu_b2, current_batch_size, nn->hidden_size, nn->output_size);
            cudaDeviceSynchronize();
            
            backward1<<<gridSize, blockSize>>>(gpu_loss, gpu_dLa2, gpu_da2z2, gpu_db2, gpu_y_true, gpu_z2, gpu_a2, current_batch_size, nn->output_size);
            cudaDeviceSynchronize();
            
            backward2<<<gridSize, blockSize>>>(gpu_dw2, gpu_dLa1, gpu_da1z1, gpu_db1, gpu_dw1, gpu_input, gpu_z1, gpu_a1, gpu_w2, gpu_db2, current_batch_size, nn->input_size, nn->hidden_size);
            cudaDeviceSynchronize();
            
            reduction<<<nn->hidden_size, 128>>>(gpu_dw2, gpu_db1, gpu_dw1, gpu_w2, gpu_b1, gpu_w1, gpu_loss, gpu_db2, gpu_epoch_loss, gpu_b2, current_batch_size, nn->input_size, nn->hidden_size, learning_rate);
        
        }
        cudaMemcpy(&epoch_loss, gpu_epoch_loss, sizeof(double), cudaMemcpyDeviceToHost);

        epoch_loss /= NUM_TRAIN_IMAGES;

        printf("Epoch %d, Loss: %.8f Accuracy: %.2f%%\n", epoch + 1, epoch_loss, test_accuracy(nn, x_test, y_test, NUM_TEST_IMAGES, gpu_input, gpu_z1, gpu_a1, gpu_z2, gpu_a2, gpu_w1, gpu_b1, gpu_w2, gpu_b2) * 100.0);
    }

    cudaFree(gpu_input);
    cudaFree(gpu_z1);
    cudaFree(gpu_a1);
    cudaFree(gpu_z2);
    cudaFree(gpu_a2);
    cudaFree(gpu_w1);
    cudaFree(gpu_b1);
    cudaFree(gpu_w2);
    cudaFree(gpu_b2);
    cudaFree(gpu_dw1);
    cudaFree(gpu_db1);
    cudaFree(gpu_dw2);
    cudaFree(gpu_db2);
    cudaFree(gpu_loss);
    cudaFree(gpu_dLa2);
    cudaFree(gpu_da2z2);
    cudaFree(gpu_dLa1);
    cudaFree(gpu_da1z1);
    cudaFree(gpu_y_true);
    cudaFree(gpu_epoch_loss);
}


// Utility function to read CSV files
void read_csv(const char *filename, double *data, int rows, int cols) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (fscanf(file, "%lf,", &data[i * cols + j]) != 1) {
                perror("Error reading file");
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(file);
}

int main() {
    srand(12);

    double *train_data = (double*)malloc(NUM_TRAIN_IMAGES * FLATTENED_SIZE * sizeof(double));
    double *train_labels = (double*)malloc(NUM_TRAIN_IMAGES * sizeof(double));
    double *test_data = (double*)malloc(NUM_TEST_IMAGES * FLATTENED_SIZE * sizeof(double));
    double *test_labels = (double*)malloc(NUM_TEST_IMAGES * sizeof(double));

    read_csv("../data/final_train_images_float32.csv", train_data, NUM_TRAIN_IMAGES, FLATTENED_SIZE);
    read_csv("../data/final_train_labels_float32.csv", train_labels, NUM_TRAIN_IMAGES, 1);
    read_csv("../data/test_images_float32.csv", test_data, NUM_TEST_IMAGES, FLATTENED_SIZE);
    read_csv("../data/test_labels_float32.csv", test_labels, NUM_TEST_IMAGES, 1);

    // Normalize the data
    for (int i = 0; i < NUM_TRAIN_IMAGES * FLATTENED_SIZE; i++) {
        train_data[i] /= 255.0;
    }
    for (int i = 0; i < NUM_TEST_IMAGES * FLATTENED_SIZE; i++) {
        test_data[i] /= 255.0;
    }

    int input_size = FLATTENED_SIZE;
    int hidden_size = 128;
    int output_size = 1;

    NeuralNetwork nn;
    init_nn(&nn, input_size, hidden_size, output_size);

    train(&nn, train_data, train_labels, 100, 0.01, BATCH_SIZE, NUM_TRAIN_IMAGES, test_data, test_labels, NUM_TEST_IMAGES);

    free_nn(&nn);
    free(train_data);
    free(train_labels);
    free(test_data);
    free(test_labels);

    return 0;
}
