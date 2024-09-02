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

// Define activation functions
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double stable_sigmoid(double x) {
    double sig = sigmoid(x);
    return fmax(fmin(sig, 1.0 - EPSILON), EPSILON);
}

double sigmoid_derivative(double x) {
    double sx = sigmoid(x);
    return sx * (1 - sx);
}

double relu(double x) {
    return x > 0 ? x : 0.0;
}

double relu_derivative(double x) {
    return x > 0 ? 1.0 : 0.0;
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

// Define the Neural Network structure
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

    nn->w1 = (double *)malloc(input_size * hidden_size * sizeof(double));
    nn->b1 = (double *)malloc(hidden_size * sizeof(double));
    nn->w2 = (double *)malloc(hidden_size * output_size * sizeof(double));
    nn->b2 = (double *)malloc(output_size * sizeof(double));

    nn->z1 = (double *)malloc(hidden_size * sizeof(double));
    nn->a1 = (double *)malloc(hidden_size * sizeof(double));
    nn->z2 = (double *)malloc(output_size * sizeof(double));
    nn->a2 = (double *)malloc(output_size * sizeof(double));

    if (!nn->w1 || !nn->b1 || !nn->w2 || !nn->b2 || !nn->z1 || !nn->a1 || !nn->z2 || !nn->a2) {
        perror("Error allocating memory");
        exit(EXIT_FAILURE);
    }

    he_init(nn->w1, input_size, hidden_size);
    he_init(nn->w2, hidden_size, output_size);
}

// Forward propagation
void forward(NeuralNetwork *nn, double *input) {
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->z1[i] = 0.0;
        for (int j = 0; j < nn->input_size; j++) {
            nn->z1[i] += input[j] * nn->w1[j * nn->hidden_size + i];
        }
        nn->z1[i] += nn->b1[i];
        nn->a1[i] = relu(nn->z1[i]);
    }

    for (int i = 0; i < nn->output_size; i++) {
        nn->z2[i] = 0.0;
        for (int j = 0; j < nn->hidden_size; j++) {
            nn->z2[i] += nn->a1[j] * nn->w2[j * nn->output_size + i];
        }
        nn->z2[i] += nn->b2[i];
        nn->a2[i] = stable_sigmoid(nn->z2[i]);
    }
    //printf("Activations a2 (predictions):\n");
    //for (int i = 0; i < nn->output_size; i++) {
        //printf("%f ", nn->a2[i]);
    //}
    //printf("\n");
}

// Compute loss
double compute_loss(double *y_true, double *y_pred, int size) {

    double loss = 0.0;
    for (int i = 0; i < size; i++) {
        double pred = y_pred[i];
        // Avoid log(0) and log(1)
        if (pred<1e-8)
            pred=1e-8;
        if(pred>1-1e-8)
            pred=1-1e-8;

        double true_value = y_true[i];
        double term1 = true_value * log(pred);
        double term2 = (1 - true_value) * log(1 - pred);

        //if (isnan(pred) || isnan(term1) || isnan(term2)) {
            //printf("NaN detected at index %d: true_value=%f, pred=%f, term1=%f, term2=%f\n", i, true_value, pred, term1, term2);
            //return -1; // Return an error code or handle it appropriately
        //}

        loss += -(term1 + term2);
    }
    return loss / size;
}

// Compute gradients
void backward(NeuralNetwork *nn, double *input, double *y_true, double *dw1, double *db1, double *dw2, double *db2) {
    double dL_da2[nn->output_size];
    double da2_dz2[nn->output_size];
    double dL_da1[nn->hidden_size];
    double da1_dz1[nn->hidden_size];

    for (int i = 0; i < nn->output_size; i++) {
        dL_da2[i] = -(*(y_true + i) / nn->a2[i] - (1 - *(y_true + i)) / (1 - nn->a2[i]));
        da2_dz2[i] = sigmoid_derivative(nn->z2[i]);
        db2[i] = dL_da2[i] * da2_dz2[i];
        for (int j = 0; j < nn->hidden_size; j++) {
            dw2[j * nn->output_size + i] = nn->a1[j] * db2[i];
        }
    }

    for (int i = 0; i < nn->hidden_size; i++) {
        dL_da1[i] = 0.0;
        for (int j = 0; j < nn->output_size; j++) {
            dL_da1[i] += dL_da2[j] * da2_dz2[j] * nn->w2[i * nn->output_size + j];
        }
        da1_dz1[i] = relu_derivative(nn->z1[i]);
        db1[i] = dL_da1[i] * da1_dz1[i];
        for (int j = 0; j < nn->input_size; j++) {
            dw1[j * nn->hidden_size + i] = input[j] * db1[i];
        }
    }
}

// Update parameters
void update_params(NeuralNetwork *nn, double *dw1, double *db1, double *dw2, double *db2, double learning_rate, int batch_size) {
    for (int i = 0; i < nn->input_size * nn->hidden_size; i++) {
        nn->w1[i] -= (learning_rate * dw1[i]) / batch_size;
    }
    for (int i = 0; i < nn->hidden_size; i++) {
        nn->b1[i] -= (learning_rate * db1[i]) / batch_size;
    }
    for (int i = 0; i < nn->hidden_size * nn->output_size; i++) {
        nn->w2[i] -= (learning_rate * dw2[i]) / batch_size;
    }
    for (int i = 0; i < nn->output_size; i++) {
        nn->b2[i] -= (learning_rate * db2[i]) / batch_size;
    }
}

// Predict function
void predict(NeuralNetwork *nn, double *input, double *output) {
    forward(nn, input);
    for (int i = 0; i < nn->output_size; i++) {
        output[i] = nn->a2[i] > 0.5 ? 1.0 : 0.0;
    }
}

// Test accuracy
double test_accuracy(NeuralNetwork *nn, double *x_test, double *y_test, int data_size) {
    int correct = 0;
    double *output = (double *)malloc(nn->output_size * sizeof(double));
    
    for (int i = 0; i < data_size; i++) {
        double *input = &x_test[i * nn->input_size];
        double y_true = y_test[i];
        
        predict(nn, input, output);
        
        if (output[0] == y_true) {
            correct++;
        }
    }

    free(output);
    return (double)correct / data_size;
}

// Train the network
void train(NeuralNetwork *nn, double *x_train, double *y_train, int epochs, double learning_rate, int batch_size, int train_data_size, double **best_w1, double **best_b1, double **best_w2, double **best_b2, double *x_test, double *y_test, int test_data_size) {
    double *dw1 = (double *)malloc(nn->input_size * nn->hidden_size * sizeof(double));
    double *db1 = (double *)malloc(nn->hidden_size * sizeof(double));
    double *dw2 = (double *)malloc(nn->hidden_size * nn->output_size * sizeof(double));
    double *db2 = (double *)malloc(nn->output_size * sizeof(double));
    double *batch_dw1 = (double *)malloc(nn->input_size * nn->hidden_size * sizeof(double));
    double *batch_db1 = (double *)malloc(nn->hidden_size * sizeof(double));
    double *batch_dw2 = (double *)malloc(nn->hidden_size * nn->output_size * sizeof(double));
    double *batch_db2 = (double *)malloc(nn->output_size * sizeof(double));
    
    if (!dw1 || !db1 || !dw2 || !db2) {
        perror("Error allocating memory for gradients");
        exit(EXIT_FAILURE);
    }
    
    double min_loss = DBL_MAX;
    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        int num_batches = (train_data_size + batch_size - 1) / batch_size; // Ceiling division
        
        for (int batch = 0; batch < num_batches; batch++) {
            int start = batch * batch_size;
            int end = start + batch_size < train_data_size ? start + batch_size : train_data_size;
            int current_batch_size = end - start;

            memset(dw1, 0, nn->input_size * nn->hidden_size * sizeof(double));
            memset(db1, 0, nn->hidden_size * sizeof(double));
            memset(dw2, 0, nn->hidden_size * nn->output_size * sizeof(double));
            memset(db2, 0, nn->output_size * sizeof(double));
            
            for (int i = start; i < end; i++) {
                double *input = x_train + i * nn->input_size;
                double *y_true = y_train + i;

                forward(nn, input);
                epoch_loss += compute_loss(y_true, nn->a2, nn->output_size);
                memset(batch_dw1, 0, nn->input_size * nn->hidden_size * sizeof(double));
                memset(batch_db1, 0, nn->hidden_size * sizeof(double));
                memset(batch_dw2, 0, nn->hidden_size * nn->output_size * sizeof(double));
                memset(batch_db2, 0, nn->output_size * sizeof(double));
                backward(nn, input, y_true, batch_dw1, batch_db1, batch_dw2, batch_db2);

                for (int j = 0; j < nn->input_size * nn->hidden_size; j++) {
                    dw1[j] += batch_dw1[j];
                }
                for (int j = 0; j < nn->hidden_size; j++) {
                    db1[j] += batch_db1[j];
                }
                for (int j = 0; j < nn->hidden_size * nn->output_size; j++) {
                    dw2[j] += batch_dw2[j];
                }
                for (int j = 0; j < nn->output_size; j++) {
                    db2[j] += batch_db2[j];
                }
            }
            update_params(nn, dw1, db1, dw2, db2, learning_rate, current_batch_size);
        }
        
        epoch_loss /= train_data_size;
        printf("Epoch %d, Loss: %.8f Accuracy: %.2f%%\n", epoch + 1, epoch_loss, test_accuracy(nn, x_test, y_test, NUM_TEST_IMAGES) * 100.0);

        if (epoch_loss < min_loss) {
            min_loss = epoch_loss;
            memcpy(*best_w1, nn->w1, nn->input_size * nn->hidden_size * sizeof(double));
            memcpy(*best_b1, nn->b1, nn->hidden_size * sizeof(double));
            memcpy(*best_w2, nn->w2, nn->hidden_size * nn->output_size * sizeof(double));
            memcpy(*best_b2, nn->b2, nn->output_size * sizeof(double));
        }
    }

    free(dw1);
    free(db1);
    free(dw2);
    free(db2);
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

    double *train_data = (double *)malloc(NUM_TRAIN_IMAGES * FLATTENED_SIZE * sizeof(double));
    double *train_labels = (double *)malloc(NUM_TRAIN_IMAGES * sizeof(double));
    double *test_data = (double *)malloc(NUM_TEST_IMAGES * FLATTENED_SIZE * sizeof(double));
    double *test_labels = (double *)malloc(NUM_TEST_IMAGES * sizeof(double));

    if (!train_data || !train_labels || !test_data || !test_labels) {
        perror("Error allocating memory for data");
        exit(EXIT_FAILURE);
    }

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

    double *best_w1 = (double *)malloc(input_size * hidden_size * sizeof(double));
    double *best_b1 = (double *)malloc(hidden_size * sizeof(double));
    double *best_w2 = (double *)malloc(hidden_size * output_size * sizeof(double));
    double *best_b2 = (double *)malloc(output_size * sizeof(double));

    if (!best_w1 || !best_b1 || !best_w2 || !best_b2) {
        perror("Error allocating memory for best parameters");
        exit(EXIT_FAILURE);
    }

    train(&nn, train_data, train_labels, 100, 0.01, BATCH_SIZE, NUM_TRAIN_IMAGES, &best_w1, &best_b1, &best_w2, &best_b2, test_data, test_labels, NUM_TEST_IMAGES);

    // Load the best parameters into the network
    memcpy(nn.w1, best_w1, input_size * hidden_size * sizeof(double));
    memcpy(nn.b1, best_b1, hidden_size * sizeof(double));
    memcpy(nn.w2, best_w2, hidden_size * output_size * sizeof(double));
    memcpy(nn.b2, best_b2, output_size * sizeof(double));

    // Free allocated memory
    free(nn.w1);
    free(nn.b1);
    free(nn.w2);
    free(nn.b2);
    free(nn.z1);
    free(nn.a1);
    free(nn.z2);
    free(nn.a2);
    free(train_data);
    free(train_labels);
    free(test_data);
    free(test_labels);
    free(best_w1);
    free(best_b1);
    free(best_w2);
    free(best_b2);

    return 0;
}
