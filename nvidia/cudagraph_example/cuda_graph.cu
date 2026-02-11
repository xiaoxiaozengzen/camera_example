#include <iostream>

#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                         \
            std::cerr << "CUDA error in " << __FILE__ << " at line "      \
                      << __LINE__ << ": " << cudaGetErrorString(err)      \
                      << " (" << err << ")" << std::endl;                 \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

__global__ void kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n/2) {
        data[idx] += 2.0f; // Example operation
    }
}

void HostCallback(void* userData) {
    float* value = static_cast<float*>(userData);
    int size = 8;
    std::cout << "Host callback called with value: ";
    for (int i = 0; i < size; ++i) {
        std::cout << value[i] << " ";
    }
    std::cout << std::endl;
}

void cudaLaunchHostFunc() {
    const int n = 8;
    float* d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));
    float* h_data = new float[n];

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // 异步操作，在指定stream上执行
    CHECK_CUDA(cudaMemsetAsync(d_data, 0, n * sizeof(float), stream));

    kernel<<<2, 4, 0, stream>>>(d_data, n);

    CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));

    // 在stream上注册一个host callback函数，确保在前面的操作完成后在主机上执行
    // 函数不能包含任何CUDA API调用，否则会有未问题
    CHECK_CUDA(cudaLaunchHostFunc(stream, HostCallback, h_data));

    CHECK_CUDA(cudaStreamSynchronize(stream));

    // Cleanup
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaStreamDestroy(stream));
    delete[] h_data;
}

__global__ void vectorAdd(float* vec, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        vec[idx] += 1.0f; // Example operation
    }
}

void cuda_graph_example() {
    float* h_vec = new float[8];
    for (int i = 0; i < 8; ++i) {
        h_vec[i] = 0.0f; // Initialize host vector
    }
    float* d_vec = nullptr;
    CHECK_CUDA(cudaMalloc(&d_vec, 8 * sizeof(float)));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaMemsetAsync(d_vec, 0, 8 * sizeof(float), stream));

    // 捕获CUDA Graph，
    // 捕获期间在stream上执行的所有操作将被记录到graph中，但是未实际执行，在cudaGraphLaunch时才会执行
    cudaGraph_t graph;
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    vectorAdd<<<1, 8, 0, stream>>>(d_vec, 4);
    CHECK_CUDA(cudaMemcpyAsync(h_vec, d_vec, 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // 实例化图
    cudaGraphExec_t graphExec;
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // 执行图
    for (int i = 0; i < 5; ++i) {
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }

    CHECK_CUDA(cudaStreamSynchronize(stream));
    std::cout << "After graph execution " << 5 << ": ";
    for (int j = 0; j < 8; ++j) {
        std::cout << h_vec[j] << " ";
    }
    std::cout << std::endl;

    // Cleanup
    CHECK_CUDA(cudaGraphExecDestroy(graphExec));
    CHECK_CUDA(cudaGraphDestroy(graph));
    CHECK_CUDA(cudaFree(d_vec));
    CHECK_CUDA(cudaStreamDestroy(stream));
    delete[] h_vec;
    
}

int main() {
    std::cout << "=================== cudaLaunchHostFunc Example ===================" << std::endl;
    cudaLaunchHostFunc();
    std::cout << "=================== cudaGraph Example ===================" << std::endl;
    cuda_graph_example();

    return 0;
}