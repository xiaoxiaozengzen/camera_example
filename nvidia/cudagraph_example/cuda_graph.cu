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

    /**
     * @brief 将device上devPtr指向的内存得前count字节设置为value，异步执行在stream上
     * @param devPtr 设备内存指针
     * @param value  要设置的值
     * @param count  要设置的字节数
     * @param stream 要执行的stream
     */
    CHECK_CUDA(cudaMemsetAsync(d_data, 0, n * sizeof(float), stream));

    kernel<<<2, 4, 0, stream>>>(d_data, n);

    CHECK_CUDA(cudaMemcpyAsync(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost, stream));

    /**
     * @brief 等待改stream上所有之前提交的任务完成后，调用host函数，host函数执行完后才继续执行stream上后续任务
     * @param stream  要等待的stream
     * @param callback 要调用的host函数
     * @param userData 传递给host函数的用户数据指针
     *
     * @note host函数不能包含任何CUDA API调用，
     */
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

/**
 * @brief 示例展示如何使用CUDA图（CUDA Graph）进行图的创建、实例化和执行
 * @note graph中可以存在的节点类型包括(每个类型都对应一个cuda函数)：
 *       - kernel节点：表示一个CUDA内核的执行
 *       - memcpy节点：表示host和device之间的内存拷贝操作
 *       - memset节点：表示对device内存进行设置的操作
 *       - host节点：执行CPU上的函数，一般都是cudaLaunchHostFunc提交的函数
 *       - subgraph节点：表示一个子图，可以嵌套图的结构
 *       - empty节点：表示一个空操作节点，可以用于同步或占位
 *       - event wait节点：表示等待一个CUDA事件完成
 *       - event record节点：表示记录一个CUDA事件
 *       - memory allocation节点：表示在图中进行内存分配操作
 *       - memory free节点：表示在图中进行内存释放操作
 * @note graph capture期间，CPU下发的kernel都没有执行，因此与GPU kernel执行状态相关的函数都不能使用，例如：
 *       - cudaStreamSynchronize
 *       - 其他会导致stream同步的函数，如cudaStreamQuery、cudaStreamWaitEvent等
 *       - cudaMemcpy替换成cudaMemcpyAsync，并指定stream参数
 *       - cudaMemset替换成cudaMemsetAsync，并指定stream参数
 *       - cudaMalloc替换成cudaMallocAsync，并指定stream参数
 *       - cudaFree替换成cudaFreeAsync，并指定stream参数
 */
void cuda_graph_example() {
    float* h_vec = nullptr;
    CHECK_CUDA(cudaMallocHost(&h_vec, 8 * sizeof(float)));
    for (int i = 0; i < 8; ++i) {
        h_vec[i] = 0.0f;
    }

    float* d_vec = nullptr;
    CHECK_CUDA(cudaMalloc(&d_vec, 8 * sizeof(float)));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    CHECK_CUDA(cudaMemsetAsync(d_vec, 0, 8 * sizeof(float), stream));

    // 1. 创建图并捕获stream上的操作
    cudaGraph_t graph;
    /**
     * @brief 开始捕获stream上的操作
     * @param stream 要捕获的stream
     * @param mode   捕获模式
     *               - cudaStreamCaptureModeGlobal: 捕获stream上所有提交的操作
     *               - cudaStreamCaptureModeThreadLocal: 只捕获当前线程提交到stream上的操作
     *               - cudaStreamCaptureModeRelaxed: 允许在捕获期间提交到stream上的操作被执行，而不是被捕获
     *
     * @note 当一个stream进入捕获模式得时候，所有提交到stream得操作都不会被执行，而是被记录下来，
     *       直到调用cudaStreamEndCapture结束捕获，才会将捕获的操作组成一个cudaGraph_t对象，
     */
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    vectorAdd<<<1, 8, 0, stream>>>(d_vec, 4);
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));

    // 2.实例化图
    cudaGraphExec_t graphExec;
    /**
     * @brief 从一个已经创建好的图对象graph中实例化一个可执行图对象graphExec
     * @param pGraphExec 输出参数，返回实例化后的可执行图对象
     * @param graph      输入参数，要实例化的图对象
     * @param pErrorNode 输出参数，如果实例化失败，返回导致失败的节点
     * @param pLogBuffer 输出参数，如果实例化失败，返回错误日志
     * @param logBufferSize 输出参数，pLogBuffer的大小
     *
     */
    CHECK_CUDA(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // 3.执行图
    for (int i = 0; i < 5; ++i) {
        /**
         * @brief 在指定的stream上执行一个已经实例化的图对象graphExec
         * @param graphExec 要执行的图对象
         * @param stream    要执行的stream
         *
         * @note 同一时刻，只有一个graphExec对象执行，
         *       每次launch，都会排队到stream上已经提交的操作之后执行，
         *       如果想并发执行graphExec，则需要创建多个graphExec对象，并在不同的stream上执行，
         */
        CHECK_CUDA(cudaGraphLaunch(graphExec, stream));
    }

    CHECK_CUDA(cudaMemcpyAsync(h_vec, d_vec, 8 * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    /**
     * @brief 以DOT格式将图对象graph输出到path中，便于可视化分析
     * @param graph  要输出的图对象
     * @param path   输出文件路径
     * @param flags  输出选项，控制输出的详细程度
     */
    CHECK_CUDA(cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose));
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
    CHECK_CUDA(cudaFreeHost(h_vec));
}

int main() {
    std::cout << "=================== cudaLaunchHostFunc Example ===================" << std::endl;
    cudaLaunchHostFunc();
    std::cout << "=================== cudaGraph Example ===================" << std::endl;
    cuda_graph_example();

    return 0;
}