#include <iostream>
#include <string>
#include <sys/types.h>
#include <vector>
#include <fstream>

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

/********************************* STITCH *********************************/

int main() {

}