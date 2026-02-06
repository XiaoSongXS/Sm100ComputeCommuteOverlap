//
// Green Context Sample: Fork-Join Pattern with CUDA Graph + CLC Persistent Kernels
// Self-contained, no external dependencies
//
// SM Partition:
//   - Primary partition: 120 SMs (reserved, no kernel launch)
//   - Remaining partition: 28 SMs (for Vector Mul kernel)
//   - Total: 148 SMs
//
// Kernel Launch:
//   - Default context: LOW priority stream, cluster 4x1x1, 512 blocks, Vector Add
//   - Remaining partition: HIGH priority stream, cluster 2x1x1, 40 blocks, Vector Mul (10s)
//   - NOTE: Primary Partition Stream is **NOT** used
//
// CLC Persistent Kernel:
//   - Kernels use ClusterLaunchControl to query next work
//   - Enables persistent kernel pattern with dynamic work fetching
//
// Synchronization Pattern (Fork-Join):
//   Sync Stream:              [Record fork_event] ---------> [Wait for Both done_events]
//                                      |                              ^
//                                      v                              |
//   Default Ctx (LOW prio):   [Wait fork_event] -> [Vec Add] -> [Record done_event]
//                                      |                                     |
//   Remaining Gtx (HIGH prio):[Wait fork_event] -> [Vec Mul] -> [Record done_event]
//

// Headers and what this file uses from each:
//
// <cuda.h> (CUDA Driver API)
//   Types:     CUresult, CUdevice, CUcontext, CUstream, CUevent, CUdevResource,
//              CUdevResourceDesc, CUgreenCtx
//   Constants: CUDA_SUCCESS, CU_STREAM_NON_BLOCKING, CU_EVENT_DEFAULT,
//              CU_DEV_RESOURCE_TYPE_SM, CU_GREEN_CTX_DEFAULT_STREAM
//   APIs:      cuInit, cuGetErrorString, cuDeviceGet, cuDevicePrimaryCtxRetain,
//              cuCtxSetCurrent, cuDeviceGetName, cuDeviceGetDevResource,
//              cuDevSmResourceSplitByCount, cuDevResourceGenerateDesc,
//              cuGreenCtxCreate, cuStreamCreateWithPriority, cuGreenCtxStreamCreate,
//              cuStreamCreate, cuEventCreate, cuEventRecord, cuStreamWaitEvent,
//              cuEventDestroy, cuStreamDestroy, cuGreenCtxDestroy,
//              cuDevicePrimaryCtxRelease
//
// <cuda_runtime.h> (CUDA Runtime API + device)
//   Host types: cudaError_t, cudaDeviceProp, cudaStream_t, cudaEvent_t,
//               cudaGraph_t, cudaGraphExec_t, cudaLaunchAttribute
//   Host constants: cudaSuccess, cudaStreamCaptureModeGlobal,
//                   cudaLaunchAttributeClusterDimension,
//                   cudaLaunchAttributeProgrammaticStreamSerialization,
//                   cudaFuncAttributeNonPortableClusterSizeAllowed,
//                   cudaFuncAttributeMaxDynamicSharedMemorySize
//   Host APIs:  cudaGetErrorString, cudaGetDeviceProperties, cudaDeviceGetAttribute,
//               cudaDeviceGetStreamPriorityRange, cudaMalloc, cudaMemset,
//               cudaFuncSetAttribute, cudaStreamBeginCapture, cudaLaunchKernelExC,
//               cudaStreamEndCapture, cudaGraphInstantiateWithFlags, cudaEventCreate,
//               cudaEventRecord, cudaGraphLaunch, cudaStreamSynchronize,
//               cudaEventElapsedTime, cudaEventDestroy, cudaGraphExecDestroy,
//               cudaGraphDestroy, cudaFree,
//               cudaGraphGetNodes, cudaGraphNodeGetType, cudaGraphDebugDotPrint
//   Device:    __global__, __device__, __shared__, __launch_bounds__,
//              __grid_constant__, __cvta_generic_to_shared__, __align__
//   Device types: uint4 (CLC response), blockIdx, threadIdx, blockDim
//   Device built-in: clock64
//
// <cstdio>
//   printf  (error messages and sample output)
//
// <cstdlib>
//   exit    (abort on CUDA error in macros)
//
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// Error Checking Macros
// ============================================================================
#define CUDA_DRIVER_CHECK(call)                                                \
    do {                                                                       \
        CUresult err = (call);                                                 \
        if (err != CUDA_SUCCESS) {                                             \
            const char* errStr;                                                \
            cuGetErrorString(err, &errStr);                                    \
            printf("CUDA Driver Error: %s at %s:%d\n", errStr, __FILE__, __LINE__); \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define CUDA_RUNTIME_CHECK(call)                                               \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            printf("CUDA Runtime Error: %s at %s:%d\n",                        \
                   cudaGetErrorString(err), __FILE__, __LINE__);               \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ============================================================================
// Constants
// ============================================================================
constexpr int THREADS_PER_BLOCK = 128;
constexpr int PRIMARY_PARTITION_SM_COUNT = 120;
constexpr int REMAINING_PARTITION_SM_COUNT = 28;
constexpr int DEFAULT_CTX_GRID_SIZE = 512;   // 512 blocks on default context
constexpr int REMAINING_GRID_SIZE = 40;      // 40 blocks on remaining partition (was 20)
constexpr int NUM_ITERATIONS = 5;            // Reduced for testing (was 50)
constexpr int VECTOR_SIZE = 1024 * 1024;     // 1M elements

// Sleep durations in seconds
constexpr float VEC_ADD_SLEEP_SECONDS = 0.2f;     // Vector Add on default context (LOW priority, 512 blocks)
constexpr float VEC_MUL_SLEEP_SECONDS = 0.1f;     // Vector Mul on remaining partition (HIGH priority, 40 blocks)

// ============================================================================
// Shared Memory Structure for CLC
// ============================================================================
struct __align__(16) CLCSharedMemory {
    // CLC response storage (16-byte aligned, 128-bit)
    alignas(16) uint4 clc_response;
    
    // mbarrier for CLC synchronization (8-byte aligned, 64-bit)
    alignas(8) uint64_t mbarrier;
};

// ============================================================================
// Kernel Parameters
// ============================================================================
struct VectorAddParams {
    int* a;
    int* b;
    int* c;
    int n;
    long long sleep_cycles;
    int grid_dim_x;
    int grid_dim_y;
    int grid_dim_z;
};

struct VectorMulParams {
    int* a;
    int* b;
    int* c;
    int n;
    long long sleep_cycles;
    int grid_dim_x;
    int grid_dim_y;
    int grid_dim_z;
};

// ============================================================================
// Device Helper Functions
// ============================================================================

// Busy-wait sleep
__device__ __forceinline__ void device_sleep(long long cycles) {
    long long start = clock64();
    while (clock64() - start < cycles) {
        // Busy wait
    }
}

// ============================================================================
// Kernel: Vector Add (runs on Default Context, LOW priority, cluster 4x1x1)
// Uses CLC to query next work for persistent execution
// ============================================================================
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
vector_add_kernel(__grid_constant__ const VectorAddParams params) {
    // Shared memory for CLC operations
    __shared__ CLCSharedMemory clc_smem;
    extern __shared__ char dynamic_smem[];
    
    const uint32_t thread_rank_in_cta = threadIdx.x;
    const uint32_t thread_cta_dim_x = blockDim.x;
    
    // Current work coordinates - initialized with blockIdx (first job)
    uint32_t curr_cta_id_in_grid_x = blockIdx.x;
    uint32_t curr_cta_id_in_grid_y = blockIdx.y;
    uint32_t curr_cta_id_in_grid_z = blockIdx.z;
    
    // Query CTA ID in cluster (for cluster 4x1x1)
    uint32_t cta_id_in_cluster_x, cta_id_in_cluster_y, cta_id_in_cluster_z;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(cta_id_in_cluster_x) : );
    asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(cta_id_in_cluster_y) : );
    asm volatile("mov.u32 %0, %%cluster_ctaid.z;\n" : "=r"(cta_id_in_cluster_z) : );
    
    // Query cta rank in cluster
    uint32_t cta_rank_in_cluster;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(cta_rank_in_cluster) : );
    
    // Initialize mbarrier (thread 0 only). Visibility: cluster (barrier.cluster only, no __syncthreads).
    if (thread_rank_in_cta == 0) {
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&clc_smem.mbarrier));
        uint32_t expected_arrive_count = 1;
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            :
            : "r"(mbar_addr), "r"(expected_arrive_count)
            : "memory"
        );
    }
    // Cluster scope sync (makes mbarrier init visible to all CTAs in cluster; per mbarrier.md no __syncthreads needed)
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" : : );
    asm volatile("barrier.cluster.wait.acquire.aligned;" : : );
    
    // Persistent loop with CLC - exits when no more work available
    uint32_t mbar_phase = 0;
    bool more_work = true;
    
    while (more_work) {
        // === PROCESS CURRENT WORK ===
        // Calculate global element index for array access
        // Note: This is a simplified 1D indexing for this example
        int element_idx = curr_cta_id_in_grid_x * thread_cta_dim_x + thread_rank_in_cta;
        
        // Perform vector addition
        if (element_idx < params.n) {
            params.c[element_idx] = params.a[element_idx] + params.b[element_idx];
        }
        
        // Sleep to simulate work
        device_sleep(params.sleep_cycles);
        
        __syncthreads();
        
        // === QUERY FOR NEXT WORK USING CLC ===
        if (thread_rank_in_cta == 0) {
            uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&clc_smem.mbarrier));
            uint32_t clc_response_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&clc_smem.clc_response));
            uint32_t tx_bytes = 16;
            
            // Set up mbarrier to expect 16 bytes transaction and arrive
            uint64_t state;
            asm volatile(
                "{\n\t"
                "  .reg .b64 state;\n\t"
                "  mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 state, [%1], %2;\n\t"
                "  mov.b64 %0, state;\n\t"
                "}"
                : "=l"(state)
                : "r"(mbar_addr), "r"(tx_bytes)
                : "memory"
            );
            
            // Only CTA rank 0 issues clc.try_cancel with multicast
            if (cta_rank_in_cluster == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
                    :
                    : "r"(clc_response_addr), "r"(mbar_addr)
                    : "memory"
                );
            }
        }
        
        // All threads wait for CLC response
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&clc_smem.mbarrier));
        uint32_t parity = mbar_phase;
        int complete = 0;
        while (!complete) {
            asm volatile(
                "{\n\t"
                "  .reg .pred P1;\n\t"
                "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2;\n\t"
                "  selp.u32 %0, 1, 0, P1;\n\t"
                "}"
                : "=r"(complete)
                : "r"(mbar_addr), "r"(parity)
                : "memory"
            );
        }
        
        mbar_phase ^= 1;
        
        // Query CLC response
        uint32_t is_canceled_result;
        asm volatile(
            "{\n\t"
            "  .reg .b128 handle;\n\t"
            "  .reg .pred p;\n\t"
            "  mov.b128 handle, {%1, %2, %3, %4};\n\t"
            "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;\n\t"
            "  selp.u32 %0, 1, 0, p;\n\t"
            "}"
            : "=r"(is_canceled_result)
            : "r"(clc_smem.clc_response.x), "r"(clc_smem.clc_response.y),
              "r"(clc_smem.clc_response.z), "r"(clc_smem.clc_response.w)
        );
        
        if (is_canceled_result != 0) {
            // Got new work - query coordinates of cluster's first CTA (cta rank 0 in cluster)
            // clustercta0_id_in_grid: The grid-level CTA ID of the first CTA in the canceled cluster
            uint32_t clustercta0_id_in_grid_x, clustercta0_id_in_grid_y, clustercta0_id_in_grid_z;
            uint32_t dummy;
            asm volatile(
                "{\n\t"
                "  .reg .b128 handle;\n\t"
                "  mov.b128 handle, {%4, %5, %6, %7};\n\t"
                "  clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, %3}, handle;\n\t"
                "}"
                : "=r"(clustercta0_id_in_grid_x), "=r"(clustercta0_id_in_grid_y), "=r"(clustercta0_id_in_grid_z), "=r"(dummy)
                : "r"(clc_smem.clc_response.x), "r"(clc_smem.clc_response.y),
                  "r"(clc_smem.clc_response.z), "r"(clc_smem.clc_response.w)
            );
            
            // Update current work coordinates: add cluster-local offset to get this CTA's grid position
            curr_cta_id_in_grid_x = clustercta0_id_in_grid_x + cta_id_in_cluster_x;
            curr_cta_id_in_grid_y = clustercta0_id_in_grid_y + cta_id_in_cluster_y;
            curr_cta_id_in_grid_z = clustercta0_id_in_grid_z + cta_id_in_cluster_z;
        } else {
            more_work = false;
        }
        
        __syncthreads();
    }
}

// ============================================================================
// Kernel: Vector Multiply (runs on Remaining Partition, HIGH priority, cluster 2x1x1)
// Uses CLC to query next work for persistent execution
// ============================================================================
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 1)
vector_mul_kernel(__grid_constant__ const VectorMulParams params) {
    // Shared memory for CLC operations
    __shared__ CLCSharedMemory clc_smem;
    extern __shared__ char dynamic_smem[];
    
    const uint32_t thread_rank_in_cta = threadIdx.x;
    const uint32_t thread_cta_dim_x = blockDim.x;
    
    // Current work coordinates - initialized with blockIdx (first job)
    uint32_t curr_cta_id_in_grid_x = blockIdx.x;
    uint32_t curr_cta_id_in_grid_y = blockIdx.y;
    uint32_t curr_cta_id_in_grid_z = blockIdx.z;
    
    // Query CTA ID in cluster (for cluster 2x1x1)
    uint32_t cta_id_in_cluster_x, cta_id_in_cluster_y, cta_id_in_cluster_z;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;\n" : "=r"(cta_id_in_cluster_x) : );
    asm volatile("mov.u32 %0, %%cluster_ctaid.y;\n" : "=r"(cta_id_in_cluster_y) : );
    asm volatile("mov.u32 %0, %%cluster_ctaid.z;\n" : "=r"(cta_id_in_cluster_z) : );
    
    // Query cta rank in cluster
    uint32_t cta_rank_in_cluster;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(cta_rank_in_cluster) : );
    
    // Initialize mbarrier (thread 0 only). Visibility: cluster (barrier.cluster only, no __syncthreads).
    if (thread_rank_in_cta == 0) {
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&clc_smem.mbarrier));
        uint32_t expected_arrive_count = 1;
        asm volatile(
            "mbarrier.init.shared::cta.b64 [%0], %1;"
            :
            : "r"(mbar_addr), "r"(expected_arrive_count)
            : "memory"
        );
    }
    // Cluster scope sync (makes mbarrier init visible to all CTAs in cluster; per mbarrier.md no __syncthreads needed)
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" : : );
    asm volatile("barrier.cluster.wait.acquire.aligned;" : : );
    
    // Persistent loop with CLC - exits when no more work available
    uint32_t mbar_phase = 0;
    bool more_work = true;
    
    while (more_work) {
        // === PROCESS CURRENT WORK ===
        // Calculate global element index for array access
        // Note: This is a simplified 1D indexing for this example
        int element_idx = curr_cta_id_in_grid_x * thread_cta_dim_x + thread_rank_in_cta;
        
        // Perform vector multiplication
        if (element_idx < params.n) {
            params.c[element_idx] = params.a[element_idx] * params.b[element_idx];
        }
        
        // Sleep to simulate work
        device_sleep(params.sleep_cycles);
        
        __syncthreads();
        
        // === QUERY FOR NEXT WORK USING CLC ===
        if (thread_rank_in_cta == 0) {
            uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&clc_smem.mbarrier));
            uint32_t clc_response_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&clc_smem.clc_response));
            uint32_t tx_bytes = 16;
            
            // Set up mbarrier to expect 16 bytes transaction and arrive
            uint64_t state;
            asm volatile(
                "{\n\t"
                "  .reg .b64 state;\n\t"
                "  mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 state, [%1], %2;\n\t"
                "  mov.b64 %0, state;\n\t"
                "}"
                : "=l"(state)
                : "r"(mbar_addr), "r"(tx_bytes)
                : "memory"
            );
            
            // Only CTA rank 0 issues clc.try_cancel with multicast
            if (cta_rank_in_cluster == 0) {
                asm volatile(
                    "clusterlaunchcontrol.try_cancel.async.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [%0], [%1];"
                    :
                    : "r"(clc_response_addr), "r"(mbar_addr)
                    : "memory"
                );
            }
        }
        
        // All threads wait for CLC response
        uint32_t mbar_addr = static_cast<uint32_t>(__cvta_generic_to_shared(&clc_smem.mbarrier));
        uint32_t parity = mbar_phase;
        int complete = 0;
        while (!complete) {
            asm volatile(
                "{\n\t"
                "  .reg .pred P1;\n\t"
                "  mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%1], %2;\n\t"
                "  selp.u32 %0, 1, 0, P1;\n\t"
                "}"
                : "=r"(complete)
                : "r"(mbar_addr), "r"(parity)
                : "memory"
            );
        }
        
        mbar_phase ^= 1;
        
        // Query CLC response
        uint32_t is_canceled_result;
        asm volatile(
            "{\n\t"
            "  .reg .b128 handle;\n\t"
            "  .reg .pred p;\n\t"
            "  mov.b128 handle, {%1, %2, %3, %4};\n\t"
            "  clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p, handle;\n\t"
            "  selp.u32 %0, 1, 0, p;\n\t"
            "}"
            : "=r"(is_canceled_result)
            : "r"(clc_smem.clc_response.x), "r"(clc_smem.clc_response.y),
              "r"(clc_smem.clc_response.z), "r"(clc_smem.clc_response.w)
        );
        
        if (is_canceled_result != 0) {
            // Got new work - query coordinates of cluster's first CTA (cta rank 0 in cluster)
            // clustercta0_id_in_grid: The grid-level CTA ID of the first CTA in the canceled cluster
            uint32_t clustercta0_id_in_grid_x, clustercta0_id_in_grid_y, clustercta0_id_in_grid_z;
            uint32_t dummy;
            asm volatile(
                "{\n\t"
                "  .reg .b128 handle;\n\t"
                "  mov.b128 handle, {%4, %5, %6, %7};\n\t"
                "  clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, %3}, handle;\n\t"
                "}"
                : "=r"(clustercta0_id_in_grid_x), "=r"(clustercta0_id_in_grid_y), "=r"(clustercta0_id_in_grid_z), "=r"(dummy)
                : "r"(clc_smem.clc_response.x), "r"(clc_smem.clc_response.y),
                  "r"(clc_smem.clc_response.z), "r"(clc_smem.clc_response.w)
            );
            
            // Update current work coordinates: add cluster-local offset to get this CTA's grid position
            curr_cta_id_in_grid_x = clustercta0_id_in_grid_x + cta_id_in_cluster_x;
            curr_cta_id_in_grid_y = clustercta0_id_in_grid_y + cta_id_in_cluster_y;
            curr_cta_id_in_grid_z = clustercta0_id_in_grid_z + cta_id_in_cluster_z;
        } else {
            more_work = false;
        }
        
        __syncthreads();
    }
}

// ============================================================================
// Dummy Delay Kernel (used with --delay_high_priority flag)
// Simple kernel that just sleeps, used to delay high priority kernel launch
// ============================================================================
__global__ void delay_kernel(long long sleep_cycles) {
    long long start = clock64();
    while (clock64() - start < sleep_cycles) {
        // Busy wait
    }
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    // ========================================================================
    // Parse Command-Line Arguments
    // ========================================================================
    bool use_cuda_graph = false;
    bool delay_high_priority = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cuda_graph") == 0) {
            use_cuda_graph = true;
        } else if (strcmp(argv[i], "--delay_high_priority") == 0) {
            delay_high_priority = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [OPTIONS]\n", argv[0]);
            printf("Options:\n");
            printf("  --cuda_graph           Use CUDA Graph for kernel launch (default: direct launch)\n");
            printf("  --delay_high_priority  Add 0.01s delay before high priority kernel\n");
            printf("  --help, -h             Show this help message\n");
            return 0;
        }
    }
    
    printf("=== Green Context Sample: Fork-Join with CLC ===\n");
    printf("Mode: %s\n", use_cuda_graph ? "CUDA Graph Launch" : "Direct Kernel Launch");
    printf("High priority delay: %s\n\n", delay_high_priority ? "Yes (0.01s)" : "No");
    
    // ========================================================================
    // CUDA Driver API Initialization
    // ========================================================================
    CUDA_DRIVER_CHECK(cuInit(0));
    
    int device_id = 0;
    CUdevice device;
    CUcontext primary_context;
    
    CUDA_DRIVER_CHECK(cuDeviceGet(&device, device_id));
    CUDA_DRIVER_CHECK(cuDevicePrimaryCtxRetain(&primary_context, device));
    CUDA_DRIVER_CHECK(cuCtxSetCurrent(primary_context));
    
    // Print device name
    char device_name[256];
    CUDA_DRIVER_CHECK(cuDeviceGetName(device_name, sizeof(device_name), device));
    printf("Device: %s\n", device_name);
    
    // ========================================================================
    // Query Device Properties
    // ========================================================================
    cudaDeviceProp prop;
    CUDA_RUNTIME_CHECK(cudaGetDeviceProperties(&prop, device_id));
    
    // Query clock rate via cudaDeviceGetAttribute (clockRate deprecated in cudaDeviceProp)
    int clock_rate_khz = 0;
    CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(&clock_rate_khz, cudaDevAttrClockRate, device_id));
    long long cycles_per_second = (long long)clock_rate_khz * 1000;
    
    printf("Clock Rate: %d kHz (%lld cycles/sec)\n", clock_rate_khz, cycles_per_second);
    printf("Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
    printf("SM Count: %d\n\n", prop.multiProcessorCount);
    
    // Calculate sleep cycles
    long long default_ctx_sleep_cycles = (long long)(VEC_ADD_SLEEP_SECONDS * cycles_per_second);
    long long remaining_sleep_cycles = (long long)(VEC_MUL_SLEEP_SECONDS * cycles_per_second);
    long long delay_sleep_cycles = (long long)(0.01f * cycles_per_second);  // 0.01s delay for --delay_high_priority
    
    printf("Default context sleep: %.1fs = %lld cycles\n", VEC_ADD_SLEEP_SECONDS, default_ctx_sleep_cycles);
    printf("Remaining partition sleep: %.1fs = %lld cycles\n", VEC_MUL_SLEEP_SECONDS, remaining_sleep_cycles);
    if (delay_high_priority) {
        printf("High priority delay: 0.01s = %lld cycles\n", delay_sleep_cycles);
    }
    printf("\n");
    
    // ========================================================================
    // Query SM Resources
    // ========================================================================
    CUdevResource device_resource;
    CUDA_DRIVER_CHECK(cuDeviceGetDevResource(device, &device_resource, CU_DEV_RESOURCE_TYPE_SM));
    printf("Device SM Count: %d\n", device_resource.sm.smCount);
    
    int total_sm_needed = PRIMARY_PARTITION_SM_COUNT + REMAINING_PARTITION_SM_COUNT;
    if (total_sm_needed > device_resource.sm.smCount) {
        printf("ERROR: Need %d SMs but device only has %d\n", 
               total_sm_needed, device_resource.sm.smCount);
        return 1;
    }
    
    // ========================================================================
    // Partition SM Resources
    // ========================================================================
    CUdevResource primary_partition_resource;
    CUdevResource remaining_partition_resource;
    unsigned int num_groups = 1;
    // **IMPORTANT** Use 0 for compute commute overlap
    const unsigned int partition_flag = 0;
    
    CUDA_DRIVER_CHECK(cuDevSmResourceSplitByCount(
        &primary_partition_resource,
        &num_groups,
        &device_resource,
        &remaining_partition_resource,
        partition_flag,
        PRIMARY_PARTITION_SM_COUNT));
    
    printf("Primary Partition SM Count: %d (reserved, no kernel launch)\n", 
           primary_partition_resource.sm.smCount);
    printf("Remaining Partition SM Count: %d (for Vector Mul kernel)\n\n",
           remaining_partition_resource.sm.smCount);
    
    // ========================================================================
    // Create Green Contexts for Both Partitions
    // Why create both: Even though primary partition is not used for kernel launch,
    // creating the green context ensures proper resource isolation and compatibility
    // with systems that may require all partitions to have associated contexts.
    // ========================================================================
    
    // Primary partition green context (128 SMs - reserved, no kernel launch)
    CUdevResourceDesc primary_partition_desc;
    CUDA_DRIVER_CHECK(cuDevResourceGenerateDesc(&primary_partition_desc, &primary_partition_resource, 1));
    
    CUgreenCtx primary_green_ctx;
    CUDA_DRIVER_CHECK(cuGreenCtxCreate(&primary_green_ctx, primary_partition_desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
    
    printf("Green context created for primary partition (%d SMs - reserved, not used)\n",
           primary_partition_resource.sm.smCount);
    
    // Remaining partition green context (20 SMs - for Vector Mul kernel)
    CUdevResourceDesc remaining_partition_desc;
    CUDA_DRIVER_CHECK(cuDevResourceGenerateDesc(&remaining_partition_desc, &remaining_partition_resource, 1));
    
    CUgreenCtx remaining_green_ctx;
    CUDA_DRIVER_CHECK(cuGreenCtxCreate(&remaining_green_ctx, remaining_partition_desc, device, CU_GREEN_CTX_DEFAULT_STREAM));
    
    printf("Green context created for remaining partition (%d SMs - for Vector Mul)\n",
           remaining_partition_resource.sm.smCount);
    
    // ========================================================================
    // Create Streams with Priorities
    // ========================================================================
    int priority_low, priority_high;
    CUDA_RUNTIME_CHECK(cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    printf("Stream priority range: [%d (highest), %d (lowest)]\n", priority_high, priority_low);
    
    // Default context: LOW priority stream (for Vector Add kernel)
    CUstream default_ctx_low_stream;
    CUDA_DRIVER_CHECK(cuStreamCreateWithPriority(&default_ctx_low_stream, 
                                                  CU_STREAM_NON_BLOCKING, priority_low));
    
    // Remaining partition: HIGH priority stream (for Vector Mul kernel)
    CUstream remaining_stream;
    CUDA_DRIVER_CHECK(cuGreenCtxStreamCreate(&remaining_stream, remaining_green_ctx, 
                                              CU_STREAM_NON_BLOCKING, priority_high));
    
    // Sync stream (for fork/join coordination)
    CUstream sync_stream;
    CUDA_DRIVER_CHECK(cuStreamCreate(&sync_stream, CU_STREAM_NON_BLOCKING));
    
    printf("Streams created:\n");
    printf("  - Default context LOW priority stream (for Vector Add)\n");
    printf("  - Remaining partition HIGH priority stream (for Vector Mul)\n");
    printf("  - Sync stream (for fork/join)\n\n");
    
    // ========================================================================
    // Allocate Memory
    // ========================================================================
    int* d_a = nullptr;
    int* d_b = nullptr;
    int* d_c_add = nullptr;
    int* d_c_mul = nullptr;
    
    size_t vector_bytes = VECTOR_SIZE * sizeof(int);
    
    CUDA_RUNTIME_CHECK(cudaMalloc(&d_a, vector_bytes));
    CUDA_RUNTIME_CHECK(cudaMalloc(&d_b, vector_bytes));
    CUDA_RUNTIME_CHECK(cudaMalloc(&d_c_add, vector_bytes));
    CUDA_RUNTIME_CHECK(cudaMalloc(&d_c_mul, vector_bytes));
    
    // Initialize with zeros
    CUDA_RUNTIME_CHECK(cudaMemset(d_a, 0, vector_bytes));
    CUDA_RUNTIME_CHECK(cudaMemset(d_b, 0, vector_bytes));
    
    printf("Memory allocated: %zu bytes per vector\n\n", vector_bytes);
    
    // ========================================================================
    // Setup Kernel Attributes for Occupancy = 1
    // ========================================================================
    size_t max_dynamic_smem = prop.sharedMemPerMultiprocessor - 4096;
    
    const void* vector_add_func = (const void*)vector_add_kernel;
    const void* vector_mul_func = (const void*)vector_mul_kernel;
    
    // Vector Add kernel attributes (cluster 4x1x1)
    CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(vector_add_func,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(vector_add_func,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem));
    
    // Vector Mul kernel attributes (cluster 2x1x1)
    CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(vector_mul_func,
        cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
    CUDA_RUNTIME_CHECK(cudaFuncSetAttribute(vector_mul_func,
        cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem));
    
    // Verify occupancy = 1 (per kernel_launch.md)
    int blocks_per_sm_add = 0;
    int blocks_per_sm_mul = 0;
    CUDA_RUNTIME_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm_add, vector_add_func, THREADS_PER_BLOCK, max_dynamic_smem));
    CUDA_RUNTIME_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm_mul, vector_mul_func, THREADS_PER_BLOCK, max_dynamic_smem));
    printf("Kernel attributes set for occupancy=1 (dynamic smem=%zu)\n", max_dynamic_smem);
    printf("Occupancy verification: Vector Add %d blocks/SM, Vector Mul %d blocks/SM (expected 1)\n",
           blocks_per_sm_add, blocks_per_sm_mul);
    printf("CLC persistent kernel mode enabled\n\n");
    
    // ========================================================================
    // Create Events for Fork-Join Synchronization
    // ========================================================================
    CUevent fork_event;
    CUevent default_ctx_done_event;
    CUevent remaining_done_event;
    
    CUDA_DRIVER_CHECK(cuEventCreate(&fork_event, CU_EVENT_DEFAULT));
    CUDA_DRIVER_CHECK(cuEventCreate(&default_ctx_done_event, CU_EVENT_DEFAULT));
    CUDA_DRIVER_CHECK(cuEventCreate(&remaining_done_event, CU_EVENT_DEFAULT));
    
    printf("Events created for fork-join synchronization\n\n");
    
    // ========================================================================
    // Setup Kernel Parameters
    // ========================================================================
    VectorAddParams add_params = {
        .a = d_a,
        .b = d_b,
        .c = d_c_add,
        .n = VECTOR_SIZE,
        .sleep_cycles = default_ctx_sleep_cycles,
        .grid_dim_x = DEFAULT_CTX_GRID_SIZE,
        .grid_dim_y = 1,
        .grid_dim_z = 1
    };
    
    VectorMulParams mul_params = {
        .a = d_a,
        .b = d_b,
        .c = d_c_mul,
        .n = VECTOR_SIZE,
        .sleep_cycles = remaining_sleep_cycles,
        .grid_dim_x = REMAINING_GRID_SIZE,
        .grid_dim_y = 1,
        .grid_dim_z = 1
    };
    
    void* add_kernel_args[] = {&add_params};
    void* mul_kernel_args[] = {&mul_params};
    
    // ========================================================================
    // Setup Launch Configurations
    // ========================================================================
    // Default context: 512 blocks, cluster 4x1x1, LOW priority
    cudaLaunchConfig_t default_ctx_config = {};
    default_ctx_config.gridDim = dim3(DEFAULT_CTX_GRID_SIZE, 1, 1);  // 512 blocks
    default_ctx_config.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
    default_ctx_config.dynamicSmemBytes = max_dynamic_smem;
    default_ctx_config.stream = (cudaStream_t)default_ctx_low_stream;
    
    cudaLaunchAttribute default_ctx_attrs[2];
    default_ctx_attrs[0].id = cudaLaunchAttributeClusterDimension;
    default_ctx_attrs[0].val.clusterDim.x = 4;
    default_ctx_attrs[0].val.clusterDim.y = 1;
    default_ctx_attrs[0].val.clusterDim.z = 1;
    
    // Enable PDL for CLC
    default_ctx_attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    default_ctx_attrs[1].val.programmaticStreamSerializationAllowed = 1;
    
    default_ctx_config.attrs = default_ctx_attrs;
    default_ctx_config.numAttrs = 2;
    
    // Remaining partition: 40 blocks, cluster 2x1x1, HIGH priority
    cudaLaunchConfig_t remaining_config = {};
    remaining_config.gridDim = dim3(REMAINING_GRID_SIZE, 1, 1);  // 40 blocks
    remaining_config.blockDim = dim3(THREADS_PER_BLOCK, 1, 1);
    remaining_config.dynamicSmemBytes = max_dynamic_smem;
    remaining_config.stream = (cudaStream_t)remaining_stream;
    
    cudaLaunchAttribute remaining_attrs[2];
    remaining_attrs[0].id = cudaLaunchAttributeClusterDimension;
    remaining_attrs[0].val.clusterDim.x = 2;
    remaining_attrs[0].val.clusterDim.y = 1;
    remaining_attrs[0].val.clusterDim.z = 1;
    
    // Enable PDL for CLC
    remaining_attrs[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    remaining_attrs[1].val.programmaticStreamSerializationAllowed = 1;
    
    remaining_config.attrs = remaining_attrs;
    remaining_config.numAttrs = 2;
    
    printf("Launch configurations:\n");
    printf("  Default context: %d blocks, cluster 4x1x1, LOW priority=%d (Vector Add)\n",
           DEFAULT_CTX_GRID_SIZE, priority_low);
    printf("  Remaining partition: %d blocks, cluster 2x1x1, HIGH priority=%d (Vector Mul)\n\n",
           REMAINING_GRID_SIZE, priority_high);
    
    // ========================================================================
    // Timing Events (used for both modes)
    // ========================================================================
    cudaEvent_t start_event, end_event;
    CUDA_RUNTIME_CHECK(cudaEventCreate(&start_event));
    CUDA_RUNTIME_CHECK(cudaEventCreate(&end_event));
    
    // Graph objects (only used in CUDA graph mode)
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    
    if (use_cuda_graph) {
        // ========================================================================
        // CUDA Graph Capture Mode
        // Note: Use Runtime API (cudaEventRecord, cudaStreamWaitEvent) for graph capture
        // to ensure proper stream capture state tracking. Driver API event calls may not
        // integrate properly with Runtime API stream capture.
        //
        // Known Limitation: When using Driver API streams (CUstream from cuStreamCreate,
        // cuGreenCtxStreamCreate) with Runtime API graph capture, event operations may
        // not appear as explicit EventRecord/WaitEvent nodes in the captured graph.
        // The kernels are still captured and dependencies are preserved through graph
        // edges, but the visual representation shows only kernel nodes.
        // ========================================================================
        printf("Starting CUDA Graph capture...\n");
        
        // Begin capture on sync stream
        CUDA_RUNTIME_CHECK(cudaStreamBeginCapture(
            (cudaStream_t)sync_stream, cudaStreamCaptureModeGlobal));
        
        // --- FORK: Record event on sync stream ---
        // Use Runtime API for proper graph capture integration
        CUDA_RUNTIME_CHECK(cudaEventRecord((cudaEvent_t)fork_event, (cudaStream_t)sync_stream));
        
        // --- Default context (LOW priority) waits for fork event, then launches Vector Add ---
        CUDA_RUNTIME_CHECK(cudaStreamWaitEvent((cudaStream_t)default_ctx_low_stream, (cudaEvent_t)fork_event, 0));
        CUDA_RUNTIME_CHECK(cudaLaunchKernelExC(&default_ctx_config, vector_add_func, add_kernel_args));
        CUDA_RUNTIME_CHECK(cudaEventRecord((cudaEvent_t)default_ctx_done_event, (cudaStream_t)default_ctx_low_stream));
        
        // --- Remaining partition (HIGH priority) waits for fork event, then launches Vector Mul ---
        CUDA_RUNTIME_CHECK(cudaStreamWaitEvent((cudaStream_t)remaining_stream, (cudaEvent_t)fork_event, 0));
        // Optional delay kernel before high priority kernel (--delay_high_priority flag)
        if (delay_high_priority) {
            delay_kernel<<<1, 1, 0, (cudaStream_t)remaining_stream>>>(delay_sleep_cycles);
        }
        CUDA_RUNTIME_CHECK(cudaLaunchKernelExC(&remaining_config, vector_mul_func, mul_kernel_args));
        CUDA_RUNTIME_CHECK(cudaEventRecord((cudaEvent_t)remaining_done_event, (cudaStream_t)remaining_stream));
        
        // --- JOIN: Sync stream waits for both to complete ---
        CUDA_RUNTIME_CHECK(cudaStreamWaitEvent((cudaStream_t)sync_stream, (cudaEvent_t)default_ctx_done_event, 0));
        CUDA_RUNTIME_CHECK(cudaStreamWaitEvent((cudaStream_t)sync_stream, (cudaEvent_t)remaining_done_event, 0));
        
        // End capture
        CUDA_RUNTIME_CHECK(cudaStreamEndCapture((cudaStream_t)sync_stream, &graph));
        
        printf("CUDA Graph captured successfully!\n");
        
        // ========================================================================
        // Print Graph Node Information
        // Why: Debugging and visualization to understand the graph structure,
        // verify correct capture of fork-join pattern with events and kernels.
        // ========================================================================
        size_t num_nodes = 0;
        CUDA_RUNTIME_CHECK(cudaGraphGetNodes(graph, nullptr, &num_nodes));
        printf("\n=== CUDA Graph Structure ===\n");
        printf("Total nodes in graph: %zu\n\n", num_nodes);
        
        if (num_nodes > 0) {
            cudaGraphNode_t* nodes = new cudaGraphNode_t[num_nodes];
            CUDA_RUNTIME_CHECK(cudaGraphGetNodes(graph, nodes, &num_nodes));
            
            // Count node types
            int kernel_nodes = 0, event_record_nodes = 0, event_wait_nodes = 0;
            int host_nodes = 0, memcpy_nodes = 0, memset_nodes = 0, other_nodes = 0;
            
            for (size_t i = 0; i < num_nodes; i++) {
                cudaGraphNodeType type;
                CUDA_RUNTIME_CHECK(cudaGraphNodeGetType(nodes[i], &type));
                
                const char* type_str;
                switch (type) {
                    case cudaGraphNodeTypeKernel:
                        type_str = "KERNEL";
                        kernel_nodes++;
                        break;
                    case cudaGraphNodeTypeMemcpy:
                        type_str = "MEMCPY";
                        memcpy_nodes++;
                        break;
                    case cudaGraphNodeTypeMemset:
                        type_str = "MEMSET";
                        memset_nodes++;
                        break;
                    case cudaGraphNodeTypeHost:
                        type_str = "HOST";
                        host_nodes++;
                        break;
                    case cudaGraphNodeTypeEventRecord:
                        type_str = "EVENT_RECORD";
                        event_record_nodes++;
                        break;
                    case cudaGraphNodeTypeWaitEvent:
                        type_str = "WAIT_EVENT";
                        event_wait_nodes++;
                        break;
                    default:
                        type_str = "OTHER";
                        other_nodes++;
                        break;
                }
                printf("  Node[%zu]: %s\n", i, type_str);
            }
            
            printf("\nNode Summary:\n");
            printf("  Kernel nodes:       %d (Vector Add + Vector Mul)\n", kernel_nodes);
            printf("  Event Record nodes: %d (fork_event + 2 done_events)\n", event_record_nodes);
            printf("  Event Wait nodes:   %d (2 fork waits + 2 join waits)\n", event_wait_nodes);
            if (memcpy_nodes > 0) printf("  Memcpy nodes:       %d\n", memcpy_nodes);
            if (memset_nodes > 0) printf("  Memset nodes:       %d\n", memset_nodes);
            if (host_nodes > 0) printf("  Host nodes:         %d\n", host_nodes);
            if (other_nodes > 0) printf("  Other nodes:        %d\n", other_nodes);
            
            delete[] nodes;
        }
        
        // ========================================================================
        // Export Graph to DOT File for Visualization
        // Why: DOT format can be converted to image using graphviz (dot -Tpng)
        // Usage: dot -Tpng green_context_graph.dot -o green_context_graph.png
        // ========================================================================
        const char* dot_filename = "green_context_graph.dot";
        CUDA_RUNTIME_CHECK(cudaGraphDebugDotPrint(graph, dot_filename, 0));
        printf("\nGraph exported to: %s\n", dot_filename);
        printf("To generate image: dot -Tpng %s -o green_context_graph.png\n", dot_filename);
        printf("=== End Graph Structure ===\n\n");
        
        // Instantiate graph (using non-deprecated API)
        CUDA_RUNTIME_CHECK(cudaGraphInstantiateWithFlags(&graph_exec, graph, 0));
        
        printf("CUDA Graph instantiated successfully!\n\n");
        
        // ========================================================================
        // Execute Graph for NUM_ITERATIONS
        // ========================================================================
        printf("Launching CUDA Graph for %d iterations...\n", NUM_ITERATIONS);
        printf("Expected time per iteration: ~%.1f seconds (max of %.1fs, %.1fs)\n\n",
               (VEC_ADD_SLEEP_SECONDS > VEC_MUL_SLEEP_SECONDS) ? VEC_ADD_SLEEP_SECONDS : VEC_MUL_SLEEP_SECONDS,
               VEC_ADD_SLEEP_SECONDS, VEC_MUL_SLEEP_SECONDS);
        
        CUDA_RUNTIME_CHECK(cudaEventRecord(start_event, (cudaStream_t)sync_stream));
        
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            CUDA_RUNTIME_CHECK(cudaGraphLaunch(graph_exec, (cudaStream_t)sync_stream));
            
            if ((iter + 1) % 10 == 0 || iter == 0) {
                CUDA_RUNTIME_CHECK(cudaStreamSynchronize((cudaStream_t)sync_stream));
                printf("  Completed iteration %d/%d\n", iter + 1, NUM_ITERATIONS);
            }
        }
        
        CUDA_RUNTIME_CHECK(cudaEventRecord(end_event, (cudaStream_t)sync_stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize((cudaStream_t)sync_stream));
    } else {
        // ========================================================================
        // Direct Kernel Launch Mode (No CUDA Graph)
        // Launches kernels directly using fork-join pattern with events
        // ========================================================================
        printf("Launching kernels directly for %d iterations...\n", NUM_ITERATIONS);
        printf("Expected time per iteration: ~%.1f seconds (max of %.1fs, %.1fs)\n\n",
               (VEC_ADD_SLEEP_SECONDS > VEC_MUL_SLEEP_SECONDS) ? VEC_ADD_SLEEP_SECONDS : VEC_MUL_SLEEP_SECONDS,
               VEC_ADD_SLEEP_SECONDS, VEC_MUL_SLEEP_SECONDS);
        
        CUDA_RUNTIME_CHECK(cudaEventRecord(start_event, (cudaStream_t)sync_stream));
        
        for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
            // --- FORK: Record event on sync stream ---
            CUDA_DRIVER_CHECK(cuEventRecord(fork_event, sync_stream));
            
            // --- Default context (LOW priority) waits for fork event, then launches Vector Add ---
            CUDA_DRIVER_CHECK(cuStreamWaitEvent(default_ctx_low_stream, fork_event, 0));
            CUDA_RUNTIME_CHECK(cudaLaunchKernelExC(&default_ctx_config, vector_add_func, add_kernel_args));
            CUDA_DRIVER_CHECK(cuEventRecord(default_ctx_done_event, default_ctx_low_stream));
            
            // --- Remaining partition (HIGH priority) waits for fork event, then launches Vector Mul ---
            CUDA_DRIVER_CHECK(cuStreamWaitEvent(remaining_stream, fork_event, 0));
            // Optional delay kernel before high priority kernel (--delay_high_priority flag)
            if (delay_high_priority) {
                delay_kernel<<<1, 1, 0, (cudaStream_t)remaining_stream>>>(delay_sleep_cycles);
            }
            CUDA_RUNTIME_CHECK(cudaLaunchKernelExC(&remaining_config, vector_mul_func, mul_kernel_args));
            CUDA_DRIVER_CHECK(cuEventRecord(remaining_done_event, remaining_stream));
            
            // --- JOIN: Sync stream waits for both to complete ---
            CUDA_DRIVER_CHECK(cuStreamWaitEvent(sync_stream, default_ctx_done_event, 0));
            CUDA_DRIVER_CHECK(cuStreamWaitEvent(sync_stream, remaining_done_event, 0));
            
            if ((iter + 1) % 10 == 0 || iter == 0) {
                CUDA_RUNTIME_CHECK(cudaStreamSynchronize((cudaStream_t)sync_stream));
                printf("  Completed iteration %d/%d\n", iter + 1, NUM_ITERATIONS);
            }
        }
        
        CUDA_RUNTIME_CHECK(cudaEventRecord(end_event, (cudaStream_t)sync_stream));
        CUDA_RUNTIME_CHECK(cudaStreamSynchronize((cudaStream_t)sync_stream));
    }
    
    float elapsed_ms = 0;
    CUDA_RUNTIME_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, end_event));
    
    printf("\n=== Results ===\n");
    printf("Total time: %.2f seconds\n", elapsed_ms / 1000.0f);
    printf("Average time per iteration: %.2f seconds\n", elapsed_ms / 1000.0f / NUM_ITERATIONS);
    printf("Expected per iteration: ~%.1f seconds\n\n", 
           (VEC_ADD_SLEEP_SECONDS > VEC_MUL_SLEEP_SECONDS) ? VEC_ADD_SLEEP_SECONDS : VEC_MUL_SLEEP_SECONDS);
    
    // ========================================================================
    // Cleanup
    // ========================================================================
    printf("Cleaning up...\n");
    
    // Destroy timing events
    CUDA_RUNTIME_CHECK(cudaEventDestroy(start_event));
    CUDA_RUNTIME_CHECK(cudaEventDestroy(end_event));
    
    // Destroy graph (only if CUDA graph mode was used)
    if (use_cuda_graph) {
        CUDA_RUNTIME_CHECK(cudaGraphExecDestroy(graph_exec));
        CUDA_RUNTIME_CHECK(cudaGraphDestroy(graph));
    }
    
    // Destroy synchronization events
    CUDA_DRIVER_CHECK(cuEventDestroy(fork_event));
    CUDA_DRIVER_CHECK(cuEventDestroy(default_ctx_done_event));
    CUDA_DRIVER_CHECK(cuEventDestroy(remaining_done_event));
    
    // Free memory
    CUDA_RUNTIME_CHECK(cudaFree(d_a));
    CUDA_RUNTIME_CHECK(cudaFree(d_b));
    CUDA_RUNTIME_CHECK(cudaFree(d_c_add));
    CUDA_RUNTIME_CHECK(cudaFree(d_c_mul));
    
    // Destroy streams
    CUDA_DRIVER_CHECK(cuStreamDestroy(default_ctx_low_stream));
    CUDA_DRIVER_CHECK(cuStreamDestroy(remaining_stream));
    CUDA_DRIVER_CHECK(cuStreamDestroy(sync_stream));
    
    // Destroy green contexts (both primary and remaining partitions)
    CUDA_DRIVER_CHECK(cuGreenCtxDestroy(primary_green_ctx));
    CUDA_DRIVER_CHECK(cuGreenCtxDestroy(remaining_green_ctx));
    
    // Release primary context
    CUDA_DRIVER_CHECK(cuDevicePrimaryCtxRelease(device));
    
    printf("Done!\n");
    
    return 0;
}
