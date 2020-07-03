#include "config.h"  // NOLINT
#include <cassert>
#ifdef USE_GPU
#include "cuda_runtime.h"  // NOLINT
#endif

int cfg::max_num_nodes = 1000000;
int cfg::bits_compress = 0;
int cfg::dim_embed = 0;
bool cfg::directed = false;
bool cfg::self_loop = false;
int cfg::gpu = -1;
bool cfg::bfs_permute = false;
int cfg::seed = 1;
std::default_random_engine cfg::generator;

void cfg::LoadParams(const int argc, const char** argv)
{
    for (int i = 1; i < argc; i += 2)
    {
        if (strcmp(argv[i], "-max_num_nodes") == 0)
            max_num_nodes = atoi(argv[i + 1]);  // NOLINT
        if (strcmp(argv[i], "-directed") == 0)
            directed = atoi(argv[i + 1]);  // NOLINT
        if (strcmp(argv[i], "-self_loop") == 0)
            self_loop = atoi(argv[i + 1]);  // NOLINT
        if (strcmp(argv[i], "-bits_compress") == 0)
            bits_compress = atoi(argv[i + 1]);  // NOLINT
        if (strcmp(argv[i], "-embed_dim") == 0)
            dim_embed = atoi(argv[i + 1]);  // NOLINT
        if (strcmp(argv[i], "-gpu") == 0)
            gpu = atoi(argv[i + 1]);  // NOLINT
        if (strcmp(argv[i], "-seed") == 0)
            seed = atoi(argv[i + 1]);  // NOLINT
        if (strcmp(argv[i], "-bfs_permute") == 0)
            bfs_permute = atoi(argv[i + 1]);  // NOLINT
    }
    std::cerr << "====== begin of tree_clib configuration ======" << std::endl;
    std::cerr << "| bfs_permute = " << bfs_permute << std::endl;
    std::cerr << "| max_num_nodes = " << max_num_nodes << std::endl;
    std::cerr << "| bits_compress = " << bits_compress << std::endl;
    std::cerr << "| dim_embed = " << dim_embed << std::endl;
    std::cerr << "| gpu = " << gpu << std::endl;
    std::cerr << "| seed = " << seed << std::endl;
    std::cerr << "======   end of tree_clib configuration ======" << std::endl;
#ifdef USE_GPU
    if (gpu >= 0)
    {
        cudaError_t t = cudaSetDevice(gpu);
        assert(t == cudaSuccess);
    }
#endif
}

void cfg::SetRandom()
{
    std::srand(cfg::seed);
    cfg::generator.seed(cfg::seed);
}
