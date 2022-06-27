// NOLINTBEGIN

#include "utils.h"
#include <random>
#include <thread>
#include <cassert>

void split_str(std::string s, std::string delim, std::vector<std::string>& result)
{
    result.clear();
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delim)) != std::string::npos) {
        token = s.substr(0, pos);
        result.push_back(token);
        s.erase(0, pos + delim.length());
    }
    result.push_back(s);
}

std::mt19937* get_local_rand_engine()
{
    static thread_local std::mt19937* local_rand_engine = nullptr;
    if (!local_rand_engine)
        local_rand_engine = new std::mt19937(clock() + (size_t)std::hash<std::thread::id>{}(std::this_thread::get_id()));
    return local_rand_engine;
}

double randu()
{
    std::mt19937* local_rand_engine = get_local_rand_engine();
    static thread_local std::uniform_real_distribution<double> real_dist(0.0, 1.0);

    return real_dist(*local_rand_engine);
}

template<typename Dtype>
Dtype rand_int(Dtype st, Dtype ed)
{
    assert(st < ed);
    std::mt19937* local_rand_engine = get_local_rand_engine();
    std::uniform_int_distribution<Dtype> dist(st, ed - 1);
    return dist(*local_rand_engine);
}

template unsigned rand_int<unsigned>(unsigned st, unsigned ed);
template uint64_t rand_int<uint64_t>(uint64_t st, uint64_t ed);

void hash_combine(long long& seed, long long key)
{
    seed ^= key + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template<>
std::string dtype2string<unsigned>()
{
    return "uint32";
}

template<>
std::string dtype2string<uint64_t>()
{
    return "uint64";
}
// NOLINTEND
