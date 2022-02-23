// NOLINTBEGIN

#include "knowledge_graph.h"
#include <cassert>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>

#include "utils.h"

template<typename Dtype>
SortedList<Dtype>::SortedList(Dtype _num_ent, Dtype _num_rel)
    : num_ent(_num_ent), num_rel(_num_rel), num_frag_rels(0), mem_shared(false)
{
    src_rel_map = rel_dst_map = dst_list = nullptr;
}

template<typename Dtype>
void SortedList<Dtype>::try_free()
{
    if (!mem_shared)
    {
        if (src_rel_map != nullptr)
            delete[] src_rel_map;
        if (rel_dst_map != nullptr)
            delete[] rel_dst_map;
        if (dst_list != nullptr)
            delete[] dst_list;
    }
    mem_shared = false;
    src_rel_map = nullptr;
    rel_dst_map = nullptr;
    dst_list = nullptr;
}

template<typename Dtype>
SortedList<Dtype>::~SortedList()
{
    try_free();
}

template<typename Dtype>
void SortedList<Dtype>::dump(FILE* fid, Dtype num_edges)
{
    assert(fwrite(&num_frag_rels, sizeof(Dtype), 1, fid) == 1ULL);
    assert(fwrite(src_rel_map, sizeof(Dtype), num_ent * 2ULL, fid) == num_ent * 2ULL);
    assert(fwrite(rel_dst_map, sizeof(Dtype), (uint64_t)num_frag_rels * 3ULL, fid) == (uint64_t)num_frag_rels * 3ULL);
    assert(fwrite(dst_list, sizeof(Dtype), num_edges, fid) == num_edges);
}

template<typename Dtype>
void SortedList<Dtype>::dump_nt(FILE* fid, Dtype num_edges)
{
    Dtype n_edges = 0;
    for (size_t i = 0; i < num_ent; ++i)
    {
        size_t rel_left = src_rel_map[i * 2ULL], rel_right = src_rel_map[i * 2ULL + 1ULL];
        for (size_t rel_ptr = rel_left; rel_ptr < rel_right; ++rel_ptr)
        {
            auto* ptr = rel_dst_map + rel_ptr * 3ULL;
            for (size_t j = ptr[1]; j < ptr[2]; ++j)
                fprintf(fid, "<e:%lld> <r:%lld> <e:%lld> .\n", (long long)i, (long long)ptr[0], (long long)dst_list[j]);
            n_edges += ptr[2] - ptr[1];
        }
    }
    assert(n_edges == num_edges);
}

template<typename Dtype>
void SortedList<Dtype>::load(FILE* fid, Dtype num_edges)
{
    try_free();
    assert(fread(&num_frag_rels, sizeof(Dtype), 1, fid) == 1);

    src_rel_map = new Dtype[num_ent * 2ULL];
    assert(fread(src_rel_map, sizeof(Dtype), num_ent * 2ULL, fid) == num_ent * 2ULL);
    rel_dst_map = new Dtype[(uint64_t)num_frag_rels * 3ULL];
    assert(fread(rel_dst_map, sizeof(Dtype), (uint64_t)num_frag_rels * 3ULL, fid) == (uint64_t)num_frag_rels * 3ULL);
    dst_list = new Dtype[num_edges];
    assert(fread(dst_list, sizeof(Dtype), num_edges, fid) == num_edges);
}

template<typename Dtype>
size_t SortedList<Dtype>::load(Dtype* buffer, Dtype num_edges)
{
    try_free();
    mem_shared = true;
    num_frag_rels = buffer[0];
    buffer++;
    src_rel_map = buffer;
    buffer += (size_t)num_ent * 2ULL;
    rel_dst_map = buffer;
    buffer += (size_t)num_frag_rels * 3ULL;
    dst_list = buffer;
    return 1ULL + (size_t)num_ent * 2ULL + (size_t)num_frag_rels * 3ULL + (size_t)num_edges;
}

template<typename Dtype>
template<int src_idx>
void SortedList<Dtype>::setup_edges(std::vector< std::tuple<Dtype, Dtype, Dtype> >& edges)
{
    if (edges.size() == 0)
        return;
    std::sort(edges.begin(), edges.end(), [](const std::tuple<Dtype, Dtype, Dtype>& x, const std::tuple<Dtype, Dtype, Dtype>& y) {
        if (std::get<src_idx>(x) <= std::get<src_idx>(y))
        {
            if (std::get<src_idx>(x) < std::get<src_idx>(y))
                return true;
            else if (std::get<1>(x) <= std::get<1>(y))
            {
                if (std::get<1>(x) < std::get<1>(y))
                    return true;
                else return (std::get<2-src_idx>(x) <= std::get<2-src_idx>(y));
            } else
                return false;
        } else
            return false;
    });
    auto last = std::unique(edges.begin(), edges.end()); // remove duplicated edges
    edges.erase(last, edges.end());
    try_free();
    dst_list = new Dtype[edges.size()];
    src_rel_map = new Dtype[num_ent * 2ULL];
    memset(src_rel_map, 0, sizeof(Dtype) * (uint64_t)num_ent * 2ULL);

    Dtype prev_ent = num_ent + 1ULL, prev_rel = num_rel + 1ULL;
    num_frag_rels = 0;
    for (uint64_t i = 0; i < edges.size(); ++i)
    {
        auto& edge = edges[i];
        Dtype src = std::get<src_idx>(edge), r = std::get<1>(edge), dst = std::get<2 - src_idx>(edge);
        dst_list[i] = dst;
        if (src != prev_ent)
        {
            num_frag_rels++;
            if (prev_ent < num_ent)
                src_rel_map[prev_ent * 2ULL + 1ULL] = num_frag_rels - 1ULL;
            src_rel_map[src * 2ULL] = num_frag_rels - 1ULL;
        } else if (r != prev_rel)
            num_frag_rels++;
        prev_ent = src;
        prev_rel = r;
    }
    src_rel_map[prev_ent * 2ULL + 1ULL] = num_frag_rels;
    rel_dst_map = new Dtype[(uint64_t)num_frag_rels * 3LL];

    prev_ent = num_ent + 1ULL;
    prev_rel = num_rel + 1ULL;
    uint64_t rel_pos = 0;
    for (uint64_t i = 0; i < edges.size(); ++i)
    {
        auto& edge = edges[i];
        Dtype src = std::get<src_idx>(edge), r = std::get<1>(edge);
        if (src != prev_ent || r != prev_rel)
        {
            if (prev_rel < num_rel)
            {
                rel_dst_map[rel_pos * 3ULL + 2ULL] = i;
                rel_pos++;
            }
            rel_dst_map[rel_pos * 3ULL] = r;
            rel_dst_map[rel_pos * 3ULL + 1ULL] = i;
        }
        prev_ent = src;
        prev_rel = r;
    }
    rel_dst_map[rel_pos * 3ULL + 2ULL] = edges.size();
    assert(rel_pos + 1ULL == num_frag_rels);    
}

template<typename Dtype>
Dtype* SortedList<Dtype>::locate_rel_dst(Dtype src, Dtype r)
{
    if (src_rel_map == nullptr)
        return nullptr;
    Dtype rel_left = src_rel_map[src * 2ULL], rel_right = src_rel_map[src * 2ULL + 1ULL];
    if (rel_left >= rel_right)
        return nullptr;
    rel_right--;
    uint64_t mid;
    while (rel_left <= rel_right)
    {
        mid = (rel_left + rel_right) / 2ULL;
        Dtype cur_r = rel_dst_map[mid * 3ULL];
        if (cur_r == r) // found
            return rel_dst_map + mid * 3ULL;
        if (r < cur_r && mid != rel_left)
            rel_right = mid - 1ULL;
        else if (r > cur_r)
            rel_left = mid + 1ULL;
        else
            return nullptr;
    }
    return nullptr;
}


template<typename Dtype>
bool SortedList<Dtype>::has_edge(Dtype src, Dtype r, Dtype dst)
{
    Dtype* range = locate_rel_dst(src, r);
    if (range == nullptr)
        return false;
    return std::binary_search(dst_list + range[1], dst_list + range[2], dst);
}

template<typename Dtype>
Dtype SortedList<Dtype>::src_degree(Dtype node)
{
    Dtype* ptr = src_rel_map + node * 2ULL;
    if (ptr[1] - ptr[0]) // has edge
    {
        Dtype first_pos = rel_dst_map[ptr[0] * 3ULL + 1ULL];
        Dtype last_pos = rel_dst_map[(ptr[1] - 1ULL) * 3ULL + 2ULL];
        return last_pos - first_pos;
    } else return 0;
}

template<typename Dtype>
Dtype SortedList<Dtype>::src_num_unique_rel(Dtype node)
{
    Dtype* ptr = src_rel_map + node * 2ULL;
    return ptr[1] - ptr[0];
}

template class SortedList<unsigned>;
template class SortedList<uint64_t>;

template<typename Dtype>
KG<Dtype>::KG() : num_ent(0), num_rel(0), num_edges(0), ent_out(nullptr), ent_in(nullptr)
{
    this->dtype = dtype2string<Dtype>();
}

template<typename Dtype>
KG<Dtype>::KG(Dtype _num_ent, Dtype _num_rel) : num_ent(_num_ent), num_rel(_num_rel), num_edges(0), ent_out(nullptr), ent_in(nullptr)
{
    this->dtype = dtype2string<Dtype>();
}

template<typename Dtype>
void KG<Dtype>::try_free()
{
    if (ent_out)
        delete ent_out;
    if (ent_in)
        delete ent_in;
    ent_out = nullptr;
    ent_in = nullptr;
}

template<typename Dtype>
KG<Dtype>::~KG()
{
    try_free();
}

template<typename Dtype>
void KG<Dtype>::load_edges_from_file(const std::string& fname, const bool has_reverse_edges, std::vector< std::tuple<Dtype, Dtype, Dtype> >& edges)
{
    std::ifstream fin(fname);
    Dtype x, r, y;
    while (fin >> x >> r >> y)
    {
        assert(x >= 0 && x < this->num_ent);
        assert(y >= 0 && y < this->num_ent);
        if (has_reverse_edges)
        {
            assert(r >= 0 && r < this->num_rel);
            edges.push_back(std::make_tuple(x, r, y));
        } else {
            r *= 2u;
            edges.push_back(std::make_tuple(x, r, y));
            r += 1u;
            edges.push_back(std::make_tuple(y, r, x));
            assert(r < this->num_rel);
        }
    }
    fin.close();
}

template<typename Dtype>
void KG<Dtype>::build_kg_from_edges(std::vector< std::tuple<Dtype, Dtype, Dtype> >& edges)
{
    ent_out = new SortedList<Dtype>(num_ent, num_rel);
    ent_out->template setup_edges<0>(edges);
    ent_in = new SortedList<Dtype>(num_ent, num_rel);
    ent_in->template setup_edges<2>(edges);
    num_edges = edges.size();
}

template<typename Dtype>
void KG<Dtype>::load_triplets(const std::string& fname, const bool has_reverse_edges)
{
    std::vector< std::tuple<Dtype, Dtype, Dtype> > edges;
    load_edges_from_file(fname, has_reverse_edges, edges);

    build_kg_from_edges(edges);
}

template<typename Dtype>
void KG<Dtype>::load_triplets_from_files(py::list list_files, const bool has_reverse_edges)
{
    std::vector< std::tuple<Dtype, Dtype, Dtype> > edges;
    for (size_t i = 0; i < py::len(list_files); ++i)
    {
        std::string fname = py::cast<std::string>(list_files[i]);
        load_edges_from_file(fname, has_reverse_edges, edges);
    }
    build_kg_from_edges(edges);
}

template<typename Dtype>
bool KG<Dtype>::has_forward_edge(Dtype src, Dtype r, Dtype dst)
{
    if (ent_out == nullptr)
        return false;
    return ent_out->has_edge(src, r, dst);
}

template<typename Dtype>
bool KG<Dtype>::has_backward_edge(Dtype src, Dtype r, Dtype dst)
{
    if (ent_in == nullptr)
        return false;
    return ent_in->has_edge(src, r, dst);
}

template<typename Dtype>
size_t KG<Dtype>::ptr()
{
    return (size_t)this;
}

template<typename Dtype>
void KG<Dtype>::load(const std::string& fname)
{
    try_free();
    FILE* fin = fopen(fname.c_str(), "rb");
    assert(fread(&num_ent, sizeof(Dtype), 1, fin) == 1);
    assert(fread(&num_rel, sizeof(Dtype), 1, fin) == 1);
    assert(fread(&num_edges, sizeof(Dtype), 1, fin) == 1);

    ent_out = new SortedList<Dtype>(num_ent, num_rel);
    ent_out->load(fin, num_edges);
    ent_in = new SortedList<Dtype>(num_ent, num_rel);
    ent_in->load(fin, num_edges);
    fclose(fin);
}

template<typename Dtype>
void KG<Dtype>::load_from_numpy(void* _triplets, size_t n_triplets, const bool has_reverse_edges)
{
    std::vector< std::tuple<Dtype, Dtype, Dtype> > edges;
    edges.reserve(n_triplets);
    int64_t* triplets = static_cast<int64_t*>(_triplets);
    for (size_t i = 0; i < n_triplets; ++i)
    {
        Dtype h = triplets[0];
        Dtype r = triplets[1];
        Dtype t = triplets[2];
        if (has_reverse_edges)
        {
            assert(r < this->num_rel);
            edges.push_back(std::make_tuple(h, r, t));
        } else { // add reverse edges
            r *= 2u;
            edges.push_back(std::make_tuple(h, r, t));
            r += 1u;
            edges.push_back(std::make_tuple(t, r, h));
            assert(r < this->num_rel);
        }
        triplets += 3;
    }
    build_kg_from_edges(edges);
}

template<typename Dtype>
void KG<Dtype>::load_from_binary(Dtype* buf_ptr, size_t n_ints)
{
    try_free();

    num_ent = buf_ptr[0];
    num_rel = buf_ptr[1];
    num_edges = buf_ptr[2];
    buf_ptr += 3ULL;

    ent_out = new SortedList<Dtype>(num_ent, num_rel);
    size_t offset_out = ent_out->load(buf_ptr, num_edges);
    ent_in = new SortedList<Dtype>(num_ent, num_rel);
    size_t offset_in = ent_in->load(buf_ptr + offset_out, num_edges);
    assert(offset_out + offset_in + 3ULL == n_ints);
}

template<typename Dtype>
void KG<Dtype>::dump(const std::string& fname)
{
    FILE* fout = fopen(fname.c_str(), "wb");
    assert(fwrite(&num_ent, sizeof(Dtype), 1, fout) == 1);
    assert(fwrite(&num_rel, sizeof(Dtype), 1, fout) == 1);
    assert(fwrite(&num_edges, sizeof(Dtype), 1, fout) == 1);

    ent_out->dump(fout, num_edges);
    ent_in->dump(fout, num_edges);
    fclose(fout);
}

template<typename Dtype>
void KG<Dtype>::dump_nt(const std::string& fname)
{
    FILE* fout = fopen(fname.c_str(), "w");
    ent_out->dump_nt(fout, num_edges);
    fclose(fout);
}

template<typename Dtype>
Dtype KG<Dtype>::in_degree(Dtype node)
{
    return ent_in->src_degree(node);
}

template<typename Dtype>
Dtype KG<Dtype>::out_degree(Dtype node)
{
    return ent_out->src_degree(node);
}

template<typename Dtype>
Dtype KG<Dtype>::unique_num_in_rel(Dtype node)
{
    return ent_in->src_num_unique_rel(node);
}

template<typename Dtype>
Dtype KG<Dtype>::unique_num_out_rel(Dtype node)
{
    return ent_out->src_num_unique_rel(node);
}

template<typename Dtype>
Dtype KG<Dtype>::max_degree()
{
    Dtype max_degree = 0;
    for (Dtype src = 0; src < num_ent; ++src)
    {
        uint64_t st = ent_out->src_rel_map[src * 2ULL];
        uint64_t ed = ent_out->src_rel_map[src * 2ULL + 1ULL];

        for (uint64_t r = st; r < ed; ++r)
        {
            auto* ptr = ent_out->rel_dst_map + r * 3ULL;
            if (ptr[2] - ptr[1] > max_degree)
                max_degree = ptr[2] - ptr[1];
        }
    }
    return max_degree;
}

template class KG<unsigned>;
template class KG<uint64_t>;

template<typename Dtype>
void sample_rand_neighbor(SortedList<Dtype>* ent_list, Dtype node, Dtype& r, Dtype& other_node)
{
    Dtype* rel_range = ent_list->src_rel_map + node * 2ULL;
    assert(rel_range[1] - rel_range[0]); // there should be some edges;
    Dtype first_pos = ent_list->rel_dst_map[rel_range[0] * 3ULL + 1ULL];
    Dtype last_pos = ent_list->rel_dst_map[(rel_range[1] - 1ULL) * 3ULL + 2ULL];
    Dtype dst_offset = rand_int(first_pos, last_pos);
    other_node = ent_list->dst_list[dst_offset];

    long long left = rel_range[0], right = rel_range[1] - 1LL;
    while (left <= right) {
        Dtype mid = (left + right) / 2LL;
        auto* cur_rel_dst = ent_list->rel_dst_map + mid * 3LL;
        if (cur_rel_dst[1] <= dst_offset && dst_offset < cur_rel_dst[2])
        {
            r = cur_rel_dst[0];
            return;
        }
        if (cur_rel_dst[2] <= dst_offset)
            left = mid + 1LL;
        else
            right = mid - 1LL;
    }
    assert(false);
}

template void sample_rand_neighbor<unsigned>(SortedList<unsigned>* ent_list, unsigned node, unsigned& r, unsigned& other_node);
template void sample_rand_neighbor<uint64_t>(SortedList<uint64_t>* ent_list, uint64_t node, uint64_t& r, uint64_t& other_node);

// NOLINTEND
