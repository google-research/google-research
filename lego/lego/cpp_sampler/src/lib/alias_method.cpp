#include <queue>
#include <iostream>
#include "alias_method.h"  // NOLINT
#include "utils.h"  // NOLINT

AliasMethod::AliasMethod()
{
    clear_table();
}

void AliasMethod::clear_table()
{
    num_choices = 0;
    prob_small.clear();
    choice_small.clear();
    choice_large.clear();
}

uint64_t AliasMethod::draw_sample()
{
    double u1 = randu();
    double u2 = randu();

    uint64_t k = this->num_choices * u1;
    if (k == this->num_choices)
        k--;
    if (u2 < this->prob_small[k])
        return this->choice_small[k];
    else
        return this->choice_large[k];
}

void AliasMethod::build_table(const double* weights, uint64_t _num_choices)
{
    clear_table();
    this->num_choices = _num_choices;
    double total_weights = 0.0;
    for (uint64_t i = 0; i < this->num_choices; ++i)
        total_weights += weights[i];
    std::vector<double> probs;
    probs.resize(this->num_choices);

    std::queue<uint64_t> buf_large, buf_small;
    double factor = this->num_choices / total_weights;
    for (uint64_t i = 0; i < this->num_choices; ++i)
    {
        probs[i] = weights[i] * factor;
        if (probs[i] >= 1.0)
            buf_large.push(i);
        else
            buf_small.push(i);
    }
    this->prob_small.resize(this->num_choices);
    this->choice_small.resize(this->num_choices);
    this->choice_large.resize(this->num_choices);

    for (uint64_t i = 0; i < this->num_choices; ++i)
    {
        if (buf_large.empty())
        {
          uint64_t small = buf_small.front();
          this->prob_small[i] = 1.0;
          this->choice_small[i] = this->choice_large[i] = small;
          continue;
        }
        uint64_t large = buf_large.front();
        buf_large.pop();
        uint64_t small = 0;
        double small_prob = 0.0;
        if (!buf_small.empty())
        {
            small = buf_small.front();
            buf_small.pop();
            small_prob = probs[small];
        }
        probs[large] -= 1.0 - small_prob;
        this->prob_small[i] = small_prob;
        this->choice_small[i] = small;
        this->choice_large[i] = large;
        if (probs[large] < 1.0)
            buf_small.push(large);
        else
            buf_large.push(large);
    }
}

void AliasMethod::setup_from_numpy(
    uint64_t num_choices,
    py::array_t<double,
                py::array::c_style | py::array::forcecast> sample_weights)
{
    const double* weights = sample_weights.unchecked<1>().data(0);
    this->build_table(weights, num_choices);
}

void AliasMethod::setup(std::vector<double>& sample_weights)
{
    const double* weights = sample_weights.data();
    this->build_table(weights, sample_weights.size());
}
