// Copyright 2021 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Computes the hacky "cross-entropy"-related measures between the two models.
//
// Given the two models, P (which we call here `source`) and Q (we call
// `destination`) assumes that P represents the `true` distribution and
// Q the predictions and computes the cross-entropy measure between the
// models H(P, Q) given by
//
//   H(P, Q) = –\sigma_{x \in in X} P(x) * \log_2(Q(x)) ,
//
// where the models are trained on the same data and evaluated on the same
// X, where only the alphabets for P and Q differ. In a similar fashion,
// we compute Kullback-Leibler divergence KL(P, Q) given by
//
//   KL(P, Q) = –\sigma_{x \in in X} P(x) * \log_2(Q(x) / P(x)) .
//
// The inverse H(Q, P) and KL(Q, P) are reported as well if the order of
// source and destination flags are reversed. Among other measures that
// we report are:
//
//   - Regular entropies:
//       H(P) = -\sigma_{x \in X} P(x) * \log_2(P(x)),
//   - Conditional entropy:
//       H(P|Q) = -\sigma{p,q \in P, Q}
//         Prob(p, q) * \log_2(Prob(p, q) / Prob(q)), where Prob(p, q)
//         is a joint distribution,
//   - Mutual information:
//       I(P, Q) = H(Q) - H(Q|P) = H(P) + H(Q) - H(P, Q),
//   - Normalized (cross-)entropies and KL-divergences:
//       H_n(P) = H(P) / \log_2(N).
//
// Both empirical (corpus-based) and model (automation state-bssed) measures
// are reported, where possible.

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "fst/flags.h"
#include "fst/extensions/far/far.h"
#include "fst/expanded-fst.h"
#include "fst/matcher.h"
#include "fst/mutable-fst.h"
#include "ngram/ngram-output.h"

DECLARE_string(start_symbol);
DECLARE_string(end_symbol);

DEFINE_bool(use_phimatcher, false,
            "Use phi matcher and composition.");

DEFINE_string(OOV_symbol, "",
              "Existing symbol for OOV class");

DEFINE_double(OOV_class_size, 10000,
              "Number of members of OOV class");

DEFINE_double(OOV_probability, 0,
              "Unigram probability for OOVs.");

DEFINE_string(context_pattern, "",
              "Restrict perplexity computation to contexts defined by"
              " pattern (default: no restriction).");

DEFINE_string(ngram_source_fst, "",
              "Source (P) ngram model FST.");

DEFINE_string(source_samples_far, "",
              "Source (P) samples FAR.");

DEFINE_string(ngram_destination_fst, "",
              "Destination (Q) ngram model FST.");

DEFINE_string(ngram_joint_fst, "",
              "Joint (P, Q) ngram model FST.");

DEFINE_string(destination_samples_far, "",
              "Destination (Q) samples FAR.");

DEFINE_bool(source_is_phonemic, true,
            "True if the source FST is phonemic.");

DEFINE_string(info_header, "",
              "Information header to print with the output.");

using fst::ArcIterator;
using fst::StdArc;
using fst::StdExpandedFst;
using fst::StdMutableFst;
using fst::StdVectorFst;
using fst::SymbolTable;

namespace {

// Helper class that exposes some of the ngram output internals.
class NGramOutputHelper : public ngram::NGramOutput {
 public:
  explicit NGramOutputHelper(StdMutableFst *infst)
      : ngram::NGramOutput(infst,
                           /* ostrm= */std::cout,
                           /* backoff_label= */0,
                           /* check_consistency= */true,
                           /* context_pattern= */"",
                           /* include_all_suffixes= */false) {}

  // Sets up the computation.
  bool Init(const std::vector<std::unique_ptr<StdVectorFst>> &samples);

  // Relabels and sets the symbols on input FST.
  std::unique_ptr<const StdVectorFst> PrepareInputFst(
      const StdVectorFst &input_fst) {
    std::unique_ptr<StdVectorFst> infst(input_fst.Copy());
    RelabelAndSetSymbols(infst.get(), *symbol_fst_);
    return infst;
  }

  // A simplified override of the state search method.
  void FindNextStateInModel(StateId *mst, Label label,
                            double OOV_cost, Label OOV_label,
                            double *neglogprob, int *word_cnt,
                            int *oov_cnt, int *skipped,
                            std::string *history,
                            std::vector<Label> *ngram,
                            std::vector<double> *arc_costs) const;

  // Mimics a \phi matcher: follow backoff arcs until label found or no backoff.
  bool FindNGramInModel(StateId *mst, int *order, Label label,
                        double *total_cost,
                        std::vector<double> *arc_costs) const;

  // Assuming the input is a linear FST, convert it to string.
  std::string ToString(const fst::Fst<StdArc> &infst) const;

  // Gets the N-gram model: each n-gram with its weight.
  void GetNGramModel(NGramOutput::ShowBackoff showeps, bool neglogs,
                     bool intcnts,
                     std::vector<std::pair<std::string,
                       std::pair<double, double>>> *ngrams) const;

  // Converts negative log prob (positive quantity) to the binary base. Assumes
  // that the original logprob was computed using natural logarithm.
  // Returns negative log prob (i.e. positive) in a binary base.
  double ToBinaryBase(double neglogprob) const {
    return -neglogprob / std::log(2.0);
  }

  Label oov_label() const { return oov_label_; }

 protected:
  // Stores n-grams leaving a particular state.
  void GetNGrams(StdArc::StateId st, const std::string &str,
                 NGramOutput::ShowBackoff showeps, bool neglogs,
                 bool intcnts,
                 std::vector<std::pair<std::string,
                   std::pair<double, double>>> *ngrams) const;

 private:
  Label oov_label_;
  std::string oov_symbol_;
  double oov_probability_;
  std::unique_ptr<StdMutableFst> symbol_fst_;
};

bool NGramOutputHelper::Init(
    const std::vector<std::unique_ptr<StdVectorFst>> &samples) {
  oov_symbol_ = FST_FLAGS_OOV_symbol;
  oov_probability_ = FST_FLAGS_OOV_probability;
  if (!GetOOVLabel(&oov_probability_, &oov_symbol_, &oov_label_)) {
    LOG(ERROR) << "Failed to fetch OOV label!";
    return false;
  }
  symbol_fst_.reset(!samples[0]->InputSymbols() ? GetMutableFst()->Copy() :
                    samples[0]->Copy());
  if (!symbol_fst_) {
    LOG(ERROR) << "Missing symbol table!";
    return false;
  }
  RenormUnigramForOOV(ngram::kSpecialLabel, oov_label_,
                      FST_FLAGS_OOV_class_size, oov_probability_);
  if (Error()) {
    LOG(ERROR) << "Failed to renorm unigram!";
    return false;
  }
  return true;
}

std::string NGramOutputHelper::ToString(
    const fst::Fst<StdArc> &infst) const {
  std::string buf;
  StateId st = infst.Start();
  while (infst.NumArcs(st) != 0) {
    ArcIterator<fst::Fst<StdArc>> aiter(infst, st);
    StdArc arc = aiter.Value();
    std::string symbol = GetFst().InputSymbols()->Find(arc.ilabel);
    if (st != infst.Start()) buf += " ";
    buf += symbol;
    st = arc.nextstate;
  }
  return buf;
}

void NGramOutputHelper::FindNextStateInModel(
    StateId *mst, Label label, double OOV_cost, Label OOV_label,
    double *neglogprob, int *word_cnt, int *oov_cnt, int *skipped,
    std::string *history, std::vector<Label> *ngram,
    std::vector<double> *arc_costs) const {
  const bool in_context = InContext(*ngram);
  int order;
  double ngram_cost = 0.0;
  const std::string symbol = GetFst().InputSymbols()->Find(label);
  ++(*word_cnt);
  std::vector<double> local_arc_costs;
  if (!FindNGramInModel(mst, &order, label, &ngram_cost,
                        &local_arc_costs)) {  // OOV
    ++(*oov_cnt);
    // Unigram state.
    ngram_cost += OOV_cost;
    arc_costs->push_back(OOV_cost);
    if (OOV_cost != StdArc::Weight::Zero().Value()) {
      if (in_context) {
        *neglogprob += ngram_cost;
        std::copy(local_arc_costs.begin(), local_arc_costs.end(),
                  std::back_inserter(*arc_costs));
      }
    } else {
      ++(*skipped);
    }
    *mst = (UnigramState() >= 0) ? UnigramState() : GetFst().Start();
    *history = "";
    *ngram = std::vector<Label>(HiOrder(), 0);
  } else {
    if (label == OOV_label) ++(*oov_cnt);
    if (in_context) {
      *neglogprob += ngram_cost;
      std::copy(local_arc_costs.begin(), local_arc_costs.end(),
                std::back_inserter(*arc_costs));
    }
    *history = symbol + " ...";
    ngram->erase(ngram->begin());
    ngram->push_back(label);
  }
}

bool NGramOutputHelper::FindNGramInModel(StateId *mst, int *order, Label label,
                                         double *total_cost,
                                         std::vector<double> *arc_costs) const {
  if (label < 0) return false;
  StateId currstate = *mst;
  *total_cost = 0;
  *mst = -1;
  const Label backoff_label = BackoffLabel();
  while (*mst < 0) {
    fst::Matcher<fst::Fst<StdArc>> matcher(GetFst(), fst::MATCH_INPUT);
    matcher.SetState(currstate);
    if (matcher.Find(label)) {  // arc found out of current state
      StdArc arc = matcher.Value();
      *order = StateOrder(currstate);
      *mst = arc.nextstate;  // assign destination as new model state
      const double cost = ScalarValue(arc.weight);
      *total_cost += cost;         // add cost to total
      arc_costs->push_back(cost);
    } else if (matcher.Find(backoff_label)) {  // follow backoff arc
      currstate = -1;
      for (; !matcher.Done(); matcher.Next()) {
        StdArc arc = matcher.Value();
        if (arc.ilabel == backoff_label) {
          currstate = arc.nextstate;  // make current state backoff state
          const double cost = ScalarValue(arc.weight);
          *total_cost += cost;  // add in backoff cost
          arc_costs->push_back(cost);
        }
      }
      if (currstate < 0) return false;
    } else {
      return false;  // Found label in symbol list, but not in model
    }
  }
  return true;
}

void NGramOutputHelper::GetNGrams(StdArc::StateId st, const std::string &str,
                                  NGramOutput::ShowBackoff showeps,
                                  bool neglogs, bool intcnts,
                                  std::vector<std::pair<std::string,
                                    std::pair<double, double>>> *ngrams) const {
  if (st < 0) return;  // ignore for st < 0
  const bool show = InContext(st);
  for (ArcIterator<StdExpandedFst> aiter(GetExpandedFst(), st); !aiter.Done();
       aiter.Next()) {
    const StdArc arc = aiter.Value();
    if (arc.ilabel == BackoffLabel() &&
        showeps != ShowBackoff::EPSILON)  // skip backoff unless showing EPSILON
      continue;
    // Find symbol str.
    std::string symbol = GetFst().InputSymbols()->Find(arc.ilabel);
    std::string newstr = str;                   // history string
    AppendWordToNGramHistory(&newstr, symbol);  // Full n-gram string
    if (show) {
      const double prob = WeightRep(arc.weight.Value(), neglogs, intcnts);
      double bo_prob = 0.0;  // Default backoff probability.
      if (showeps == ShowBackoff::INLINE &&
          StateOrder(arc.nextstate) > StateOrder(st))  // show backoff
        bo_prob = WeightRep(GetBackoffCost(arc.nextstate).Value(), neglogs,
                            intcnts);
      ngrams->push_back(std::make_pair(newstr, std::make_pair(prob, bo_prob)));
    }
    if (arc.ilabel != BackoffLabel() &&  // depth-first traversal
        StateOrder(arc.nextstate) > StateOrder(st))
      GetNGrams(arc.nextstate, newstr, showeps, neglogs, intcnts, ngrams);
  }
  if (show &&
      GetFst().Final(st) != StdArc::Weight::Zero()) {  // show </s> counts
    std::string history;
    if (!str.empty())  // if history string, print it
      history = str + " ";
    history += FST_FLAGS_end_symbol;
    const double prob = WeightRep(GetFst().Final(st).Value(),
                                  neglogs, intcnts);
    ngrams->push_back(std::make_pair(history, std::make_pair(prob, 0.0)));
  }
}

void NGramOutputHelper::GetNGramModel(
    NGramOutput::ShowBackoff showeps, bool neglogs,
    bool intcnts,
    std::vector<std::pair<std::string,
    std::pair<double, double>>> *ngrams) const {
  if (Error()) return;
  std::string str = "";  // init n-grams from unigram state
  double start_wt;  // weight of <s> (count or prob) same as unigram </s>
  if (UnigramState() >= 0) {  // show n-grams from unigram state
    GetNGrams(UnigramState(), str, showeps, neglogs, intcnts, ngrams);
    start_wt =
        WeightRep(GetFst().Final(UnigramState()).Value(), neglogs, intcnts);
    str = FST_FLAGS_start_symbol;  // init n-grams from <s> state
  } else {
    start_wt =
        WeightRep(GetFst().Final(GetFst().Start()).Value(), neglogs, intcnts);
  }
  // print <s> unigram following SRILM
  if (InContext(GetFst().Start())) {
    std::string history = FST_FLAGS_start_symbol;
    double bo_prob = 0.0;
    if (showeps == ShowBackoff::INLINE &&
        UnigramState() >= 0)  // <s> state exists, then show backoff
      bo_prob = WeightRep(GetBackoffCost(GetFst().Start()).Value(), neglogs,
                          intcnts);
    ngrams->push_back(std::make_pair(history,
                                     std::make_pair(start_wt, bo_prob)));
  }
  GetNGrams(GetFst().Start(), str, showeps, neglogs, intcnts, ngrams);
}

// Reads the supplied FST archive and returns individual FSTs in a vector.
std::vector<std::unique_ptr<StdVectorFst>> LoadSamples(
    const std::string &far_path) {
  LOG(INFO) << "Loading samples from " << far_path;
  std::unique_ptr<fst::FarReader<StdArc>> far_reader(
      fst::FarReader<StdArc>::Open(far_path));
  if (!far_reader) {
    LOG(FATAL) << "Unable to open fst archive " << far_path;
  }
  std::vector<std::unique_ptr<StdVectorFst>> fsts;
  while (!far_reader->Done()) {
    fsts.push_back(absl::make_unique<StdVectorFst>(*far_reader->GetFst()));
    far_reader->Next();
  }
  LOG(INFO) << "Loaded " << fsts.size() << " samples.";
  return fsts;
}

// Applies n-gram models to FSTs. Assumes linear FSTs, accumulates stats.
void ApplyNGramsToFsts(const StdVectorFst &input_source_fst,
                       NGramOutputHelper *model_source,
                       const StdVectorFst &input_dest_fst,
                       NGramOutputHelper *model_dest,
                       double *logprob_ce, double *logprob_kl,
                       int *words, int *oovs,
                       int *words_skipped) {
  std::unique_ptr<const StdVectorFst> infst_source =
      model_source->PrepareInputFst(input_source_fst);
  std::unique_ptr<const StdVectorFst> infst_dest = model_dest->PrepareInputFst(
      input_dest_fst);
  if (infst_source->NumStates() != infst_dest->NumStates()) {
    LOG(FATAL) << "Mismatching number of states in input FSTs: "
               << "source (" << infst_source->NumStates() << "), "
               << "destination (" << infst_dest->NumStates() << ")\n\t"
               << "     source = " << model_source->ToString(*infst_source)
               << "\n\tdestination = " << model_dest->ToString(*infst_dest);
  }
  const double oov_cost = StdArc::Weight::Zero().Value();
  StdArc::StateId st_source = infst_source->Start();
  StdArc::StateId mst_source = model_source->GetFst().Start();
  const StdArc::Label oov_label_source = model_source->oov_label();
  StdArc::StateId st_dest = infst_dest->Start();
  StdArc::StateId mst_dest = model_dest->GetFst().Start();
  const StdArc::Label oov_label_dest = model_dest->oov_label();
  int word_cnt = 0, oov_cnt = 0, skipped = 0, dummy = 0;
  std::string history_source, history_dest;
  std::vector<StdArc::Label> ngram_source(model_source->HiOrder(), 0);
  std::vector<StdArc::Label> ngram_dest(model_dest->HiOrder(), 0);
  double total_ce = 0.0;  // Cross-entropy: \sum_x P(x) * log_2(Q(x)).
  double total_kl = 0.0;  // KL: \sum_x P(x) * log_2(Q(x) / P(x)).
  while (infst_source->NumArcs(st_source) != 0) {
    // Compute P(x).
    ArcIterator<fst::Fst<StdArc>> aiter_source(*infst_source, st_source);
    const StdArc arc_source = aiter_source.Value();
    st_source = arc_source.nextstate;
    double neglogprob_source = 0.0;  // negative natural log score.
    std::vector<double> arc_costs_source;
    model_source->FindNextStateInModel(
        &mst_source, arc_source.ilabel, oov_cost, oov_label_source,
        &neglogprob_source, &word_cnt, &oov_cnt, &skipped,
        &history_source, &ngram_source, &arc_costs_source);
    const double prob_source = std::exp(-neglogprob_source);

    // Compute Q(x).
    ArcIterator<fst::Fst<StdArc>> aiter_dest(*infst_dest, st_dest);
    const StdArc arc_dest = aiter_dest.Value();
    st_dest = arc_dest.nextstate;
    double neglogprob_dest = 0.0;  // negative natural log score.
    std::vector<double> arc_costs_dest;
    model_dest->FindNextStateInModel(
        &mst_dest, arc_dest.ilabel, oov_cost, oov_label_dest, &neglogprob_dest,
        &dummy, &dummy, &dummy, &history_dest, &ngram_dest, &arc_costs_dest);
    const double prob_dest = std::exp(-neglogprob_dest);

    // Accumulate P(x)\log_2(Q(x)) and P(x)\log_2(Q(x)/P(x)).
    //
    // Note: the sizes of the per-arc costs for the source and the destination
    // don't necessarily match because the number of arcs in the two models may
    // be different. Hence, it's not clear how to perform the per-arc
    // synchronous accumulation.
    total_ce += (prob_source * model_dest->ToBinaryBase(neglogprob_dest));
    total_kl += (prob_source * std::log2(prob_dest / prob_source));
  }

  // Add final costs.
  int order;
  const double final_prob_source = std::exp(
      -model_source->ScalarValue(model_source->FinalCostInModel(
          mst_source, &order)));
  const double ngram_cost_dest = model_dest->ScalarValue(
      model_dest->FinalCostInModel(mst_dest, &order));
  const double final_prob_dest = std::exp(-ngram_cost_dest);
  if (model_source->InContext(ngram_source) &&
      model_dest->InContext(ngram_dest)) {
    total_ce += (final_prob_source * model_dest->ToBinaryBase(ngram_cost_dest));
    total_kl += (final_prob_source * std::log2(
        final_prob_dest / final_prob_source));
  }

  // Finalize counters.
  *logprob_ce += total_ce;
  *logprob_kl += total_kl;
  *words += word_cnt;
  *oovs += oov_cnt;
  *words_skipped += skipped;
}

// Computes cross-entropy between two models using an empirical estimate
// that uses the samples from the test corpus.
void EmpiricalCrossEntropy(
    NGramOutputHelper *model_source,
    const std::vector<std::unique_ptr<StdVectorFst>> &source_samples,
    NGramOutputHelper *model_dest,
    const std::vector<std::unique_ptr<StdVectorFst>> &dest_samples) {
  if (source_samples.size() != dest_samples.size()) {
    LOG(FATAL) << "Cardinalities of sample sets should match!";
  }
  const int num_samples = source_samples.size();
  int word_cnt = 0, oov_cnt = 0, words_skipped = 0;
  double logprob_ce = 0.0, logprob_kl = 0.0;
  for (int i = 0; i < num_samples; ++i) {
    ApplyNGramsToFsts(*(source_samples[i]), model_source,
                      *(dest_samples[i]), model_dest,
                      &logprob_ce, &logprob_kl,
                      &word_cnt, &oov_cnt, &words_skipped);
  }
  LOG(INFO) << num_samples << " sentences, " << word_cnt << " words, "
            << oov_cnt << " OOVs.";
  if (words_skipped > 0) {
    LOG(INFO) << "NOTE: " << words_skipped << " OOVs with no probability"
              << " were skipped in cross-entropy calculation.";
    word_cnt -= words_skipped;
  }
  const double ce = -logprob_ce;
  const double nce = ce / std::log2(word_cnt + num_samples);
  const double kl = -logprob_kl;
  const double nkl = kl / std::log2(word_cnt + num_samples);
  LOG(INFO) << "Cross-entropy H(P, Q): " << ce << " bits.";
  LOG(INFO) << "\"Normalized\" cross-entopy H(P, Q): " << nce << " bits.";
  LOG(INFO) << "KL divergence KL(P||Q): " << kl << " bits.";
  LOG(INFO) << "\"Normalized\" KL divergence KL(P||Q): " << nkl << " bits.";
  std::cout << FST_FLAGS_info_header
            << " Empirical metrics: " << ce << "\t" << nce << "\t"
            << kl << "\t" << nkl << std::endl;
}

// Computes model entropy (regular and normalized) from ngrams.
std::pair<double, double> Entropy(const std::vector<std::pair<std::string,
                                    std::pair<double, double>>> &ngrams) {
  double logprob = 0.0;
  for (const auto &ngram : ngrams) {
    const double prob = ngram.second.first;
    if (prob == 0.0) continue;
    logprob += (prob * std::log2(prob));
  }
  return std::make_pair(-logprob, -logprob / std::log(ngrams.size()));
}

// Computes model entropy (regular and normalized) from marginal state
// distributions. The sum of all the state probabilities is equal to
// the model order.
std::pair<double, double> StateEntropy(const NGramOutputHelper &model) {
  std::vector<double> state_probs;
  model.CalculateStateProbs(&state_probs, /* stationary= */false);
  double logprob = 0.0;
  for (const double prob : state_probs) {
    logprob += (prob * std::log2(prob));
  }
  return std::make_pair(-logprob, -logprob / std::log2(state_probs.size()));
}

// Computes model (non-empirical, exact) entropies.
void ModelEntropy(const NGramOutputHelper &model_source,
                  const NGramOutputHelper &model_dest) {
  const ngram::NGramOutput::ShowBackoff show_backoff =
      ngram::NGramOutput::ShowBackoff::NONE;
  std::vector<std::pair<std::string, std::pair<double, double>>> ngrams_source;
  model_source.GetNGramModel(show_backoff, false, false, &ngrams_source);
  const std::pair<double, double> e_source = Entropy(ngrams_source);
  const std::pair<double, double> se_source = StateEntropy(model_source);
  LOG(INFO) << "Source model entropy H(P): " << e_source.first;
  LOG(INFO) << "Source model normalized entropy H_n(P): " << e_source.second;
  LOG(INFO) << "Source model state entropy H_s(P): " << se_source.first;
  LOG(INFO) << "Source model normalized state entropy H_ns(P): "
            << se_source.second;

  std::vector<std::pair<std::string, std::pair<double, double>>> ngrams_dest;
  model_dest.GetNGramModel(show_backoff, false, false, &ngrams_dest);
  const std::pair<double, double> e_dest = Entropy(ngrams_dest);
  const std::pair<double, double> se_dest = StateEntropy(model_dest);
  LOG(INFO) << "Destination model entropy H(Q): " << e_dest.first;
  LOG(INFO) << "Destination model normalized entropy H_n(Q): " << e_dest.second;
  LOG(INFO) << "Destination model state entropy H_s(P): " << se_dest.first;
  LOG(INFO) << "Destination model normalized state entropy H_ns(P): "
            << se_dest.second;

  const double e_diff = e_dest.first - e_source.first;
  const double ne_diff = e_dest.second - e_source.second;
  const double se_diff = se_dest.first - se_source.first;
  const double nse_diff = se_dest.second - se_source.second;
  LOG(INFO) << "Model diff: Destination H(Q) - Source H(P): " << e_diff;
  LOG(INFO) << "Model diff: Destination H_n(Q) - Source H_n(P): " << ne_diff;
  LOG(INFO) << "Model diff: Destination H_s(Q) - Source H_s(P): " << se_diff;
  LOG(INFO) << "Model diff: Destination H_ns(Q) - Source H_ns(P): " << nse_diff;

  std::cout << FST_FLAGS_info_header
            << " Model metrics: " << e_source.first << "\t" << e_source.second
            << "\t" << se_source.first << "\t" << se_source.second << "\t"
            << e_dest.first << "\t" << e_dest.second << "\t"
            << se_dest.first << "\t" << se_dest.second << "\t" << e_diff
            << "\t" << ne_diff << "\t" << se_diff << "\t" << nse_diff
            << std::endl;
}

// Computes the mapping between the joint distribution labels and corresponding
// marginal distribution labels. Side ID specifies which side of the joint
// symbol to look at it: graphemic (0) or phonemic (1).
void JointToMarginalLabelMapping(
    const SymbolTable &joint_symbols, const SymbolTable &marginal_symbols,
    int side_id,
    std::map<StdArc::Label, StdArc::Label> *joint2marginal) {
  const int64 num_joint_symbols = joint_symbols.NumSymbols();
  for (int64 pos = 0; pos < num_joint_symbols; ++pos) {
    const int64 key = joint_symbols.GetNthKey(pos);
    if (key == fst::kNoSymbol) {
      LOG(FATAL) << "Invalid position: " << pos;
    }
    const std::string joint_symbol = joint_symbols.Find(key);
    if (joint_symbol.empty()) {
      LOG(FATAL) << "No symbol for key: " << key;
    }
    const std::vector<std::string> &toks = absl::StrSplit(joint_symbol, '/');
    std::string part_symbol = joint_symbol;
    if (joint_symbol != "<epsilon>" && joint_symbol != "<UNK>") {
      if (toks.size() != 2) {
        LOG(FATAL) << "Expected two parts for symbol: " << joint_symbol;
      }
      part_symbol = toks[side_id];
    }
    const int64 marginal_key = marginal_symbols.Find(part_symbol);
    if (marginal_key == fst::kNoSymbol) {
      LOG(FATAL) << "Failed to find: " << part_symbol;
    }
    joint2marginal->insert(std::make_pair(key, marginal_key));
  }
}

// Computes model (non-empirical, exact) conditional entropies.
void ModelConditionalEntropy(const NGramOutputHelper &model_joint,
                             const NGramOutputHelper &model_source,
                             const NGramOutputHelper &model_dest) {
  int source_joint_side = 0, dest_joint_side = 1;
  if (FST_FLAGS_source_is_phonemic) {
    source_joint_side = 1;  // phonemic
    dest_joint_side = 0;  // graphemic
  }
  std::map<StdArc::Label, StdArc::Label> joint2source;
  JointToMarginalLabelMapping(*model_joint.GetFst().InputSymbols(),
                              *model_source.GetFst().InputSymbols(),
                              source_joint_side, &joint2source);
  std::map<StdArc::Label, StdArc::Label> joint2dest;
  JointToMarginalLabelMapping(*model_joint.GetFst().InputSymbols(),
                              *model_dest.GetFst().InputSymbols(),
                              dest_joint_side, &joint2dest);

  std::vector<double> joint_state_probs;
  model_joint.CalculateStateProbs(&joint_state_probs, /* stationary= */false);
  CHECK_EQ(model_joint.NumStates(), joint_state_probs.size());

  std::vector<double> source_state_probs;
  model_source.CalculateStateProbs(&source_state_probs, /* stationary= */false);
  CHECK_EQ(model_source.NumStates(), source_state_probs.size());

  std::vector<double> dest_state_probs;
  model_dest.CalculateStateProbs(&dest_state_probs, /* stationary= */false);
  CHECK_EQ(model_dest.NumStates(), dest_state_probs.size());

  // Compute conditional entropies: H(Q|P) and H(P|Q), where P is the source
  // and Q is the destination distribution.
  double ce_logprob_source = 0.0, ce_logprob_dest = 0.0;
  double source_prob, dest_prob;
  for (uint64 joint_state_id = 0; joint_state_id < joint_state_probs.size();
       ++joint_state_id) {
    const double joint_prob = joint_state_probs[joint_state_id];
    const auto &joint_state_labels = model_joint.StateNGram(joint_state_id);
    if (joint_state_id != 0) {
      CHECK_EQ(1, joint_state_labels.size());
      const StdArc::Label joint_lab = joint_state_labels[0];
      const std::vector<StdArc::Label> source_labs = {
        joint2source[joint_lab] };
      const int64 source_state_id = model_source.NGramState(source_labs);
      source_prob = source_state_probs[source_state_id];
      const std::vector<StdArc::Label> dest_labs = { joint2dest[joint_lab] };
      const int64 dest_state_id = model_dest.NGramState(dest_labs);
      dest_prob = dest_state_probs[dest_state_id];
    } else {
      source_prob = source_state_probs[0];
      dest_prob = dest_state_probs[0];
    }
    ce_logprob_source += (joint_prob * std::log2(joint_prob / source_prob));
    ce_logprob_dest += (joint_prob * std::log2(joint_prob / dest_prob));
  }
  LOG(INFO) << "Conditional entropy: H(Q|P): " << -ce_logprob_source;
  LOG(INFO) << "Inverse conditional entropy: H(P|Q): " << -ce_logprob_dest;

  // Report everything + mutual information I(P, Q).
  const auto &joint_stats = StateEntropy(model_joint);
  const double mi = joint_stats.first + ce_logprob_source + ce_logprob_dest;
  const double norm = std::log2(joint_state_probs.size());
  std::cout << FST_FLAGS_info_header
            << " Conditional entropies: H(Q|P): " << -ce_logprob_source
            << " H(P|Q): " << -ce_logprob_dest
            << " H_n(Q|P): " << -ce_logprob_source / norm
            << " H_n(P|Q): " << -ce_logprob_dest / norm
            << " I(P,Q): " << mi
            << " I_n(P,Q): " << mi / norm
            << std::endl;
}

// Actual logic.
void Run() {
  const std::string context_pattern = FST_FLAGS_context_pattern;

  // Read joint model.
  const std::string ngram_joint_path = FST_FLAGS_ngram_joint_fst;
  LOG(INFO) << "Reading joint model from " << ngram_joint_path << " ...";
  std::unique_ptr<StdMutableFst> joint_ngram_fst(
      StdMutableFst::Read(ngram_joint_path, true));
  if (!joint_ngram_fst) {
    LOG(FATAL) << "Failed to load joint model from: " << ngram_joint_path;
  }
  NGramOutputHelper ngram_joint(joint_ngram_fst.get());
  if (ngram_joint.Error()) {
    LOG(FATAL) << "Failed to construct joint model!";
  }

  // Read source model and samples.
  const std::string ngram_source_path = FST_FLAGS_ngram_source_fst;
  LOG(INFO) << "Reading source model from " << ngram_source_path << " ...";
  std::unique_ptr<StdMutableFst> source_ngram_fst(
      StdMutableFst::Read(ngram_source_path, true));
  if (!source_ngram_fst) {
    LOG(FATAL) << "Failed to load source model from: " << ngram_source_path;
  }
  std::vector<std::unique_ptr<StdVectorFst>> source_samples =
      LoadSamples(FST_FLAGS_source_samples_far);
  NGramOutputHelper ngram_source(source_ngram_fst.get());
  if (ngram_source.Error()) {
    LOG(FATAL) << "Failed to construct source model!";
  }
  if (!ngram_source.Init(source_samples)) {
    LOG(FATAL) << "Failed to initialize source model!";
  }

  // Read destination model and samples.
  const std::string ngram_dest_path = FST_FLAGS_ngram_destination_fst;
  LOG(INFO) << "Reading destination model from " << ngram_dest_path << " ...";
  std::unique_ptr<StdMutableFst> dest_ngram_fst(
      StdMutableFst::Read(ngram_dest_path, true));
  if (!dest_ngram_fst) {
    LOG(FATAL) << "Failed to load destination model from: " << ngram_dest_path;
  }
  std::vector<std::unique_ptr<StdVectorFst>> dest_samples =
      LoadSamples(FST_FLAGS_destination_samples_far);
  NGramOutputHelper ngram_dest(dest_ngram_fst.get());
  if (ngram_dest.Error()) {
    LOG(FATAL) << "Failed to construct destination model!";
  }
  if (!ngram_dest.Init(dest_samples)) {
    LOG(FATAL) << "Failed to initialize destination model!";
  }

  // Compute empirical cross-entropy.
  LOG(INFO) << "Computing emprical cross-entropy ...";
  EmpiricalCrossEntropy(&ngram_source, source_samples,
                        &ngram_dest, dest_samples);

  // Compute exact model-based metrics.
  LOG(INFO) << "Computing model metrics ...";
  ModelEntropy(ngram_source, ngram_dest);
  ModelConditionalEntropy(ngram_joint, ngram_source, ngram_dest);
}

}  // namespace

int main(int argc, char **argv) {
  std::string usage = "Computes information-theoretic measures given the "
                      "data.\n\n  Usage: ";
  usage += argv[0];
  usage += " [--options]\n";
  std::set_new_handler(FailedNewHandler);
  SET_FLAGS(usage.c_str(), &argc, &argv, true);

  if (FST_FLAGS_ngram_joint_fst.empty()) {
    LOG(FATAL) << "Specify joint model (--ngram_joint_fst)!";
  }
  if (FST_FLAGS_ngram_source_fst.empty()) {
    LOG(FATAL) << "Specify source model (--ngram_source_fst)!";
  }
  if (FST_FLAGS_ngram_destination_fst.empty()) {
    LOG(FATAL) << "Specify destination model (--ngram_destination_fst)!";
  }

  Run();
  return 0;
}
