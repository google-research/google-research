/* Python extension module for fast quartic-fermionic-term accumulation.

  Terminology: The left and right <psi_...|gamma^...|psi_...> of a
  quartic term <psi_...|gamma^...|psi_...> <psi_...|gamma^...|psi_...>
  are called the "left (fermionic) bilinear" and "right (fermionic)
  bilinear".

  Compile With:
  g++ -I /usr/include/python3.10 -fPIC -O3 -mavx -shared -o psi4_accum.so psi4_accum.cpp
  (will require removing -mavx on CPUs that do not have AVX).

  ;; Emacs [C-x C-e]-able block for recompilation:
  (save-excursion
    (beginning-of-buffer)
    (re-search-forward "Compile With:\n *\\(.*\\)")
    (let ((command (buffer-substring (match-beginning 1)
                                     (match-end 1))))
      (shell-command command)))

  The code for sorting four items via nested comparisons was taken from:
  https://arxiv.org/format/hep-th/0412331 (=> Download source):
  src.tar.bz2:src/helpers/meta-loopless.lhs
*/

#define DEBUG 0

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


#include <Python.h>
#include <numpy/arrayobject.h>


#include <cstdint>
#include <functional>
#include <vector>


namespace {

// D=1+10 -> D=1+0 reduction of supergravity leaves us with 11x32
// psi_IA gravitino field components, but "gamma-tracelessness" of the
// vector-spinor allows us to eliminate 32. In the de-symbolized
// calculation, we identify these with "tags" [1..320] (in lexical order),
// with vector-indices starting at 1 (i.e. excluding "time-coordinate" 0)
// and spinor-indices starting at 0, so tag 1 == psi_I=1,A=0,
// tag 2 == psi_I=1,A=1, tag 320 == psi_I=10,A=31. A negative tag
// indicates that the variable comes with a factor -1. A field's "index"
// is one less than the absolute value of the tag, so [0..319].
const int TAG_MAX = 320;

// Orders four numbers and returns the sign of the permutation used to
// order items, or 0 if any two items were the same (in which case
// there is no attempt to order items).
// Machine-generated code, as for hep-th/0412331.
inline int sort4(int* n0123_in, int* n_out) {
  int b1 = n0123_in[0], b2 = n0123_in[1], b3 = n0123_in[2], b4 = n0123_in[3];
  if (b1 < b3) {
    if (b2 < b4) {
      if (b1 < b2) {
        if (b3 < b2) {
          n_out[0] = b1, n_out[1] = b3, n_out[2] = b2, n_out[3] = b4;
          return -1;  // 1324
        } else {
          if (b3 < b4) {
            n_out[0] = b1, n_out[1] = b2, n_out[2] = b3, n_out[3] = b4;
            return +1;  // 1234
          } else {
            n_out[0] = b1, n_out[1] = b2, n_out[2] = b4, n_out[3] = b3;
            return -1;  // 1243
          }
        }
      } else {
        if (b1 < b4) {
          if (b3 < b4) {
            n_out[0] = b2, n_out[1] = b1, n_out[2] = b3, n_out[3] = b4;
            return -1;  // 2134
          } else {
            n_out[0] = b2, n_out[1] = b1, n_out[2] = b4, n_out[3] = b3;
            return +1;  // 2143
          }
        } else {
          n_out[0] = b2, n_out[1] = b4, n_out[2] = b1, n_out[3] = b3;
          return -1;  // 2413
        }
      }
    } else {
      if (b1 < b4) {
        if (b3 < b4) {
          n_out[0] = b1, n_out[1] = b3, n_out[2] = b4, n_out[3] = b2;
          return +1;  // 1342
        } else {
          if (b3 < b2) {
            n_out[0] = b1, n_out[1] = b4, n_out[2] = b3, n_out[3] = b2;
            return -1;  // 1432
          } else {
            n_out[0] = b1, n_out[1] = b4, n_out[2] = b2, n_out[3] = b3;
            return +1;  // 1423
          }
        }
      } else {
        if (b1 < b2) {
          if (b3 < b2) {
            n_out[0] = b4, n_out[1] = b1, n_out[2] = b3, n_out[3] = b2;
            return +1;  // 4132
          } else {
            n_out[0] = b4, n_out[1] = b1, n_out[2] = b2, n_out[3] = b3;
            return -1;  // 4123
          }
        } else {
          n_out[0] = b4, n_out[1] = b2, n_out[2] = b1, n_out[3] = b3;
          return +1;  // 4213
        }
      }
    }
  } else {
    if (b2 < b4) {
      if (b3 < b2) {
        if (b1 < b2) {
          n_out[0] = b3, n_out[1] = b1, n_out[2] = b2, n_out[3] = b4;
          return +1;  // 3124
        } else {
          if (b1 < b4) {
            n_out[0] = b3, n_out[1] = b2, n_out[2] = b1, n_out[3] = b4;
            return -1;  // 3214
          } else {
            n_out[0] = b3, n_out[1] = b2, n_out[2] = b4, n_out[3] = b1;
            return +1;  // 3241
          }
        }
      } else {
        if (b3 < b4) {
          if (b1 < b4) {
            n_out[0] = b2, n_out[1] = b3, n_out[2] = b1, n_out[3] = b4;
            return +1;  // 2314
          } else {
            n_out[0] = b2, n_out[1] = b3, n_out[2] = b4, n_out[3] = b1;
            return -1;  // 2341
          }
        } else {
          n_out[0] = b2, n_out[1] = b4, n_out[2] = b3, n_out[3] = b1;
          return +1;  // 2431
        }
      }
    } else {
      if (b3 < b4) {
        if (b1 < b4) {
          n_out[0] = b3, n_out[1] = b1, n_out[2] = b4, n_out[3] = b2;
          return -1;  // 3142
        } else {
          if (b1 < b2) {
            n_out[0] = b3, n_out[1] = b4, n_out[2] = b1, n_out[3] = b2;
            return +1;  // 3412
          } else {
            n_out[0] = b3, n_out[1] = b4, n_out[2] = b2, n_out[3] = b1;
            return -1;  // 3421
          }
        }
      } else {
        if (b3 < b2) {
          if (b1 < b2) {
            n_out[0] = b4, n_out[1] = b3, n_out[2] = b1, n_out[3] = b2;
            return -1;  // 4312
          } else {
            n_out[0] = b4, n_out[1] = b3, n_out[2] = b2, n_out[3] = b1;
            return +1;  // 4321
          }
        } else {
          n_out[0] = b4, n_out[1] = b2, n_out[2] = b3, n_out[3] = b1;
          return -1;  // 4231
        }
      }
    }
  }
}



// Basic processing only requires four nested loops, two for the
// left-fermionic bilinear, and two for the right. Each pair of nested
// loops has a vector-spacetime-index loop that either iterates over 1
// or 10 indices, depending on whether we are processing psi_I=0,A,
// which is a sum of ten contributions determined by the gauge
// condition, and a spinor-index loop, which iterates over 32 values.
//
// As we determine each of the psi-tags that enter our quartic term,
// we may in general have to undergo simple linear substitution of
// these psis. This is done in-situ, but since nesting eight loops
// (one more to go through each possible expansion of each psi
// would have looked visually cluttered and hard to follow, we instead
// abstract out the substitution-expansion into a callable which itself
// calls a loop-body anonymous-function that is a by-reference closure
// over all the relevant inner context.
//
// This is the type of the loop-body function. Due to using
// reference-closures to access context, this is void-to-void.
typedef std::function < void()> loop_body_fn;

// Returns `True` on successful processing, `False` if there was an
// error (such as: input data not in expected form, out-of-range
// tags, etc.)
//
// Note: the body of this function looks lengthy and scary, but really
// is very straightforward: Python arg-parsing,
// setting up the linear-substitution mapping if substitution is needed,
// then nested looping that generates the four factors.
// At the center of it, coefficient-accumulation.
// Since these different steps do need to have the processing-context,
// not much can be gained here in clarity by splitting up the body.
PyObject* collect_quartic_terms(PyObject* self, PyObject* args) {
  // Substutition-info. `None` if no linear substitution is to
  // be done, tuple describing the substitution otherwise.
  PyObject* po_subs_info;
  // The raw python object that will be checked to contain
  // the coefficients of the ordered-4-tuple-ranking polynomial.
  PyObject* po_q4_ranking_polynomial_denom_1dc2b3a4;
  // Output-accumulator.
  PyObject* po_accum_out;
  // Data buffers.
  // b0 refers to "left-bilinear", b1 to "right-bilinear",
  // le/ri are the left and right side spinor on the bilinear.
  // So, a quartic contribution has structure <b0_le | b0_ri> <b1_le | b1_ri>.
  // Argument order is: b0le, b0ri, b1le, b1ri
  PyObject* po_b01_le_ri[4];
  int b01_le_ri_index[4];
  float term_coefficient;
  // Optional substitution-mapping. Initialized later if we use substitution.
  int max_mapping_num_summands;
  std::vector<int> subs_index_by_psi_index_and_num_summand;
  std::vector<float> coeff_by_psi_index_and_num_summand;

  // It is no problem that this helper has some seriously
  // complex call-interface, since (a) calls are "comparatively rare",
  // and (b) this is internal; the user does not need to care about.
  if (!PyArg_ParseTuple(
          args,
          "OOOOOOOiiiif",  // "OOf" ;-).
          &po_subs_info,  // optional substitution-info.
          &po_q4_ranking_polynomial_denom_1dc2b3a4,  // Quadruple ranking.
          &po_accum_out,  // Accumulator for collecting coefficients.
          // First/second bilinear, left/right factor.
          &po_b01_le_ri[0], &po_b01_le_ri[1],
          &po_b01_le_ri[2], &po_b01_le_ri[3],
          &b01_le_ri_index[0], &b01_le_ri_index[1],
          &b01_le_ri_index[2], &b01_le_ri_index[3],
          &term_coefficient)) {
    return nullptr;
  }
  if (term_coefficient == 0.0) {
    // Can early exit on any "gets multiplied with a factor zero" problem.
    Py_RETURN_TRUE;
  }
  // With any FFI, we still always be careful and check whether the
  // data we received matches expectations: Providing bad data to a
  // foreign-function-interface call should not ever allow us to crash
  // the process. We here merely return `None` on any invalid data.
  if (!(PyArray_Check(po_q4_ranking_polynomial_denom_1dc2b3a4) &&
        PyArray_Check(po_accum_out) &&
        PyArray_Check(po_b01_le_ri[0]) &&
        PyArray_Check(po_b01_le_ri[1]) &&
        PyArray_Check(po_b01_le_ri[2]) &&
        PyArray_Check(po_b01_le_ri[3]))) {
    Py_RETURN_FALSE;
  }
  // We now know that the arguments we captured as PyObject are indeed
  // array-objects.
  PyArrayObject* accum = reinterpret_cast<PyArrayObject*>(po_accum_out);
  PyArrayObject* b01_le_ri[4];
  for (int i = 0; i < 4; ++i) {
    b01_le_ri[i] = reinterpret_cast<PyArrayObject*>(po_b01_le_ri[i]);
  }
  // Expectation: accumulator holds binary32 IEEE754 float entries,
  // in machine byte order.
  if (!(PyArray_NDIM(accum) == 1 &&
        PyArray_ISFLOAT(accum) &&
        PyArray_ISNOTSWAPPED(accum) &&
        PyArray_ITEMSIZE(accum) == sizeof(float))) {
    Py_RETURN_FALSE;
  }
  npy_intp accum_size = PyArray_DIM(accum, 0);  // For bounds checking.
  // Likewise, we check the data type and shape of the tag-arrays.
  for (int k = 0; k < 4; ++k) {
    if (!(0 <= b01_le_ri_index[k] &&
          b01_le_ri_index[k] < 11 &&
          PyArray_ISSIGNED(b01_le_ri[k]) &&
          PyArray_NDIM(b01_le_ri[k]) == 2 &&
          PyArray_DIM(b01_le_ri[k], 0) == 11 &&
          PyArray_DIM(b01_le_ri[k], 1) == 32
          )) {
      Py_RETURN_FALSE;
    }
  }

  // Processing the ranking polynomial that maps four indices in
  // lexicographic order to a linear accumulator-index.
  // The specific polynomial will have to be picked to align
  // with the number of variables retained post-linear-substitution.
  // One finds that these ranking polynomials are not generic
  // 4th-order polynomials in four indices, but do not have mixed terms.
  // This code hence only supports the no-mixed-terms form.
  // Since we use this polynomial often, we PyArray_GETPTR1()
  // its entries once and then use that.
  PyArrayObject* q4_ranking_polynomial =
    reinterpret_cast<PyArrayObject*>(po_q4_ranking_polynomial_denom_1dc2b3a4);
  int64_t ranking_polynomial_data[12];
  if (!(PyArray_NDIM(q4_ranking_polynomial) == 1 &&
        PyArray_DIM(q4_ranking_polynomial, 0) == 12 &&
        PyArray_ISSIGNED(q4_ranking_polynomial) &&
        PyArray_ITEMSIZE(q4_ranking_polynomial) == sizeof(int64_t) &&
        PyArray_ISNOTSWAPPED(q4_ranking_polynomial))) {
    Py_RETURN_FALSE;
  }
  for (int i = 0; i < 12; ++i) {
    ranking_polynomial_data[i] =
      *reinterpret_cast<int64_t*>PyArray_GETPTR1(q4_ranking_polynomial, i);
  }
  // Rank an ordered 0-based-indices 4-tuple.
  auto fn_rank4 = [&ranking_polynomial_data](int* abcd)
    -> int64_t {
    const auto& r = ranking_polynomial_data;
    int64_t a  = abcd[0], b = abcd[1], c = abcd[2], d = abcd[3];
    int64_t aa = a * a, bb = b*b, cc = c*c;
    // Int64-array, entries: denominator, 1-coeff, d-coeff, c-coeff,
    // c**2-coeff, b-coeff, b**2-coeff, b**3-coeff, a-coeff,
    // a**2-coeff, a**3-coeff, a**4-coeff.
    return (r[1] + d*r[2] + c*r[3] + cc*r[4] + b*r[5] + bb*r[6] + bb*b*r[7] +
            a*r[8] + aa*r[9] + aa*a*r[10] + aa*aa*r[11]) / r[0];
  };
  // If a bilinear's left-index is == 0, this means we have to
  // iterate 1..10 (inclusive). If it is k>0, we have to iterate
  // over ONLY the value zero.
  int b0_le_start, b0_le_end, b1_le_start, b1_le_end;
  if (b01_le_ri_index[0] == 0) {
    b0_le_start = 1;
    b0_le_end = 11;
  } else {
    b0_le_start = 0;
    b0_le_end = 1;
  }
  if (b01_le_ri_index[2] == 0) {
    b1_le_start = 1;
    b1_le_end = 11;
  } else {
    b1_le_start = 0;
    b1_le_end = 1;
  }
  int q4_index;

  // There are two main cases: Either we are being asked to do linear
  // substitutions, or not. We might fold the "no substitutions" case
  // into the "with substitutions" case, but there are two reasons not
  // to do so:
  //
  // 1. The code to do this also has some complexity, which must be
  //    weighed against the complexity (and also benefits) of having a
  //    simpler (and more understandable, hence more accessible) 2nd
  //    implementation of our main loop.
  // 2. The extra indirection is avoidable deadweight that damages
  //    performance.
  // We hence go for two separate main loops, one with substitution,
  // one without.
  // Note: Indices are 0 <= num_factor < TAG_MAX here.
  int t4_indices[4];
  int t4_indices_final[4];  // post-substitution, sorted.

  // We incrementally propagate factors as we add more terms.
  // (If no substitution is requested, the scheme below leapfrogs
  // every other entry.)
  // - factor_so_far[0] is the term-coefficient.
  // - factor_so_far[1] = factor_so_far[0] * psi_term_factor[0]
  // - factor_so_far[2] = factor_so_far[1] * psi0_subs_term_factor
  // - factor_so_far[3] = factor_so_far[2] * psi_term_factor[1]
  // - factor_so_far[4] = factor_so_far[3] * psi1_subs_term_factor
  // ...and so on.
  // When incrementing an accumulator-entry, we read off factor_so_far[8]
  // (and still need to take a re-ordering sign into account).
  float factor_so_far[9];
  factor_so_far[0] = term_coefficient;

  if (po_subs_info == Py_None) {
    //// Case: No substitution requested.
    for (int b0_le = b0_le_start; b0_le < b0_le_end; ++b0_le) {
      for (int b0s = 0; b0s < 32; ++b0s) {
        int16_t tag_b0_le = *static_cast<int16_t*>(
          PyArray_GETPTR2(b01_le_ri[0], b0_le, b0s));
        int abs_tag0 = abs(tag_b0_le);
        // In principle, we should range-check the tag,
        // but since we do range-check any index derived
        // from this tag later anyhow, bad data may at worst
        // lead to bad results, but cannot cause memory-corruption.
        // Hence, we do not do this:
        //   if (abs_tag0 > TAG_MAX) Py_RETURN_FALSE;
        // Likewise for other tags.
        t4_indices[0] = abs_tag0 - 1;
        factor_so_far[2] = tag_b0_le == abs_tag0?
          factor_so_far[0] : -factor_so_far[0];
        int16_t tag_b0_ri = *static_cast<int16_t*>(
            PyArray_GETPTR2(b01_le_ri[1], b01_le_ri_index[1], b0s));
        int abs_tag1 = abs(tag_b0_ri);
        t4_indices[1] = abs_tag1 - 1;
        if (t4_indices[1] == t4_indices[0]) {  // Duplicate psi.
          continue;
        }
        factor_so_far[4] = tag_b0_ri == abs_tag1?
          factor_so_far[2] : -factor_so_far[2];
        for (int b1_le = b1_le_start; b1_le < b1_le_end; ++b1_le) {
          for (int b1s = 0; b1s < 32; ++b1s) {
            int16_t tag_b1_le = *static_cast<int16_t*>(
                PyArray_GETPTR2(b01_le_ri[2], b1_le, b1s));
            int abs_tag2 = abs(tag_b1_le);
            t4_indices[2] = abs_tag2 - 1;
            if (t4_indices[2] == t4_indices[0] ||
                t4_indices[2] == t4_indices[1]) {  // Duplicate psi.
              continue;
            }
            factor_so_far[6] = tag_b1_le == abs_tag2?
              factor_so_far[4] : -factor_so_far[4];
            int16_t tag_b1_ri = *static_cast<int16_t*>(
                PyArray_GETPTR2(b01_le_ri[3], b01_le_ri_index[3], b1s));
            int abs_tag3 = abs(tag_b1_ri);
            t4_indices[3] = abs_tag3 - 1;
            if (t4_indices[3] == t4_indices[0] ||
                t4_indices[3] == t4_indices[1] ||
                t4_indices[3] == t4_indices[2]) {  // Duplicate psi.
              continue;
            }
            factor_so_far[8] = tag_b1_ri == abs_tag3?
              factor_so_far[6] : -factor_so_far[6];
            float contrib = factor_so_far[8] * sort4(t4_indices,
                                                     t4_indices_final);
            int accum_index = fn_rank4(t4_indices_final);
            if (!(0 <= accum_index && accum_index < accum_size)) {
              Py_RETURN_FALSE;
            }
            *reinterpret_cast<float*>(
                PyArray_GETPTR1(accum, accum_index)) += contrib;
          }}}}
    Py_RETURN_TRUE;
  }

  //// Case: Linear substitution required.
  // ps == post-substitution.
  int ps_t4_indices[4];

  // We need a way to iterate over all substitutions for the k-th
  // factor. This will be a void->void function. Due to call-nesting,
  // we lose the ability to early-return and instead have to keep
  // track of an error-condition.
  //
  // Note: comparison with earlier, simpler versions of this code that
  // did not support linear substitution do seem to suggest that we
  // may be able to reclaim a speed factor 4 or so by more careful
  // coding that mostly avoids taking closures (and in particular
  // std::function). Given that the full calculation still takes only
  // some minutes on a single core on a laptop, this is not considered
  // worth the effort here.
  bool error_occurred = false;
  std::function<void(int, loop_body_fn)> fn_doall_subs_kth;
  // First of all, we need to parse the substitution-info.
  // This may well fail.
  //
  // Substitution-info data format is:
  // (subs_index_by_psi_index, subs_coefficients)
  // where subs_index_by_psi_index is 0-based, [320, nmax], and
  // subs_coefficients is [320, nmax]. The rule is:
  // - For a given index, we scan until either nmax,
  // or we hit a substitution-coefficient of zero.
  PyObject* po_subs_mapping;
  PyObject* po_subs_coeffs;
  if (!PyArg_ParseTuple(po_subs_info, "OO",
                        &po_subs_mapping,
                        &po_subs_coeffs)) {
    Py_RETURN_FALSE;
  }
  PyArrayObject* subs_mapping =
    reinterpret_cast<PyArrayObject*>(po_subs_mapping);
  PyArrayObject* subs_coeffs =
    reinterpret_cast<PyArrayObject*>(po_subs_coeffs);
  // Arrays must be 2-index, with .shape[0] == TAG_MAX,
  // the index-array being not-byteswapped numpy.int16,
  // and the coefficient-array having compatible shape,
  // being not-byteswapped float32.
  if (!(PyArray_NDIM(subs_mapping) == 2 &&
        PyArray_DIM(subs_mapping, 0) == TAG_MAX &&
        PyArray_ISSIGNED(subs_mapping) &&
        PyArray_ITEMSIZE(subs_mapping) == sizeof(int16_t) &&
        PyArray_ISNOTSWAPPED(subs_mapping) &&
        PyArray_NDIM(subs_coeffs) == 2 &&
        PyArray_DIM(subs_coeffs, 0) == TAG_MAX &&
        PyArray_DIM(subs_coeffs, 1) == PyArray_DIM(subs_mapping, 1) &&
        PyArray_ISFLOAT(subs_coeffs) &&
        PyArray_ISNOTSWAPPED(subs_coeffs) &&
        PyArray_ITEMSIZE(subs_coeffs) == sizeof(float))) {
    Py_RETURN_FALSE;
  }
  // Rather than going through PyArray_GETPTR2() array-access macros
  // over and over again, we transfer substitution-information
  // to C++ vectors once and only once.
  max_mapping_num_summands = PyArray_DIM(subs_coeffs, 1);
  for (int s = 0; s < max_mapping_num_summands; ++s) {
    for (int i = 0; i < TAG_MAX; i++) {
      subs_index_by_psi_index_and_num_summand.push_back(
        *static_cast<int16_t*>(PyArray_GETPTR2(subs_mapping, i, s)));
      coeff_by_psi_index_and_num_summand.push_back(
        *static_cast<float*>(PyArray_GETPTR2(subs_coeffs, i, s)));
    }
  }
  // We now have all the data set up and fast-accessible for doing
  // linear substitution. Relevant context is referenced via a
  // C++11 by-reference closure.
  fn_doall_subs_kth =
    ([&](int k, loop_body_fn f_body) {
      if (error_occurred) return;  // propagate error up.
      int idx = t4_indices[k];
      for (int s = 0; s < max_mapping_num_summands; ++s) {
        int i_lookup = s * TAG_MAX + idx;
        float map_factor = coeff_by_psi_index_and_num_summand[i_lookup];
        if (map_factor == 0.0) {
          // Early-exit: hit zero-coefficient on a ragged-table, row is done.
          break;
        }
        float propagated_factor = factor_so_far[2*k+1] * map_factor;
        factor_so_far[2*k+2] = propagated_factor;
        int map_index = subs_index_by_psi_index_and_num_summand[i_lookup];
        ps_t4_indices[k] = map_index;
        // If the index equals any earlier one, anything here is zero,
        // and we are done.
        for (int kk = 0; kk < k; ++kk) {
          if (ps_t4_indices[kk] == map_index) {
            goto cont_s_loop;  // continue with next iteration on outer s-loop.
          }
        }
        f_body();
      cont_s_loop:
        // No-op. C++ syntactically needs a statement after the label.
        // Clang++ complains if this is e.g. `true;`.
        do {} while (0);
      }
    });

  // Our main loop, with substitutions.
  for (int b0_le = b0_le_start; b0_le < b0_le_end; ++b0_le) {
    for (int b0s = 0; b0s < 32; ++b0s) {
      int16_t tag_b0_le = *static_cast<int16_t*>(
          PyArray_GETPTR2(b01_le_ri[0], b0_le, b0s));
      int abs_tag = abs(tag_b0_le);
      // Here, we actually should range-check tags, since these are used
      // as indices into substitution-tables. The NumPy reference does
      // not promise PyArray_GETPTR*() calls to do range-checking.
      if (abs_tag > TAG_MAX) {
        error_occurred = true;
        break;  // break on outermost loop.
      }
      t4_indices[0] = abs_tag - 1;
      factor_so_far[1] = tag_b0_le == abs_tag?
        factor_so_far[0] : -factor_so_far[0];
      fn_doall_subs_kth(0, [&](){
        int16_t tag_b0_ri = *static_cast<int16_t*>(
            PyArray_GETPTR2(b01_le_ri[1], b01_le_ri_index[1], b0s));
        int abs_tag = abs(tag_b0_ri);
        if (abs_tag > TAG_MAX) {
          error_occurred = true;
          return;  // return from closure!
        }
        t4_indices[1] = abs_tag - 1;
        factor_so_far[3] = tag_b0_ri == abs_tag?
          factor_so_far[2] : -factor_so_far[2];
        fn_doall_subs_kth(1, [&](){
          for (int b1_le = b1_le_start; b1_le < b1_le_end; ++b1_le) {
            for (int b1s = 0; b1s < 32; ++b1s) {
              int16_t tag_b1_le = *static_cast<int16_t*>(
                  PyArray_GETPTR2(b01_le_ri[2], b1_le, b1s));
              int abs_tag = abs(tag_b1_le);
              if (abs_tag > TAG_MAX) {
                error_occurred = true;
                return;
              }
              t4_indices[2] = abs_tag - 1;
              factor_so_far[5] = tag_b1_le == abs_tag?
                factor_so_far[4] : -factor_so_far[4];
              fn_doall_subs_kth(2, [&](){
                int16_t tag_b1_ri = *static_cast<int16_t*>(
                  PyArray_GETPTR2(b01_le_ri[3], b01_le_ri_index[3], b1s));
                int abs_tag = abs(tag_b1_ri);
                if (abs_tag > TAG_MAX) {
                  error_occurred = true;
                  return;
                }
                t4_indices[3] = abs_tag - 1;
                factor_so_far[7] = tag_b1_ri == abs_tag?
                  factor_so_far[6] : -factor_so_far[6];
                fn_doall_subs_kth(3, [&](){
                  float contrib = factor_so_far[8] * sort4(ps_t4_indices,
                                                           t4_indices_final);
                  int accum_index = fn_rank4(t4_indices_final);
                  if (!(0 <= accum_index && accum_index < accum_size)) {
                    error_occurred = true;
                    return;
                  }
                  // Q: Would we perhaps want to change the API to return
                  // -1 on error, and the number of contributions summed
                  // otherwise? This might in some situations be useful.
                  *reinterpret_cast<float*>(
                    PyArray_GETPTR1(accum, accum_index)) += contrib;
                });
              });
            }}});
      });
    }}
  // We are done!
  if (error_occurred) {
    Py_RETURN_FALSE;
  } else {
    Py_RETURN_TRUE;
  }
}


static PyMethodDef methods[] = {
  {"collect_quartic_terms", (PyCFunction)collect_quartic_terms, METH_VARARGS,
   "Collects quartic terms."},
  {NULL, NULL, 0, NULL}  // Sentinel.
};

static struct PyModuleDef module_def = {
  PyModuleDef_HEAD_INIT,
  "psi4_accum",
  NULL,
  -1,
  methods
};

}  // namespace


PyMODINIT_FUNC PyInit_psi4_accum(void) {
  import_array();  // Note: NumPy using code needs this!
  if (PyErr_Occurred()) {
    return nullptr;
  }
  return PyModule_Create(&module_def);
}
