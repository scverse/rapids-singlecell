#pragma once

#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>

namespace gsea {

struct GseaInput {
    const float* values;  // (row, rank), sorted by descending feature value.
    const int* ranks;  // (row, gene), mapping original genes to sorted ranks.
    const int* targets;
    const int* starts;
    const int* sizes;
    const int* sources;
    // Use n_genes to retain all hits in general rows.
    const int* positive_counts;
    // (old rank, permutation) -> shuffled rank.
    const int* inverse_permutations;
    size_t n_rows;
    size_t n_genes;
    size_t n_sets;
    size_t n_permutations;
    size_t n_sources;
    // Row stride of the transposed permutation view.
    size_t permutation_stride;
};

// Replay the CPU's scalar arithmetic for observations and close comparisons.
// Zero-weight hits follow the positive prefix and cannot raise the tail.
template <typename Position>
__device__ double replay_running_sum(const float* values,
                                     const Position* hit_positions, int n_hits,
                                     int set_size, size_t n_genes,
                                     double total_weight) {
    const double decrement = 1.0 / (n_genes - set_size);
    double running_sum = 0.0;
    double positive_peak = 0.0;
    double negative_peak = 0.0;
    int position = 0;
    for (int i = 0; i < n_hits; ++i) {
        for (; position < hit_positions[i]; ++position) {
            running_sum -= decrement;
        }
        negative_peak = fmin(negative_peak, running_sum);
        running_sum +=
            fabs(static_cast<double>(values[hit_positions[i]])) / total_weight;
        positive_peak = fmax(positive_peak, running_sum);
        ++position;
    }
    const size_t remaining_misses = n_genes - position - (set_size - n_hits);
    const double tail_estimate = running_sum - remaining_misses * decrement;
    const double tail_error =
        4.0 * DBL_EPSILON * (remaining_misses + 2) *
        (fabs(running_sum) + remaining_misses * decrement + 1.0);
    const double score =
        positive_peak > -negative_peak ? positive_peak : negative_peak;
    if (tail_estimate - tail_error > -fabs(score)) {
        return score;
    }
    for (size_t i = 0; i < remaining_misses; ++i) {
        running_sum -= decrement;
    }
    negative_peak = fmin(negative_peak, running_sum);
    return positive_peak > -negative_peak ? positive_peak : negative_peak;
}

// Clear the sign bit without flushing a subnormal weight to zero.
__device__ float absolute_weight(float value) {
    return __uint_as_float(__float_as_uint(value) & 0x7fffffffu);
}

// Integer scales compare cumulative_weight*n_misses - missed*total_weight.
// Floating scales normalize each hit; close comparisons replay CPU divisions.
template <typename Arithmetic, typename Position>
__device__ __forceinline__ double score_sorted_hits(
    const float* values, const Position* hit_positions, int n_hits,
    int set_size, size_t n_genes, double total_weight, Arithmetic hit_scale,
    Arithmetic miss_scale, double normalization, double observed_score,
    bool force_exact) {
    if (!force_exact) {
        Arithmetic prefix_sum = 0;
        Arithmetic positive_peak = 0;
        Arithmetic negative_peak = 0;
        for (int i = 0; i < n_hits; ++i) {
            const Arithmetic miss_penalty = (hit_positions[i] - i) * miss_scale;
            negative_peak = min(negative_peak, prefix_sum - miss_penalty);
            prefix_sum += static_cast<Arithmetic>(
                              absolute_weight(values[hit_positions[i]])) *
                          hit_scale;
            positive_peak = max(positive_peak, prefix_sum - miss_penalty);
        }
        const double positive_score = positive_peak * normalization;
        const double negative_score = negative_peak * normalization;
        const double enrichment_score =
            positive_peak > -negative_peak ? positive_score : negative_score;
        const double tolerance = fmax(1e-12, 8.0 * DBL_EPSILON * n_genes);
        if (fabs(positive_score + negative_score) >= tolerance &&
            fabs(enrichment_score - observed_score) >= tolerance) {
            return enrichment_score;
        }
    }
    return replay_running_sum(values, hit_positions, n_hits, set_size, n_genes,
                              total_weight);
}

template <int Capacity, typename Position>
__device__ __forceinline__ double score_gene_set(
    GseaInput input, size_t row_index, int set_index, size_t permutation_index,
    double observed_score, bool force_exact) {
    const float* values = input.values + row_index * input.n_genes;
    const int* ranks = input.ranks + row_index * input.n_genes;
    const int set_size = input.sizes[set_index];
    Position hit_positions[Capacity];
    int n_hits = 0;
    int first_hit = static_cast<int>(input.n_genes);
    int last_hit = -1;
    float proof_weight_sum = 0.0f;
    for (int i = 0; i < set_size; ++i) {
        const int rank = ranks[input.targets[input.starts[set_index] + i]];
        const int position =
            input.inverse_permutations[rank * input.permutation_stride +
                                       permutation_index];
        if (position < input.positive_counts[row_index]) {
            hit_positions[n_hits++] = static_cast<Position>(position);
            first_hit = min(first_hit, position);
            last_hit = max(last_hit, position);
            proof_weight_sum += absolute_weight(values[position]);
        }
    }
    if (!n_hits) {
        return 0.0;
    }
    if (static_cast<size_t>(set_size) == input.n_genes) {
        for (int i = 0; i < n_hits; ++i) {
            if (__float_as_uint(values[hit_positions[i]]) & 0x7fffffffu) {
                return 1.0;
            }
        }
        return 0.0;
    }
    const size_t n_misses = input.n_genes - set_size;
    const double decrement = 1.0 / n_misses;
    const double tolerance = fmax(1e-12, 8.0 * DBL_EPSILON * input.n_genes);
    // Prove the positive peak occurs at the last hit before sorting any hits.
    // Its weight bounds remaining weight only within a sorted positive prefix.
    // Upward rounding and the sum margin cover roundoff and flushed subnormals.
    const double weight_sum_upper_bound =
        static_cast<double>(proof_weight_sum) *
            (1.0 + 2.0 * n_hits * FLT_EPSILON) +
        n_hits * static_cast<double>(FLT_MIN);
    const double span_penalty = (last_hit - first_hit - n_hits + 1) * decrement;
    const double last_hit_bound =
        __dmul_ru(__dadd_ru(span_penalty, tolerance), weight_sum_upper_bound);
    const double final_score = 1.0 - (last_hit - n_hits + 1) * decrement;
    if (!force_exact && input.positive_counts[row_index] < input.n_genes &&
        values[last_hit] > last_hit_bound &&
        1.0 - 2.0 * last_hit * decrement > tolerance &&
        fabs(final_score - observed_score) > tolerance) {
        return final_score;
    }
    for (int i = 1; i < n_hits; ++i) {
        const Position key = hit_positions[i];
        int j = i;
        while (j > 0 && hit_positions[j - 1] > key) {
            hit_positions[j] = hit_positions[j - 1];
            --j;
        }
        hit_positions[j] = key;
    }
    unsigned long long integer_total = 0;
    bool integer_weights = true;
    for (int i = 0; i < n_hits; ++i) {
        const float weight = absolute_weight(values[hit_positions[i]]);
        const unsigned int weight_bits = __float_as_uint(weight);
        const int missed = hit_positions[i] - i;
        // Float32 2^31 is the first positive value outside int32 conversion.
        if (weight_bits >= 0x4f000000u || missed < 0 ||
            static_cast<size_t>(missed) > n_misses) {
            integer_weights = false;
            break;
        }
        const int integer_weight = static_cast<int>(weight);
        if (__float_as_uint(static_cast<float>(integer_weight)) !=
            weight_bits) {
            integer_weights = false;
            break;
        }
        integer_total += static_cast<unsigned int>(integer_weight);
    }
    // The checked product bounds every integer intermediate, including misses.
    if (integer_weights && integer_total > 0 && integer_total <= INT32_MAX &&
        integer_total * n_misses <= INT32_MAX) {
        return score_sorted_hits(
            values, hit_positions, n_hits, set_size, input.n_genes,
            static_cast<double>(integer_total), static_cast<int>(n_misses),
            static_cast<int>(integer_total),
            1.0 / static_cast<double>(integer_total * n_misses), observed_score,
            force_exact);
    }
    double total_weight = 0.0;
    for (int i = 0; i < n_hits; ++i) {
        total_weight += fabs(static_cast<double>(values[hit_positions[i]]));
    }
    return total_weight == 0.0
               ? 0.0
               : score_sorted_hits(values, hit_positions, n_hits, set_size,
                                   input.n_genes, total_weight,
                                   1.0 / total_weight, decrement, 1.0,
                                   observed_score, force_exact);
}

// Observed scores use one thread per set; nulls use adjacent permutation lanes.
// All lanes reach the reductions, including padding in a partial final warp.
template <int Capacity, typename Position>
__global__ void sparse_kernel(GseaInput input, double* enrichment_scores,
                              double* null_sum, long long* same_count,
                              long long* extreme_count) {
    const size_t thread_stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    if (!null_sum) {
        for (size_t task =
                 static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
             task < input.n_rows * input.n_sources; task += thread_stride) {
            const size_t row_index = task / input.n_sources;
            const int set_index = input.sources[task % input.n_sources];
            enrichment_scores[row_index * input.n_sets + set_index] =
                score_gene_set<Capacity, Position>(input, row_index, set_index,
                                                   0, 0.0, true);
        }
        return;
    }
    const int lane = threadIdx.x % 32;
    const size_t permutation_chunks = (input.n_permutations + 31) / 32;
    for (size_t task = static_cast<size_t>(blockIdx.x) * (blockDim.x / 32) +
                       threadIdx.x / 32;
         task < input.n_rows * input.n_sources * permutation_chunks;
         task += thread_stride / 32) {
        const size_t permutation_index =
            (task % permutation_chunks) * 32 + lane;
        const size_t row_index = task / (permutation_chunks * input.n_sources);
        const int set_index =
            input.sources[(task / permutation_chunks) % input.n_sources];
        const size_t output_index = row_index * input.n_sets + set_index;
        const double observed_score = enrichment_scores[output_index];
        if (observed_score == 0.0) {
            continue;
        }
        const bool valid_permutation = permutation_index < input.n_permutations;
        const double null_score =
            valid_permutation ? score_gene_set<Capacity, Position>(
                                    input, row_index, set_index,
                                    permutation_index, observed_score, false)
                              : 0.0;
        unsigned int same_sign_count =
            valid_permutation &&
            (observed_score >= 0.0 ? null_score >= 0.0 : null_score < 0.0);
        unsigned int extreme_count_local =
            same_sign_count &&
            (observed_score >= 0.0 ? null_score >= observed_score
                                   : null_score <= observed_score);
        double null_sum_local = same_sign_count ? fabs(null_score) : 0.0;
#pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            null_sum_local +=
                __shfl_down_sync(0xffffffffu, null_sum_local, offset);
            same_sign_count +=
                __shfl_down_sync(0xffffffffu, same_sign_count, offset);
            extreme_count_local +=
                __shfl_down_sync(0xffffffffu, extreme_count_local, offset);
        }
        if (lane == 0) {
            if (null_sum_local > 0.0) {
                atomicAdd(null_sum + output_index, null_sum_local);
            }
            if (same_sign_count) {
                atomicAdd(reinterpret_cast<unsigned long long*>(same_count +
                                                                output_index),
                          same_sign_count);
            }
            if (extreme_count_local) {
                atomicAdd(reinterpret_cast<unsigned long long*>(extreme_count +
                                                                output_index),
                          extreme_count_local);
            }
        }
    }
}

// Oversized sets use an exact scalar walk without a per-task hit allocation.
__global__ void dense_kernel(const float* sorted_values,
                             const int* sorted_order, const bool* membership,
                             const int* forward_permutations, size_t n_genes,
                             size_t n_sets, size_t n_permutations,
                             size_t n_tasks, double* enrichment_scores) {
    for (size_t task =
             static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         task < n_tasks; task += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float* row_values =
            sorted_values + (task / (n_permutations * n_sets)) * n_genes;
        const int* gene_order =
            sorted_order + (task / (n_permutations * n_sets)) * n_genes;
        const bool* members = membership + (task % n_sets) * n_genes;
        const int* permutation =
            forward_permutations + ((task / n_sets) % n_permutations) * n_genes;
        double total_weight = 0.0;
        size_t n_hits = 0;
        for (size_t k = 0; k < n_genes; ++k) {
            if (members[gene_order[permutation[k]]]) {
                total_weight += fabs(static_cast<double>(row_values[k]));
                ++n_hits;
            }
        }
        double running_sum = 0.0;
        double positive_peak = 0.0;
        double negative_peak = 0.0;
        const double decrement =
            n_hits < n_genes ? 1.0 / (n_genes - n_hits) : 0.0;
        if (total_weight > 0.0) {
            for (size_t k = 0; k < n_genes; ++k) {
                running_sum += members[gene_order[permutation[k]]]
                                   ? fabs(static_cast<double>(row_values[k])) /
                                         total_weight
                                   : -decrement;
                positive_peak = fmax(positive_peak, running_sum);
                negative_peak = fmin(negative_peak, running_sum);
            }
        }
        enrichment_scores[task] =
            positive_peak > -negative_peak ? positive_peak : negative_peak;
    }
}

}  // namespace gsea
