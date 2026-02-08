#include "bootstrapping.cuh"

#include "common.h"
#include "ntt.cuh"
#include "uintmodmath.cuh"

using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

namespace {

    inline std::uint64_t ceil_div_uint64(std::uint64_t numerator, std::uint64_t denominator) {
        return (numerator + denominator - 1) / denominator;
    }

    __global__ void mod_raise_centered_kernel(const std::uint64_t *input,
                                              std::uint64_t *output,
                                              const DModulus *modulus,
                                              std::size_t n,
                                              std::size_t q_size) {
        const std::size_t total = n * q_size;
        for (std::size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < total; tid += blockDim.x * gridDim.x) {
            const std::size_t prime_index = tid / n;
            const std::size_t coeff_index = tid % n;

            const std::uint64_t q0 = modulus[0].value();
            const std::uint64_t qi = modulus[prime_index].value();
            const std::uint64_t coeff = input[coeff_index];

            if (prime_index == 0) {
                output[tid] = coeff;
                continue;
            }

            const std::uint64_t threshold = q0 >> 1;
            if (coeff >= threshold) {
                const std::uint64_t centered = q0 - coeff;
                const std::uint64_t reduced =
                        barrett_reduce_uint64_uint64(centered, qi, modulus[prime_index].const_ratio()[1]);
                output[tid] = (reduced == 0) ? 0 : (qi - reduced);
            } else {
                output[tid] = barrett_reduce_uint64_uint64(coeff, qi, modulus[prime_index].const_ratio()[1]);
            }
        }
    }

    void validate_mod_up_inputs(const PhantomContext &context, const PhantomCiphertext &ciphertext) {
        if (context.get_context_data(context.get_first_index()).parms().scheme() != scheme_type::ckks) {
            throw std::invalid_argument("mod_up_from_q0 only supports CKKS");
        }

        const std::size_t max_chain_index = context.total_parm_size() - 1;
        if (ciphertext.chain_index() != max_chain_index) {
            throw std::invalid_argument("mod_up_from_q0 expects ciphertext at the last chain index");
        }

        if (ciphertext.coeff_modulus_size() != 1) {
            throw std::invalid_argument("mod_up_from_q0 expects a single-modulus ciphertext");
        }

        if (ciphertext.size() < 2) {
            throw std::invalid_argument("ciphertext size is invalid");
        }
    }

} // namespace

namespace phantom {

    PhantomCiphertext mod_up_from_q0(const PhantomContext &context, const PhantomCiphertext &ciphertext) {
        validate_mod_up_inputs(context, ciphertext);

        const auto &stream = cudaStreamPerThread;
        const auto &target_context_data = context.get_context_data(context.get_first_index());
        const auto &target_parms = target_context_data.parms();
        const std::size_t coeff_count = target_parms.poly_modulus_degree();
        const std::size_t target_coeff_modulus_size = target_parms.coeff_modulus().size();

        auto input_coeff = make_cuda_auto_ptr<std::uint64_t>(ciphertext.size() * coeff_count, stream);

        for (std::size_t poly_index = 0; poly_index < ciphertext.size(); poly_index++) {
            const auto *src_poly = ciphertext.data() + poly_index * coeff_count;
            auto *dst_poly = input_coeff.get() + poly_index * coeff_count;
            cudaMemcpyAsync(dst_poly, src_poly, coeff_count * sizeof(std::uint64_t), cudaMemcpyDeviceToDevice, stream);
            if (ciphertext.is_ntt_form()) {
                nwt_2d_radix8_backward_inplace(dst_poly, context.gpu_rns_tables(), 1, 0, stream);
            }
        }

        PhantomCiphertext destination;
        destination.resize(context, context.get_first_index(), ciphertext.size(), stream);
        destination.set_ntt_form(ciphertext.is_ntt_form());
        destination.set_scale(ciphertext.scale());
        destination.set_correction_factor(ciphertext.correction_factor());
        destination.SetNoiseScaleDeg(ciphertext.GetNoiseScaleDeg());

        const std::uint64_t grid_dim =
                ceil_div_uint64(coeff_count * target_coeff_modulus_size, static_cast<std::uint64_t>(blockDimGlb.x));

        for (std::size_t poly_index = 0; poly_index < ciphertext.size(); poly_index++) {
            const auto *src_poly = input_coeff.get() + poly_index * coeff_count;
            auto *dst_poly = destination.data() + poly_index * coeff_count * target_coeff_modulus_size;
            mod_raise_centered_kernel<<<grid_dim, blockDimGlb, 0, stream>>>(
                    src_poly,
                    dst_poly,
                    context.gpu_rns_tables().modulus(),
                    coeff_count,
                    target_coeff_modulus_size);

            if (ciphertext.is_ntt_form()) {
                nwt_2d_radix8_forward_inplace(dst_poly, context.gpu_rns_tables(), target_coeff_modulus_size, 0, stream);
            }
        }

        return destination;
    }

    PhantomCiphertext regular_bootstrapping_v2(const PhantomContext &context,
                                               const PhantomCiphertext &ciphertext,
                                               const PhantomGaloisKey *galois_key,
                                               const PhantomRelinKey *relin_key,
                                               const CKKSBootstrapConfig &config) {
        (void) galois_key;
        (void) relin_key;

        if (config.enable_eval_mod) {
            throw std::invalid_argument("regular_bootstrapping_v2 EvalMod phase is not implemented yet");
        }

        // Phase 1 MVP: keep HEonGPU v2's centered ModRaise behavior as the correctness-critical foundation.
        return mod_up_from_q0(context, ciphertext);
    }

} // namespace phantom
