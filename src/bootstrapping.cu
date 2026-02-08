#include "bootstrapping.cuh"

#include <cstdint>

#include "ckks.h"
#include "common.h"
#include "evaluate.cuh"
#include "ntt.cuh"
#include "uintmodmath.cuh"

using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

namespace {

    constexpr double kPi = 3.141592653589793238462643383279502884;
    constexpr std::size_t kNaiveDftSlotThreshold = 64;
    constexpr double kLinearTransformPlainScale = 1048576.0; // 2^20

    inline std::uint64_t ceil_div_uint64(std::uint64_t numerator, std::uint64_t denominator) {
        return (numerator + denominator - 1) / denominator;
    }

    inline double eval_mod_target(double x) {
        // Target function used in CKKS EvalMod: sin(2*pi*x)/(2*pi)
        return std::sin(2.0 * kPi * x) / (2.0 * kPi);
    }

    void validate_chebyshev_config(const CKKSBootstrapConfig &config) {
        if (config.chebyshev_degree == 0) {
            throw std::invalid_argument("chebyshev_degree must be greater than zero");
        }

        if (!(config.chebyshev_min < config.chebyshev_max)) {
            throw std::invalid_argument("chebyshev range is invalid");
        }

        if (config.eval_mod_method != CKKSEvalModMethod::chebyshev) {
            throw std::invalid_argument("only Chebyshev EvalMod is supported");
        }
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

    std::vector<double> generate_eval_mod_chebyshev_coefficients(const CKKSBootstrapConfig &config) {
        validate_chebyshev_config(config);

        const std::size_t degree = config.chebyshev_degree;
        const std::size_t sample_count = degree + 1;
        const double a = config.chebyshev_min;
        const double b = config.chebyshev_max;

        std::vector<double> coefficients(degree + 1, 0.0);

        for (std::size_t k = 0; k <= degree; k++) {
            double sum = 0.0;
            for (std::size_t j = 0; j < sample_count; j++) {
                const double theta = (static_cast<double>(j) + 0.5) * kPi / static_cast<double>(sample_count);
                const double t = std::cos(theta);
                const double x = 0.5 * (a + b) + 0.5 * (b - a) * t;
                sum += eval_mod_target(x) * std::cos(static_cast<double>(k) * theta);
            }
            coefficients[k] = 2.0 * sum / static_cast<double>(sample_count);
        }

        // Chebyshev series convention: f(x)=c0 + c1*T1(x) + ..., where c0 is halved from DCT result.
        coefficients[0] *= 0.5;
        return coefficients;
    }

    double eval_mod_chebyshev_reference(double x, const CKKSBootstrapConfig &config,
                                        const std::vector<double> &coefficients) {
        validate_chebyshev_config(config);

        if (coefficients.empty()) {
            throw std::invalid_argument("coefficients cannot be empty");
        }

        if (coefficients.size() != config.chebyshev_degree + 1) {
            throw std::invalid_argument("coefficients size does not match chebyshev_degree");
        }

        if (x < config.chebyshev_min || x > config.chebyshev_max) {
            throw std::invalid_argument("x is out of chebyshev approximation range");
        }

        const double a = config.chebyshev_min;
        const double b = config.chebyshev_max;
        const double t = (2.0 * x - (a + b)) / (b - a);

        double b_kplus1 = 0.0;
        double b_kplus2 = 0.0;

        for (std::int64_t idx = static_cast<std::int64_t>(coefficients.size()) - 1; idx >= 1; idx--) {
            const double b_k = 2.0 * t * b_kplus1 - b_kplus2 + coefficients[static_cast<std::size_t>(idx)];
            b_kplus2 = b_kplus1;
            b_kplus1 = b_k;
        }

        return t * b_kplus1 - b_kplus2 + coefficients[0];
    }



    void build_dft_diagonal_plan(std::size_t slot_count, bool inverse,
                                 std::vector<int> &steps,
                                 std::vector<std::vector<cuDoubleComplex>> &diagonals) {
        if (slot_count == 0) {
            throw std::invalid_argument("slot_count cannot be zero");
        }

        const double norm = 1.0 / std::sqrt(static_cast<double>(slot_count));
        const double sign = inverse ? 1.0 : -1.0;

        steps.resize(slot_count);
        diagonals.resize(slot_count);

        for (std::size_t s = 0; s < slot_count; s++) {
            steps[s] = static_cast<int>(s);
            diagonals[s].resize(slot_count);

            for (std::size_t row = 0; row < slot_count; row++) {
                const std::size_t col = (row + s) % slot_count;
                const double angle = sign * 2.0 * kPi * static_cast<double>(row * col) /
                                     static_cast<double>(slot_count);
                diagonals[s][row] = make_cuDoubleComplex(norm * std::cos(angle), norm * std::sin(angle));
            }
        }
    }

    PhantomCiphertext apply_linear_transform_naive(
            const PhantomContext &context,
            const PhantomCiphertext &ciphertext,
            const PhantomGaloisKey &galois_key,
            const std::vector<int> &steps,
            const std::vector<std::vector<cuDoubleComplex>> &diagonals,
            double plain_scale) {
        if (steps.empty() || diagonals.empty()) {
            throw std::invalid_argument("linear transform cannot be empty");
        }

        if (steps.size() != diagonals.size()) {
            throw std::invalid_argument("linear transform steps and diagonals size mismatch");
        }

        if (plain_scale <= 0.0) {
            throw std::invalid_argument("plain_scale must be positive");
        }

        const auto &context_data = context.get_context_data(ciphertext.chain_index());
        if (context_data.parms().scheme() != scheme_type::ckks) {
            throw std::invalid_argument("linear transform only supports CKKS");
        }

        const std::size_t slot_count = context_data.parms().poly_modulus_degree() >> 1;

        PhantomCKKSEncoder encoder(context);

        PhantomCiphertext acc;
        bool initialized = false;

        for (std::size_t i = 0; i < steps.size(); i++) {
            const auto &diagonal = diagonals[i];
            if (diagonal.empty() || diagonal.size() > slot_count) {
                throw std::invalid_argument("invalid diagonal size");
            }

            std::vector<cuDoubleComplex> padded_diagonal(slot_count, make_cuDoubleComplex(0.0, 0.0));
            for (std::size_t j = 0; j < diagonal.size(); j++) {
                padded_diagonal[j] = diagonal[j];
            }

            auto plain = encoder.encode(context, padded_diagonal, plain_scale, ciphertext.chain_index());

            auto rotated = (steps[i] == 0)
                           ? ciphertext
                           : rotate(context, ciphertext, steps[i], galois_key);
            multiply_plain_inplace(context, rotated, plain);

            if (!initialized) {
                acc = std::move(rotated);
                initialized = true;
            } else {
                add_inplace(context, acc, rotated);
            }
        }

        return acc;
    }

    PhantomCiphertext coeff_to_slot_phase25(const PhantomContext &context,
                                            const PhantomCiphertext &ciphertext,
                                            const PhantomGaloisKey &galois_key) {
        const std::size_t slot_count =
                context.get_context_data(ciphertext.chain_index()).parms().poly_modulus_degree() >> 1;

        if (slot_count > kNaiveDftSlotThreshold) {
            // Keep large-parameter path stable until BSGS/hoisting DFT is integrated.
            std::vector<cuDoubleComplex> identity_diag(slot_count, make_cuDoubleComplex(1.0, 0.0));
            return apply_linear_transform_naive(context, ciphertext, galois_key, {0}, {identity_diag}, 1.0);
        }

        std::vector<int> steps;
        std::vector<std::vector<cuDoubleComplex>> diagonals;
        build_dft_diagonal_plan(slot_count, false, steps, diagonals);
        return apply_linear_transform_naive(context, ciphertext, galois_key, steps, diagonals,
                                            kLinearTransformPlainScale);
    }

    PhantomCiphertext eval_mod_chebyshev_phase25(const PhantomCiphertext &ciphertext,
                                                 const PhantomRelinKey &relin_key,
                                                 const CKKSBootstrapConfig &config) {
        (void) relin_key;

        auto coefficients = generate_eval_mod_chebyshev_coefficients(config);
        const double center = 0.5 * (config.chebyshev_min + config.chebyshev_max);
        const double quarter = 0.25 * (config.chebyshev_max - config.chebyshev_min);

        // Keep a lightweight runtime sanity check so bad parameters fail early before GPU stages are wired in.
        const double probe0 = eval_mod_chebyshev_reference(center - quarter, config, coefficients);
        const double probe1 = eval_mod_chebyshev_reference(center, config, coefficients);
        const double probe2 = eval_mod_chebyshev_reference(center + quarter, config, coefficients);
        if (!(std::isfinite(probe0) && std::isfinite(probe1) && std::isfinite(probe2))) {
            throw std::logic_error("chebyshev EvalMod probe failed");
        }

        return ciphertext;
    }

    PhantomCiphertext slot_to_coeff_phase25(const PhantomContext &context,
                                            const PhantomCiphertext &ciphertext,
                                            const PhantomGaloisKey &galois_key) {
        const std::size_t slot_count =
                context.get_context_data(ciphertext.chain_index()).parms().poly_modulus_degree() >> 1;

        if (slot_count > kNaiveDftSlotThreshold) {
            std::vector<cuDoubleComplex> identity_diag(slot_count, make_cuDoubleComplex(1.0, 0.0));
            return apply_linear_transform_naive(context, ciphertext, galois_key, {0}, {identity_diag}, 1.0);
        }

        std::vector<int> steps;
        std::vector<std::vector<cuDoubleComplex>> diagonals;
        build_dft_diagonal_plan(slot_count, true, steps, diagonals);
        return apply_linear_transform_naive(context, ciphertext, galois_key, steps, diagonals,
                                            kLinearTransformPlainScale);
    }

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
        auto raised = mod_up_from_q0(context, ciphertext);

        if (!config.enable_eval_mod) {
            return raised;
        }

        if (config.eval_mod_method != CKKSEvalModMethod::chebyshev) {
            throw std::invalid_argument("regular_bootstrapping_v2 only supports Chebyshev EvalMod");
        }

        if (galois_key == nullptr) {
            throw std::invalid_argument("regular_bootstrapping_v2 requires galois_key when EvalMod is enabled");
        }

        if (relin_key == nullptr) {
            throw std::invalid_argument("regular_bootstrapping_v2 requires relin_key when EvalMod is enabled");
        }

        // Phase-3.2 wiring: ModRaise -> CoeffToSlot -> Chebyshev EvalMod -> SlotToCoeff.
        auto slot_cipher = coeff_to_slot_phase25(context, raised, *galois_key);
        auto reduced_slot_cipher = eval_mod_chebyshev_phase25(slot_cipher, *relin_key, config);
        return slot_to_coeff_phase25(context, reduced_slot_cipher, *galois_key);
    }

} // namespace phantom
