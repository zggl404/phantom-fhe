#include "phantom.h"

#include <cmath>
#include <stdexcept>
#include <vector>

using namespace phantom;
using namespace phantom::arith;

namespace {

    EncryptionParameters create_ckks_test_parms() {
        EncryptionParameters parms(scheme_type::ckks);
        const std::size_t poly_modulus_degree = 8192;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {50, 40, 40, 50}));
        return parms;
    }


    EncryptionParameters create_ckks_phase32_parms() {
        EncryptionParameters parms(scheme_type::ckks);
        const std::size_t poly_modulus_degree = 2048;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {40, 30, 30, 40}));
        return parms;
    }

    EncryptionParameters create_ckks_unsupported_evalmod_parms() {
        EncryptionParameters parms(scheme_type::ckks);
        const std::size_t poly_modulus_degree = 1024;
        parms.set_poly_modulus_degree(poly_modulus_degree);
        parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {40, 30, 40}));
        return parms;
    }

    template<typename Func>
    void expect_invalid_argument(Func &&func, const char *message) {
        try {
            func();
        } catch (const std::invalid_argument &) {
            return;
        }
        throw std::logic_error(message);
    }

    std::uint64_t centered_mod_raise_expected(std::uint64_t value, std::uint64_t q0, std::uint64_t qi) {
        if (value >= (q0 >> 1)) {
            const std::uint64_t delta = q0 - value;
            const std::uint64_t reduced = delta % qi;
            return (reduced == 0) ? 0 : (qi - reduced);
        }
        return value % qi;
    }

    void test_mod_up_from_q0_centered_boundary() {
        const auto parms = create_ckks_test_parms();
        PhantomContext context(parms);

        const auto last_chain_index = context.total_parm_size() - 1;
        const auto first_chain_index = context.get_first_index();
        const auto &last_context_data = context.get_context_data(last_chain_index);
        const auto &first_context_data = context.get_context_data(first_chain_index);

        const std::size_t n = last_context_data.parms().poly_modulus_degree();
        const std::uint64_t q0 = last_context_data.parms().coeff_modulus().front().value();
        const auto &q = first_context_data.parms().coeff_modulus();

        std::vector<std::uint64_t> boundary_values = {
                0,
                1,
                (q0 >> 1) - 1,
                (q0 >> 1),
                q0 - 1};

        PhantomCiphertext input;
        input.resize(context, last_chain_index, 2, cudaStreamPerThread);
        input.set_ntt_form(false);
        input.set_scale(1.0);

        std::vector<std::uint64_t> host_input(2 * n, 0);
        for (std::size_t i = 0; i < boundary_values.size(); i++) {
            host_input[i] = boundary_values[i];
            host_input[n + i] = boundary_values[(i + 1) % boundary_values.size()];
        }

        cudaMemcpyAsync(input.data(), host_input.data(), host_input.size() * sizeof(std::uint64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
        cudaStreamSynchronize(cudaStreamPerThread);

        auto raised = mod_up_from_q0(context, input);

        if (raised.chain_index() != first_chain_index) {
            throw std::logic_error("mod_up_from_q0 returned invalid chain index");
        }
        if (raised.coeff_modulus_size() != q.size()) {
            throw std::logic_error("mod_up_from_q0 returned invalid coeff modulus size");
        }
        if (raised.is_ntt_form()) {
            throw std::logic_error("mod_up_from_q0 should preserve coefficient form for this test");
        }

        std::vector<std::uint64_t> host_raised(raised.size() * raised.coeff_modulus_size() * n, 0);
        cudaMemcpyAsync(host_raised.data(), raised.data(), host_raised.size() * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost, cudaStreamPerThread);
        cudaStreamSynchronize(cudaStreamPerThread);

        for (std::size_t prime_index = 0; prime_index < q.size(); prime_index++) {
            const std::uint64_t qi = q[prime_index].value();
            for (std::size_t i = 0; i < boundary_values.size(); i++) {
                const auto expected_c0 = centered_mod_raise_expected(boundary_values[i], q0, qi);
                const auto actual_c0 = host_raised[prime_index * n + i];
                if (actual_c0 != expected_c0) {
                    throw std::logic_error("mod_up_from_q0 centered mapping mismatch on c0");
                }

                const auto c1_value = boundary_values[(i + 1) % boundary_values.size()];
                const auto expected_c1 = centered_mod_raise_expected(c1_value, q0, qi);
                const auto c1_base_offset = raised.coeff_modulus_size() * n;
                const auto actual_c1 = host_raised[c1_base_offset + prime_index * n + i];
                if (actual_c1 != expected_c1) {
                    throw std::logic_error("mod_up_from_q0 centered mapping mismatch on c1");
                }
            }
        }
    }


    void test_bootstrap_guard_checks() {
        const auto parms = create_ckks_test_parms();
        PhantomContext context(parms);

        PhantomSecretKey secret_key(context);
        PhantomCKKSEncoder encoder(context);

        std::vector<double> input{0.5, -1.25, 2.0};
        const double scale = std::pow(2.0, 20);

        auto plain_first = encoder.encode(context, input, scale, context.get_first_index());
        auto cipher_first = secret_key.encrypt_symmetric(context, plain_first);
        expect_invalid_argument(
                [&]() { (void) mod_up_from_q0(context, cipher_first); },
                "mod_up_from_q0 should reject non-last-chain ciphertext");

        auto plain_last = encoder.encode(context, input, scale, context.total_parm_size() - 1);
        auto cipher_last = secret_key.encrypt_symmetric(context, plain_last);

        CKKSBootstrapConfig config;
        config.enable_eval_mod = true;
        expect_invalid_argument(
                [&]() { (void) regular_bootstrapping_v2(context, cipher_last, nullptr, nullptr, config); },
                "regular_bootstrapping_v2 should reject unsupported EvalMod mode");
    }



    void test_linear_transform_naive_identity() {
        const auto parms = create_ckks_phase32_parms();
        PhantomContext context(parms);

        PhantomSecretKey secret_key(context);
        PhantomCKKSEncoder encoder(context);
        auto galois_key = secret_key.create_galois_keys(context);

        std::vector<double> input{1.25, -2.5, 3.75, -4.125, 5.5};
        const double scale = std::pow(2.0, 15);

        auto plain = encoder.encode(context, input, scale, context.total_parm_size() - 1);
        auto cipher = secret_key.encrypt_symmetric(context, plain);

        const std::size_t slot_count = encoder.slot_count();
        std::vector<cuDoubleComplex> identity_diag(slot_count, make_cuDoubleComplex(1.0, 0.0));
        auto transformed = apply_linear_transform_naive(context, cipher, galois_key, {0}, {identity_diag}, 1.0);

        auto plain_out = secret_key.decrypt(context, transformed);
        auto output = encoder.decode<double>(context, plain_out);

        for (std::size_t i = 0; i < input.size(); i++) {
            if (std::fabs(output[i] - input[i]) > 1e-3) {
                throw std::logic_error("linear transform identity validation failed");
            }
        }
    }

    void test_linear_transform_naive_negation() {
        const auto parms = create_ckks_phase32_parms();
        PhantomContext context(parms);

        PhantomSecretKey secret_key(context);
        PhantomCKKSEncoder encoder(context);
        auto galois_key = secret_key.create_galois_keys(context);

        std::vector<double> input{0.75, -1.0, 1.5, -2.25, 3.0};
        const double scale = std::pow(2.0, 15);

        auto plain = encoder.encode(context, input, scale, context.total_parm_size() - 1);
        auto cipher = secret_key.encrypt_symmetric(context, plain);

        const std::size_t slot_count = encoder.slot_count();
        std::vector<cuDoubleComplex> neg_diag(slot_count, make_cuDoubleComplex(-1.0, 0.0));
        auto transformed = apply_linear_transform_naive(context, cipher, galois_key, {0}, {neg_diag}, 1.0);

        auto plain_out = secret_key.decrypt(context, transformed);
        auto output = encoder.decode<double>(context, plain_out);

        for (std::size_t i = 0; i < input.size(); i++) {
            if (std::fabs(output[i] + input[i]) > 1e-3) {
                throw std::logic_error("linear transform negation validation failed");
            }
        }
    }


    void test_phase32_unsupported_degree_guard() {
        const auto parms = create_ckks_unsupported_evalmod_parms();
        PhantomContext context(parms);

        PhantomSecretKey secret_key(context);
        auto galois_key = secret_key.create_galois_keys(context);
        auto relin_key = secret_key.gen_relinkey(context);

        const std::size_t last_chain_index = context.total_parm_size() - 1;
        const std::size_t coeff_count = context.get_context_data(last_chain_index).parms().poly_modulus_degree();

        PhantomCiphertext cipher;
        cipher.resize(context, last_chain_index, 2, cudaStreamPerThread);
        cipher.set_ntt_form(false);
        cipher.set_scale(std::pow(2.0, 10));

        std::vector<std::uint64_t> host_input(2 * coeff_count, 0);
        for (std::size_t i = 0; i < coeff_count; i++) {
            host_input[i] = static_cast<std::uint64_t>(i & 1U);
            host_input[coeff_count + i] = static_cast<std::uint64_t>((i + 1U) & 1U);
        }
        cudaMemcpyAsync(cipher.data(), host_input.data(), host_input.size() * sizeof(std::uint64_t),
                        cudaMemcpyHostToDevice, cudaStreamPerThread);
        cudaStreamSynchronize(cudaStreamPerThread);

        CKKSBootstrapConfig config;
        config.enable_eval_mod = true;
        config.eval_mod_method = CKKSEvalModMethod::chebyshev;

        expect_invalid_argument(
                [&]() { (void) regular_bootstrapping_v2(context, cipher, &galois_key, &relin_key, config); },
                "regular_bootstrapping_v2 should reject unsupported poly_modulus_degree for EvalMod");
    }

    void test_phase32_supported_degree_roundtrip() {
        const auto parms = create_ckks_phase32_parms();
        PhantomContext context(parms);

        PhantomSecretKey secret_key(context);
        PhantomCKKSEncoder encoder(context);
        auto galois_key = secret_key.create_galois_keys(context);
        auto relin_key = secret_key.gen_relinkey(context);

        const std::size_t last_chain_index = context.total_parm_size() - 1;

        std::vector<double> input{0.03125, -0.0625, 0.09375, -0.125, 0.15625, -0.1875};
        const double scale = std::pow(2.0, 15);

        auto plain = encoder.encode(context, input, scale, last_chain_index);
        auto cipher = secret_key.encrypt_symmetric(context, plain);

        CKKSBootstrapConfig config;
        config.enable_eval_mod = true;
        config.eval_mod_method = CKKSEvalModMethod::chebyshev;
        config.chebyshev_degree = 31;
        config.chebyshev_min = -0.25;
        config.chebyshev_max = 0.25;

        auto phase32_output = regular_bootstrapping_v2(context, cipher, &galois_key, &relin_key, config);
        auto mod_up_only = mod_up_from_q0(context, cipher);

        auto plain_phase32 = secret_key.decrypt(context, phase32_output);
        auto plain_mod_up = secret_key.decrypt(context, mod_up_only);

        auto decoded_phase32 = encoder.decode<double>(context, plain_phase32);
        auto decoded_mod_up = encoder.decode<double>(context, plain_mod_up);

        for (std::size_t i = 0; i < input.size(); i++) {
            if (std::fabs(decoded_phase32[i] - decoded_mod_up[i]) > 1e-1) {
                throw std::logic_error("phase3.2 fallback round-trip validation failed");
            }
        }
    }

    void test_phase25_eval_mod_pipeline_with_keys() {
        const auto parms = create_ckks_test_parms();
        PhantomContext context(parms);

        PhantomSecretKey secret_key(context);
        PhantomCKKSEncoder encoder(context);
        auto galois_key = secret_key.create_galois_keys(context);
        auto relin_key = secret_key.gen_relinkey(context);

        const std::size_t last_chain_index = context.total_parm_size() - 1;

        std::vector<double> input{0.125, -0.25, 0.375, -0.5, 0.625};
        const double scale = std::pow(2.0, 20);

        auto plain = encoder.encode(context, input, scale, last_chain_index);
        auto cipher = secret_key.encrypt_symmetric(context, plain);

        CKKSBootstrapConfig config;
        config.enable_eval_mod = true;
        config.eval_mod_method = CKKSEvalModMethod::chebyshev;
        config.chebyshev_degree = 31;
        config.chebyshev_min = -0.25;
        config.chebyshev_max = 0.25;

        auto phase25_output = regular_bootstrapping_v2(context, cipher, &galois_key, &relin_key, config);
        auto mod_up_only = mod_up_from_q0(context, cipher);

        if (phase25_output.size() != mod_up_only.size() ||
            phase25_output.coeff_modulus_size() != mod_up_only.coeff_modulus_size() ||
            phase25_output.poly_modulus_degree() != mod_up_only.poly_modulus_degree()) {
            throw std::logic_error("phase2.5 pipeline output shape mismatch");
        }

        auto plain_phase25 = secret_key.decrypt(context, phase25_output);
        auto plain_mod_up = secret_key.decrypt(context, mod_up_only);

        auto decoded_phase25 = encoder.decode<double>(context, plain_phase25);
        auto decoded_mod_up = encoder.decode<double>(context, plain_mod_up);

        for (std::size_t i = 0; i < input.size(); i++) {
            if (std::fabs(decoded_phase25[i] - decoded_mod_up[i]) > 1e-3) {
                throw std::logic_error("phase2.5 large-slot fallback should match ModRaise semantics");
            }
        }
    }

    void test_regular_bootstrapping_v2_matches_mod_up() {
        const auto parms = create_ckks_test_parms();
        PhantomContext context(parms);

        PhantomSecretKey secret_key(context);
        PhantomCKKSEncoder encoder(context);

        const std::size_t last_chain_index = context.total_parm_size() - 1;
        const std::size_t first_chain_index = context.get_first_index();

        std::vector<double> input{0.5, -1.25, 2.0, -3.5, 4.25, -5.75};
        const double scale = std::pow(2.0, 20);

        auto plain = encoder.encode(context, input, scale, last_chain_index);
        auto cipher = secret_key.encrypt_symmetric(context, plain);

        CKKSBootstrapConfig config;
        // Phase-1 only implements ModRaise, so regular_bootstrapping_v2 should be equivalent to mod_up_from_q0.
        auto mod_up_only = mod_up_from_q0(context, cipher);
        auto bootstrapped = regular_bootstrapping_v2(context, cipher, nullptr, nullptr, config);

        if (bootstrapped.chain_index() != first_chain_index) {
            throw std::logic_error("regular_bootstrapping_v2 returned invalid chain index");
        }
        if (!bootstrapped.is_ntt_form()) {
            throw std::logic_error("regular_bootstrapping_v2 should return NTT-form ciphertext");
        }

        if (bootstrapped.size() != mod_up_only.size() ||
            bootstrapped.coeff_modulus_size() != mod_up_only.coeff_modulus_size() ||
            bootstrapped.poly_modulus_degree() != mod_up_only.poly_modulus_degree()) {
            throw std::logic_error("regular_bootstrapping_v2 output shape mismatch");
        }

        const std::size_t total_uint64 =
                bootstrapped.size() * bootstrapped.coeff_modulus_size() * bootstrapped.poly_modulus_degree();
        std::vector<std::uint64_t> host_boot(total_uint64, 0);
        std::vector<std::uint64_t> host_mod_up(total_uint64, 0);

        cudaMemcpyAsync(host_boot.data(), bootstrapped.data(), total_uint64 * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost, cudaStreamPerThread);
        cudaMemcpyAsync(host_mod_up.data(), mod_up_only.data(), total_uint64 * sizeof(std::uint64_t),
                        cudaMemcpyDeviceToHost, cudaStreamPerThread);
        cudaStreamSynchronize(cudaStreamPerThread);

        for (std::size_t i = 0; i < total_uint64; i++) {
            if (host_boot[i] != host_mod_up[i]) {
                throw std::logic_error("regular_bootstrapping_v2 does not match mod_up_from_q0 in phase-1 MVP");
            }
        }
    }

} // namespace

int main() {
    test_mod_up_from_q0_centered_boundary();
    test_bootstrap_guard_checks();
    test_linear_transform_naive_identity();
    test_linear_transform_naive_negation();
    test_regular_bootstrapping_v2_matches_mod_up();
    test_phase32_unsupported_degree_guard();
    test_phase32_supported_degree_roundtrip();
    test_phase25_eval_mod_pipeline_with_keys();
    return 0;
}
