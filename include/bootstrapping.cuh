#pragma once

#include <cmath>
#include <vector>

#include "ciphertext.h"

class PhantomGaloisKey;
class PhantomRelinKey;

namespace phantom {

    enum class CKKSEvalModMethod {
        chebyshev,
    };

    struct CKKSBootstrapConfig {
        // Keep the same knobs as HEonGPU regular_bootstrapping_v2 for forward compatibility.
        double message_ratio = 256.0;
        double scaling_factor = std::pow(2.0, 40);
        bool enable_eval_mod = false;

        // Phase-2: EvalMod is planned to use Chebyshev approximation instead of Taylor.
        CKKSEvalModMethod eval_mod_method = CKKSEvalModMethod::chebyshev;
        std::size_t chebyshev_degree = 31;
        double chebyshev_min = -0.25;
        double chebyshev_max = 0.25;
    };

    // Generate Chebyshev coefficients for EvalMod target f(x)=sin(2*pi*x)/(2*pi) on [chebyshev_min, chebyshev_max].
    std::vector<double> generate_eval_mod_chebyshev_coefficients(const CKKSBootstrapConfig &config);

    // Evaluate Chebyshev series at x using Clenshaw recurrence on [chebyshev_min, chebyshev_max].
    double eval_mod_chebyshev_reference(double x, const CKKSBootstrapConfig &config,
                                        const std::vector<double> &coefficients);

    // Raise a ciphertext from the last CKKS level (single-q0) to the first data level using centered representation.
    PhantomCiphertext mod_up_from_q0(const PhantomContext &context, const PhantomCiphertext &ciphertext);

    // Regular bootstrapping v2 entrypoint. This MVP currently executes ModRaise with centered semantics.
    PhantomCiphertext regular_bootstrapping_v2(const PhantomContext &context,
                                               const PhantomCiphertext &ciphertext,
                                               const PhantomGaloisKey *galois_key,
                                               const PhantomRelinKey *relin_key,
                                               const CKKSBootstrapConfig &config = {});

} // namespace phantom
