#pragma once

#include <cmath>

#include "ciphertext.h"

class PhantomGaloisKey;
class PhantomRelinKey;

namespace phantom {

    struct CKKSBootstrapConfig {
        // Keep the same knobs as HEonGPU regular_bootstrapping_v2 for forward compatibility.
        double message_ratio = 256.0;
        double scaling_factor = std::pow(2.0, 40);
        bool enable_eval_mod = false;
    };

    // Raise a ciphertext from the last CKKS level (single-q0) to the first data level using centered representation.
    PhantomCiphertext mod_up_from_q0(const PhantomContext &context, const PhantomCiphertext &ciphertext);

    // Regular bootstrapping v2 entrypoint. This MVP currently executes ModRaise with centered semantics.
    PhantomCiphertext regular_bootstrapping_v2(const PhantomContext &context,
                                               const PhantomCiphertext &ciphertext,
                                               const PhantomGaloisKey *galois_key,
                                               const PhantomRelinKey *relin_key,
                                               const CKKSBootstrapConfig &config = {});

} // namespace phantom
