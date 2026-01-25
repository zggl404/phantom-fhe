#pragma once

#include <string>
#include <vector>

#include "boot/Bootstrapper.cuh"
#include "phantom.h"

namespace phantom {

PhantomCiphertext evalConv_BNRelu_new(
    const PhantomContext &context,
    CKKSEvaluator &ckks_evaluator,
    PhantomCKKSEncoder &encoder,
    const PhantomCiphertext &ct_input,
    const std::vector<double> &ker_in,
    const std::vector<double> &bn_a,
    const std::vector<double> &bn_b,
    double alpha,
    double pow,
    int in_wid,
    int kp_wid,
    int ker_wid,
    int real_ib,
    int real_ob,
    int norm,
    int pack_pos,
    int step,
    int iter,
    int log_sparse,
    const std::string &kind,
    bool fast_pack,
    bool debug);

} // namespace phantom
