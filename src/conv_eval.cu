#include "conv_eval.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

using namespace std;

namespace phantom {

static vector<vector<double>> reshape_ker(const vector<double> &ker_in,
                                          int k_sz,
                                          int out_batch,
                                          bool trans) {
    vector<vector<double>> ker_out(out_batch, vector<double>(k_sz * (static_cast<int>(ker_in.size()) / (k_sz * out_batch)), 0.0));
    int in_batch = static_cast<int>(ker_in.size()) / (k_sz * out_batch);

    for (int i = 0; i < out_batch; i++) {
        for (int j = 0; j < in_batch; j++) {
            for (int k = 0; k < k_sz; k++) {
                if (trans) {
                    ker_out[i][j * k_sz + (k_sz - 1 - k)] =
                        ker_in[j + i * in_batch + k * out_batch * in_batch];
                } else {
                    ker_out[i][j * k_sz + k] =
                        ker_in[i + j * out_batch + k * out_batch * in_batch];
                }
            }
        }
    }

    return ker_out;
}

static vector<double> encode_ker_final(const vector<vector<double>> &ker_in,
                                       int pos,
                                       int i,
                                       int in_wid,
                                       int in_batch,
                                       int ker_wid) {
    int vec_size = in_wid * in_wid * in_batch;
    vector<double> output(vec_size, 0.0);
    int bias = pos * ker_wid * ker_wid * in_batch;
    int k_sz = ker_wid * ker_wid;

    for (int j = 0; j < in_batch; j++) {
        for (int k = 0; k < k_sz; k++) {
            output[(in_wid * (k / ker_wid) + k % ker_wid) * in_batch + j] =
                ker_in[i][(in_batch - 1 - j) * k_sz + (k_sz - 1 - k) + bias];
        }
    }

    int adj = (in_batch - 1) + (in_batch) * (in_wid + 1) * (ker_wid - 1) / 2;
    vector<double> tmp(adj, 0.0);
    for (int idx = 0; idx < adj; idx++) {
        tmp[idx] = output[vec_size - adj + idx];
        output[vec_size - adj + idx] = -output[idx];
    }
    for (int idx = 0; idx < vec_size - 2 * adj; idx++) {
        output[idx] = output[idx + adj];
    }
    for (int idx = 0; idx < adj; idx++) {
        output[idx + vec_size - 2 * adj] = tmp[idx];
    }

    return output;
}

static vector<PhantomPlaintext> prep_ker(const PhantomContext &context,
                                         PhantomCKKSEncoder &encoder,
                                         const vector<double> &ker_in,
                                         const vector<double> &bn_a,
                                         int in_wid,
                                         int ker_wid,
                                         int real_ib,
                                         int real_ob,
                                         int norm,
                                         int pos,
                                         bool trans,
                                         double scale) {
    int max_bat = static_cast<int>(context.key_context_data().parms().poly_modulus_degree()) / (in_wid * in_wid);
    int ker_size = ker_wid * ker_wid;
    vector<vector<double>> ker_rs = reshape_ker(ker_in, ker_size, real_ob, trans);

    for (int i = 0; i < real_ob; i++) {
        for (size_t j = 0; j < ker_rs[i].size(); j++) {
            ker_rs[i][j] *= bn_a[i];
        }
    }

    vector<vector<double>> max_ker_rs(max_bat, vector<double>(max_bat * ker_size, 0.0));
    for (int i = 0; i < real_ob; i++) {
        for (int j = 0; j < real_ib; j++) {
            for (int k = 0; k < ker_size; k++) {
                max_ker_rs[norm * i][norm * j * ker_size + k] =
                    ker_rs[i][j * ker_size + k];
            }
        }
    }

    vector<PhantomPlaintext> pl_ker(max_bat);
    for (int i = 0; i < max_bat; i++) {
        encoder.encode_coeffs(context,
                              encode_ker_final(max_ker_rs, pos, i, in_wid, max_bat, ker_wid),
                              scale,
                              pl_ker[i]);
    }

    return pl_ker;
}

static vector<PhantomPlaintext> gen_idxNlogs(const PhantomContext &context,
                                             PhantomCKKSEncoder &encoder) {
    size_t N = context.key_context_data().parms().poly_modulus_degree();
    int logN = static_cast<int>(round(log2(static_cast<double>(N))));
    vector<PhantomPlaintext> idx(logN);
    vector<double> coeffs(N, 0.0);

    for (int i = 0; i < logN; i++) {
        coeffs[1 << i] = 1.0;
        encoder.encode_coeffs(context, coeffs, 1.0, idx[i]);
        coeffs[1 << i] = 0.0;
    }

    return idx;
}

static PhantomCiphertext pack_ctxts(CKKSEvaluator &ckks_evaluator,
                                    const vector<PhantomCiphertext> &ctxts_in,
                                    int max_cnum,
                                    int real_cnum,
                                    const vector<PhantomPlaintext> &idx,
                                    int logN) {
    int norm = max_cnum / real_cnum;
    vector<PhantomCiphertext> ctxts(max_cnum);

    for (int i = 0; i < max_cnum; i++) {
        if (i % norm == 0) {
            ctxts[i] = ctxts_in[i];
            ctxts[i].set_scale(ctxts[i].scale() * static_cast<double>(real_cnum));
        }
    }

    int step = max_cnum / 2;
    int logStep = 0;
    for (int i = step; i > 1; i /= 2) {
        logStep++;
    }
    int j = logN - logStep;

    PhantomCiphertext tmp1, tmp2, rot;
    for (; step >= norm; step /= 2) {
        for (int i = 0; i < step; i += norm) {
            ckks_evaluator.evaluator.multiply_plain(ctxts[i + step], idx[logStep], tmp1);
            ckks_evaluator.evaluator.sub(ctxts[i], tmp1, tmp2);
            ckks_evaluator.evaluator.add(ctxts[i], tmp1, tmp1);
            uint32_t elt = (1u << j) + 1;
            ckks_evaluator.evaluator.apply_galois(tmp2, elt, *(ckks_evaluator.galois_keys), rot);
            ckks_evaluator.evaluator.add(tmp1, rot, ctxts[i]);
        }
        logStep--;
        j++;
    }

    return ctxts[0];
}

static PhantomCiphertext conv_then_pack(const PhantomContext &context,
                                        CKKSEvaluator &ckks_evaluator,
                                        const PhantomCiphertext &ctxt_in,
                                        const vector<PhantomPlaintext> &pl_ker,
                                        const vector<PhantomPlaintext> &plain_idx,
                                        int max_ob,
                                        int norm,
                                        double out_scale,
                                        int logN) {
    vector<PhantomCiphertext> ctxt_out(max_ob);

    for (int i = 0; i < max_ob; i++) {
        if (i % norm == 0) {
            ckks_evaluator.evaluator.multiply_plain(ctxt_in, pl_ker[i], ctxt_out[i]);
            ctxt_out[i].set_scale(out_scale / static_cast<double>(max_ob / norm));
        }
    }

    return pack_ctxts(ckks_evaluator, ctxt_out, max_ob, max_ob / norm, plain_idx, logN);
}

static PhantomCiphertext evalConv_BN(const PhantomContext &context,
                                     CKKSEvaluator &ckks_evaluator,
                                     PhantomCKKSEncoder &encoder,
                                     const PhantomCiphertext &ct_input,
                                     const vector<double> &ker_in,
                                     const vector<double> &bn_a,
                                     const vector<double> &bn_b,
                                     int in_wid,
                                     int ker_wid,
                                     int real_ib,
                                     int real_ob,
                                     int norm,
                                     double out_scale,
                                     bool trans) {
    int max_batch = static_cast<int>(context.key_context_data().parms().poly_modulus_degree()) / (in_wid * in_wid);
    vector<PhantomPlaintext> pl_ker = prep_ker(context, encoder, ker_in, bn_a,
                                               in_wid, ker_wid, real_ib, real_ob,
                                               norm, 0, trans, ct_input.scale());
    vector<PhantomPlaintext> plain_idx = gen_idxNlogs(context, encoder);
    int logN = static_cast<int>(round(log2(static_cast<double>(context.key_context_data().parms().poly_modulus_degree()))));

    PhantomCiphertext ct_conv = conv_then_pack(context, ckks_evaluator, ct_input, pl_ker,
                                               plain_idx, max_batch, norm, out_scale, logN);
    double pack_factor = static_cast<double>(max_batch / norm);
    ct_conv.set_scale(ct_conv.scale() * pack_factor);

    vector<double> b_coeffs(context.key_context_data().parms().poly_modulus_degree(), 0.0);
    for (int i = 0; i < real_ib; i++) {
        for (int j = 0; j < in_wid * in_wid; j++) {
            b_coeffs[norm * i + j * max_batch] = bn_b[i];
        }
    }
    PhantomPlaintext pt_bn_b;
    encoder.encode_coeffs(context, b_coeffs, ct_conv.scale(), pt_bn_b, ct_conv.chain_index());
    ckks_evaluator.evaluator.add_plain_inplace(ct_conv, pt_bn_b);

    return ct_conv;
}

static PhantomCiphertext relu_coeff(const PhantomContext &context,
                                    CKKSEvaluator &ckks_evaluator,
                                    Bootstrapper &bootstrapper,
                                    const PhantomCiphertext &ct_in) {
    PhantomCiphertext ct_manual = ct_in;
    while (ct_manual.coeff_modulus_size() > 1) {
        ckks_evaluator.evaluator.mod_switch_to_next_inplace(ct_manual);
    }

    bootstrapper.initial_scale = ct_manual.scale();
    ckks_evaluator.evaluator.multiply_const_inplace(ct_manual, scale_for_eval);
    bootstrapper.modraise_inplace(ct_manual);

    const auto &modulus = context.first_context_data().parms().coeff_modulus();
    ct_manual.set_scale(static_cast<double>(modulus[0].value()));

    PhantomCiphertext ct_slot_1, ct_slot_2;
    bootstrapper.coefftoslot_full_3(ct_slot_1, ct_slot_2, ct_manual);

    PhantomCiphertext ct_red_1, ct_red_2;
    bootstrapper.mod_reducer->modular_reduction_relu(ct_red_1, ct_slot_1);
    bootstrapper.mod_reducer->modular_reduction_relu(ct_red_2, ct_slot_2);

    PhantomCiphertext ct_out;
    bootstrapper.slottocoeff_full_3(ct_out, ct_red_1, ct_red_2);
    ct_out.set_scale(bootstrapper.final_scale);
    return ct_out;
}

PhantomCiphertext evalConv_BNRelu_new(
    const PhantomContext &context,
    CKKSEvaluator &ckks_evaluator,
    PhantomCKKSEncoder &encoder,
    const PhantomCiphertext &ct_input,
    const vector<double> &ker_in,
    const vector<double> &bn_a,
    const vector<double> &bn_b,
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
    const string &kind,
    bool fast_pack,
    bool debug) {
    (void)alpha;
    (void)kp_wid;
    (void)pack_pos;
    (void)step;
    (void)iter;
    (void)fast_pack;
    (void)debug;

    bool trans = false;
    if (kind != "Conv" && kind != "Conv_sparse") {
        throw invalid_argument("evalConv_BNRelu_new: unsupported kind");
    }

    double out_scale = ct_input.scale() * ct_input.scale();
    PhantomCiphertext ct_conv = evalConv_BN(context, ckks_evaluator, encoder, ct_input,
                                            ker_in, bn_a, bn_b, in_wid, ker_wid,
                                            real_ib, real_ob, norm, out_scale, trans);
    ct_conv.set_scale(ct_conv.scale() * std::pow(2.0, pow));

    long logN = static_cast<long>(round(log2(static_cast<double>(context.key_context_data().parms().poly_modulus_degree()))));
    long logn = logN - 1;
    if (log_sparse > 0) {
        logn = std::max(0L, logn - log_sparse);
    }
    long loge = 10;
    long boundary_K = 25;
    long deg = 59;
    long scale_factor = 2;
    long inverse_deg = 127;
    long total_level = static_cast<long>(context.key_context_data().parms().coeff_modulus().size()) - 1;

    Bootstrapper bootstrapper(loge, logn, logN - 1, total_level, ct_conv.scale(),
                              boundary_K, deg, scale_factor, inverse_deg, &ckks_evaluator);
    bootstrapper.prepare_mod_polynomial();
    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < logN - 1; i++) {
        gal_steps_vector.push_back(1 << i);
    }
    bootstrapper.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));
    bootstrapper.slot_vec.push_back(logn);
    bootstrapper.generate_LT_coefficient_3();

    return relu_coeff(context, ckks_evaluator, bootstrapper, ct_conv);
}

} // namespace phantom
