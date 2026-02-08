#include "phantom.h"

#include <cmath>
#include <vector>

using namespace phantom;
using namespace phantom::arith

int main() {
    EncryptionParameters parms(scheme_type::ckks);
    std::size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

    PhantomContext context(parms);
    PhantomSecretKey secret_key(context);
    PhantomCKKSEncoder encoder(context);

    std::vector<double> input{1.25, -2.5, 3.75, -4.125, 5.5};
    double scale = std::pow(2.0, 30);

    PhantomCKKSEncoder::set_strict_encode_boundary_check(false);
    auto plain = encoder.encode(context, input, scale, 1);
    auto cipher = secret_key.encrypt_symmetric(context, plain);
    auto plain_out = secret_key.decrypt(context, cipher);
    auto output = encoder.decode<double>(context, plain_out);

    for (size_t i = 0; i < input.size(); i++) {
        if (std::fabs(output[i] - input[i]) > 1e-3) {
            throw std::logic_error("ckks round-trip validation failed");
        }
    }

    std::vector<double> output_async;
    encoder.decode_async(context, plain_out, output_async);
    cudaStreamSynchronize(cudaStreamPerThread);
    for (size_t i = 0; i < input.size(); i++) {
        if (std::fabs(output_async[i] - input[i]) > 1e-3) {
            throw std::logic_error("ckks async decode validation failed");
        }
    }

    PhantomCKKSEncoder::set_strict_encode_boundary_check(true);
    auto plain_checked = encoder.encode(context, input, scale, 1);
    auto cipher_checked = secret_key.encrypt_symmetric(context, plain_checked);
    auto plain_checked_out = secret_key.decrypt(context, cipher_checked);
    auto output_checked = encoder.decode<double>(context, plain_checked_out);
    for (size_t i = 0; i < input.size(); i++) {
        if (std::fabs(output_checked[i] - input[i]) > 1e-3) {
            throw std::logic_error("ckks strict boundary check validation failed");
        }
    }

    PhantomCKKSEncoder::set_strict_encode_boundary_check(false);
    return 0;
}
