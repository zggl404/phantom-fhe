#include "phantom.h"

#include <cmath>
#include <vector>

using namespace phantom;
using namespace phantom::arith;
int main() {
    EncryptionParameters parms(scheme_type::ckks);
    std::size_t poly_modulus_degree = 8192;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {60, 40, 40, 60}));

    PhantomContext context(parms);

    if (context.total_parm_size() < 2) {
        throw std::logic_error("test requires multi-level context");
    }

    context.first_parm_index_ = 0;

    auto &first_data = context.first_context_data();
    auto &level0_data = context.get_context_data(0);

    if (&first_data != &level0_data) {
        throw std::logic_error("first_context_data should follow first_parm_index_");
    }

    if (context.get_first_index() != 0) {
        throw std::logic_error("first parameter index should be zero after override");
    }

    PhantomSecretKey secret_key(context);
    PhantomCKKSEncoder encoder(context);

    std::vector<double> input{1.25, -2.5, 3.75, -4.125};
    double scale = std::pow(2.0, 30);
    auto plain = encoder.encode(context, input, scale, 1);
    auto cipher = secret_key.encrypt_symmetric(context, plain);
    auto plain_out = secret_key.decrypt(context, cipher);
    auto output = encoder.decode<double>(context, plain_out);

    for (size_t i = 0; i < input.size(); i++) {
        if (std::fabs(output[i] - input[i]) > 1e-3) {
            throw std::logic_error("ckks round-trip validation failed");
        }
    }

    return 0;
}

