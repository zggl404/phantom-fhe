#include "phantom.h"

using namespace phantom;

int main() {
    EncryptionParameters parms(scheme_type::bfv);
    std::size_t poly_modulus_degree = 4096;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, {35, 35}));
    parms.set_plain_modulus(PlainModulus::Batching(poly_modulus_degree, 20));
    parms.set_special_modulus_size(0);

    PhantomContext context(parms);

    auto &first_data = context.first_context_data();
    auto &key_data = context.key_context_data();

    if (&first_data != &key_data) {
        throw std::logic_error("first_context_data should point to key level when chain size is one");
    }

    if (context.get_first_index() != 0) {
        throw std::logic_error("first parameter index should be zero for single-level chain");
    }

    return 0;
}
