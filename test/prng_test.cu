#include "phantom.h"

using namespace phantom;
using namespace phantom::arith;
using namespace phantom::util;

int main() {
    cuda_stream_wrapper stream_wrapper;
    const auto &stream = stream_wrapper.get_stream();

    constexpr size_t poly_degree = 64;
    constexpr size_t coeff_mod_size = 2;
    constexpr size_t total_uint64 = poly_degree * coeff_mod_size;

    auto h_modulus = CoeffModulus::Create(poly_degree, {50, 50});
    auto d_modulus = make_cuda_auto_ptr<DModulus>(coeff_mod_size, stream);
    for (size_t i = 0; i < coeff_mod_size; i++) {
        d_modulus.get()[i].set(h_modulus[i].value(), h_modulus[i].const_ratio()[0], h_modulus[i].const_ratio()[1]);
    }

    std::vector<uint8_t> seed(global_variables::prng_seed_byte_count, 0xAB);
    auto d_seed = make_cuda_auto_ptr<uint8_t>(global_variables::prng_seed_byte_count, stream);
    cudaMemcpyAsync(d_seed.get(), seed.data(), seed.size(), cudaMemcpyHostToDevice, stream);

    auto d_out = make_cuda_auto_ptr<uint64_t>(total_uint64, stream);
    sample_uniform_poly_wrap(d_out.get(), d_seed.get(), d_modulus.get(), poly_degree, coeff_mod_size, stream);

    std::vector<uint64_t> h_out(total_uint64);
    cudaMemcpyAsync(h_out.data(), d_out.get(), total_uint64 * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    for (size_t i = 0; i < total_uint64; i++) {
        size_t mod_idx = i / poly_degree;
        if (h_out[i] >= h_modulus[mod_idx].value()) {
            throw std::logic_error("uniform sample is out of modulus range");
        }
    }

    bool has_non_zero = false;
    for (size_t i = 0; i < total_uint64; i++) {
        if (h_out[i] != 0) {
            has_non_zero = true;
            break;
        }
    }
    if (!has_non_zero) {
        throw std::logic_error("uniform sample unexpectedly all zero");
    }

    return 0;
}
