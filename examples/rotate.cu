#include <random>

#include "boot/Bootstrapper.cuh"
#include "phantom.h"

using namespace std;
using namespace phantom;

void random_real(vector<double> &vec, size_t size)
{
    random_device rn;
    mt19937_64 rnd(rn());
    thread_local std::uniform_real_distribution<double> distribution(-1, 1);

    vec.reserve(size);

    for (size_t i = 0; i < size; i++)
    {
        vec[i] = distribution(rnd);
    }
}

int main()
{
    std::cout << "Setting Parameters..." << endl;

    phantom::EncryptionParameters parms(scheme_type::ckks);
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(
        poly_modulus_degree, {60, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 60, 60, 60, 60}));
    // hybrid key-switching
    parms.set_special_modulus_size(4);
    double scale = pow(2.0, 40);

    PhantomContext context(parms);

    PhantomSecretKey secret_key(context);
    PhantomPublicKey public_key = secret_key.gen_publickey(context);
    PhantomRelinKey relin_keys = secret_key.gen_relinkey(context);
    PhantomGaloisKey galois_keys;
    PhantomCKKSEncoder encoder(context);

    CKKSEvaluator ckks_evaluator(&context, &public_key, &secret_key, &encoder, &relin_keys, &galois_keys, scale);

    vector<int> gal_steps_vector;
    gal_steps_vector.push_back(0);
    for (int i = 0; i < 10; i++)
    {
        gal_steps_vector.push_back((1 << i));
    }

    ckks_evaluator.decryptor.create_galois_keys_from_steps(gal_steps_vector, *(ckks_evaluator.galois_keys));
    vector<double> input(1 << (logN - 1));
    vector<double> before(1 << (logN - 1));
    vector<double> after(1 << (logN - 1));

    PhantomPlaintext plain;
    PhantomCiphertext cipher;

    for (int i = 0; i < sparse_slot_count; i++)
    {
        input[i] = -1 + (1. - -1) * i / sparse_slot_count;
    }
    input.resize(1 << (logN - 1));
    for (int i = sparse_slot_count; i < input.size(); i++)
    {
        input[i] = input[i % sparse_slot_count];
    }

    ckks_evaluator.encoder.encode(input, scale, plain);
    ckks_evaluator.encryptor.encrypt(plain, cipher);

    rotate_inplace(context, cipher, 4, *(ckks_evaluator.galois_keys));

    ckks_evaluator.decryptor.decrypt(rtn, plain);
    ckks_evaluator.encoder.decode(plain, after);

    for (size_t i = 0; i < sparse_slot_count; i++)
    {

        cout << " " << after[i];
    }
    std::cout << std::endl;

    return 0;
}
