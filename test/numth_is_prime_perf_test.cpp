#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

namespace {
uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t mod) {
    return static_cast<uint64_t>((static_cast<unsigned __int128>(a) * b) % mod);
}

uint64_t pow_mod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    while (exp) {
        if (exp & 1) {
            result = mul_mod(result, base, mod);
        }
        base = mul_mod(base, base, mod);
        exp >>= 1;
    }
    return result;
}

bool is_prime_old(uint64_t value, size_t num_rounds = 40) {
    if (value < 2) {
        return false;
    }
    if (value == 2) {
        return true;
    }
    if ((value & 1ULL) == 0) {
        return false;
    }
    if (value == 3) {
        return true;
    }
    if (value % 3 == 0) {
        return false;
    }
    if (value == 5) {
        return true;
    }
    if (value % 5 == 0) {
        return false;
    }
    if (value == 7) {
        return true;
    }
    if (value % 7 == 0) {
        return false;
    }
    if (value == 11) {
        return true;
    }
    if (value % 11 == 0) {
        return false;
    }
    if (value == 13) {
        return true;
    }
    if (value % 13 == 0) {
        return false;
    }

    uint64_t d = value - 1;
    uint64_t r = 0;
    while ((d & 1ULL) == 0) {
        d >>= 1;
        ++r;
    }
    if (r == 0) {
        return false;
    }

    std::random_device rand;
    std::uniform_int_distribution<unsigned long long> dist(3, value - 1);
    for (size_t i = 0; i < num_rounds; ++i) {
        uint64_t a = i ? dist(rand) : 2;
        uint64_t x = pow_mod(a, d, value);
        if (x == 1 || x == value - 1) {
            continue;
        }

        uint64_t count = 0;
        do {
            x = mul_mod(x, x, value);
            ++count;
        } while (x != value - 1 && count < r - 1);

        if (x != value - 1) {
            return false;
        }
    }
    return true;
}

bool is_prime_new(uint64_t value, size_t num_rounds = 40) {
    if (value < 2) {
        return false;
    }
    if (value == 2) {
        return true;
    }
    if ((value & 1ULL) == 0) {
        return false;
    }
    if (value == 3) {
        return true;
    }
    if (value % 3 == 0) {
        return false;
    }
    if (value == 5) {
        return true;
    }
    if (value % 5 == 0) {
        return false;
    }
    if (value == 7) {
        return true;
    }
    if (value % 7 == 0) {
        return false;
    }
    if (value == 11) {
        return true;
    }
    if (value % 11 == 0) {
        return false;
    }
    if (value == 13) {
        return true;
    }
    if (value % 13 == 0) {
        return false;
    }

    uint64_t d = value - 1;
    uint64_t r = 0;
    while ((d & 1ULL) == 0) {
        d >>= 1;
        ++r;
    }
    if (r == 0) {
        return false;
    }

    static constexpr std::array<uint64_t, 7> kWitnesses = {
        2ULL, 325ULL, 9375ULL, 28178ULL, 450775ULL, 9780504ULL, 1795265022ULL
    };

    const size_t rounds = std::min(num_rounds, kWitnesses.size());
    if (rounds == 0) {
        return true;
    }

    for (size_t i = 0; i < rounds; ++i) {
        uint64_t a = kWitnesses[i] % value;
        if (a <= 1) {
            continue;
        }

        uint64_t x = pow_mod(a, d, value);
        if (x == 1 || x == value - 1) {
            continue;
        }

        uint64_t count = 0;
        do {
            x = mul_mod(x, x, value);
            ++count;
        } while (x != value - 1 && count < r - 1);

        if (x != value - 1) {
            return false;
        }
    }

    return true;
}

uint64_t reduction_threshold_new(uint64_t q_msb) {
    const uint64_t bits_per_uint64 = 64;
    const uint64_t shift = bits_per_uint64 - q_msb - 1;
    return (uint64_t(1) << shift) - 1;
}
}

int main() {
    constexpr size_t kCount = 12000;
    std::vector<uint64_t> candidates;
    candidates.reserve(kCount);
    uint64_t start = (1ULL << 52) + 1;
    if ((start & 1ULL) == 0) {
        ++start;
    }
    for (size_t i = 0; i < kCount; ++i) {
        candidates.push_back(start + (i * 2ULL));
    }

    std::vector<uint8_t> old_results(kCount);
    std::vector<uint8_t> new_results(kCount);

    const auto old_begin = std::chrono::steady_clock::now();
    size_t old_prime_count = 0;
    for (size_t i = 0; i < kCount; ++i) {
        old_results[i] = static_cast<uint8_t>(is_prime_old(candidates[i]));
        old_prime_count += old_results[i];
    }
    const auto old_end = std::chrono::steady_clock::now();

    const auto new_begin = std::chrono::steady_clock::now();
    size_t new_prime_count = 0;
    for (size_t i = 0; i < kCount; ++i) {
        new_results[i] = static_cast<uint8_t>(is_prime_new(candidates[i]));
        new_prime_count += new_results[i];
    }
    const auto new_end = std::chrono::steady_clock::now();

    size_t mismatch = 0;
    for (size_t i = 0; i < kCount; ++i) {
        mismatch += static_cast<size_t>(old_results[i] != new_results[i]);
    }

    const auto old_us = std::chrono::duration_cast<std::chrono::microseconds>(old_end - old_begin).count();
    const auto new_us = std::chrono::duration_cast<std::chrono::microseconds>(new_end - new_begin).count();
    const double speedup = static_cast<double>(old_us) / static_cast<double>(new_us);

    std::cout << "Candidates: " << kCount << '\n';
    std::cout << "Old primes: " << old_prime_count << ", New primes: " << new_prime_count << '\n';
    std::cout << "Result mismatch count: " << mismatch << '\n';
    std::cout << "Old duration (us): " << old_us << '\n';
    std::cout << "New duration (us): " << new_us << '\n';
    std::cout << "Speedup: " << speedup << "x" << '\n';

    std::cout << "\nReduction-threshold checks:" << '\n';
    for (uint64_t q_msb : {20ULL, 30ULL, 40ULL, 50ULL, 60ULL}) {
        const uint64_t shift = 64ULL - q_msb - 1ULL;
        const uint64_t new_v = reduction_threshold_new(q_msb);

        std::cout << "qMSB=" << q_msb << " shift=" << shift;
        if (shift >= 32ULL) {
            std::cout << " old=UB-risk";
        } else {
            const uint64_t old_safe_v = (uint64_t(1U) << shift) - 1U;
            std::cout << " old=" << old_safe_v;
        }
        std::cout << " new=" << new_v << '\n';
    }

    return mismatch == 0 ? 0 : 1;
}
