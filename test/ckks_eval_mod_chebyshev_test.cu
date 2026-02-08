#include "phantom.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace phantom;

namespace {

    constexpr double kPi = 3.141592653589793238462643383279502884;

    template<typename Func>
    void expect_invalid_argument(Func &&func, const char *message) {
        try {
            func();
        } catch (const std::invalid_argument &) {
            return;
        }
        throw std::logic_error(message);
    }

    double eval_mod_target(double x) {
        return std::sin(2.0 * kPi * x) / (2.0 * kPi);
    }

    void test_chebyshev_config_validation() {
        CKKSBootstrapConfig bad_degree;
        bad_degree.chebyshev_degree = 0;
        expect_invalid_argument(
                [&]() { (void) generate_eval_mod_chebyshev_coefficients(bad_degree); },
                "chebyshev_degree=0 should be rejected");

        CKKSBootstrapConfig bad_range;
        bad_range.chebyshev_min = 0.25;
        bad_range.chebyshev_max = 0.25;
        expect_invalid_argument(
                [&]() { (void) generate_eval_mod_chebyshev_coefficients(bad_range); },
                "invalid chebyshev range should be rejected");

        CKKSBootstrapConfig bad_method;
        bad_method.eval_mod_method = static_cast<CKKSEvalModMethod>(999);
        expect_invalid_argument(
                [&]() { (void) generate_eval_mod_chebyshev_coefficients(bad_method); },
                "invalid eval_mod_method should be rejected");
    }

    void test_chebyshev_approximation_accuracy() {
        CKKSBootstrapConfig config;
        config.enable_eval_mod = true;
        config.eval_mod_method = CKKSEvalModMethod::chebyshev;
        config.chebyshev_degree = 31;
        config.chebyshev_min = -0.25;
        config.chebyshev_max = 0.25;

        auto coeffs = generate_eval_mod_chebyshev_coefficients(config);
        if (coeffs.size() != config.chebyshev_degree + 1) {
            throw std::logic_error("chebyshev coefficient size mismatch");
        }

        const std::size_t samples = 4096;
        double max_error = 0.0;

        for (std::size_t i = 0; i < samples; i++) {
            const double x = config.chebyshev_min +
                             (config.chebyshev_max - config.chebyshev_min) * static_cast<double>(i) /
                                     static_cast<double>(samples - 1);
            const double approx = eval_mod_chebyshev_reference(x, config, coeffs);
            const double exact = eval_mod_target(x);
            max_error = std::max(max_error, std::fabs(approx - exact));
        }

        if (max_error > 1e-10) {
            throw std::logic_error("chebyshev approximation error is larger than expected");
        }

        expect_invalid_argument(
                [&]() { (void) eval_mod_chebyshev_reference(config.chebyshev_max + 1e-6, config, coeffs); },
                "out-of-range x should be rejected");
    }

    void test_chebyshev_odd_symmetry() {
        CKKSBootstrapConfig config;
        config.chebyshev_degree = 31;
        config.chebyshev_min = -0.25;
        config.chebyshev_max = 0.25;

        auto coeffs = generate_eval_mod_chebyshev_coefficients(config);

        const std::vector<double> sample_points{0.01, 0.05, 0.12, 0.2};
        for (double x: sample_points) {
            const double fx = eval_mod_chebyshev_reference(x, config, coeffs);
            const double fneg = eval_mod_chebyshev_reference(-x, config, coeffs);
            if (std::fabs(fx + fneg) > 1e-10) {
                throw std::logic_error("chebyshev approximation should keep odd symmetry");
            }
        }
    }

} // namespace

int main() {
    test_chebyshev_config_validation();
    test_chebyshev_approximation_accuracy();
    test_chebyshev_odd_symmetry();
    return 0;
}
