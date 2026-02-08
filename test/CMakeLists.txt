add_executable(ntt_test ntt_test.cu)
target_link_libraries(ntt_test Phantom)
target_include_directories(ntt_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(context_test context_test.cu)
target_link_libraries(context_test Phantom)
target_include_directories(context_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(prng_test prng_test.cu)
target_link_libraries(prng_test Phantom)
target_include_directories(prng_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(ckks_encoder_test ckks_encoder_test.cu)
target_link_libraries(ckks_encoder_test Phantom)
target_include_directories(ckks_encoder_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(ckks_bootstrap_test ckks_bootstrap_test.cu)
target_link_libraries(ckks_bootstrap_test Phantom)
target_include_directories(ckks_bootstrap_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(ckks_eval_mod_chebyshev_test ckks_eval_mod_chebyshev_test.cu)
target_link_libraries(ckks_eval_mod_chebyshev_test Phantom)
target_include_directories(ckks_eval_mod_chebyshev_test PUBLIC ${CMAKE_SOURCE_DIR}/include)

# find CUDAToolkit to enable cuFFT
find_package(CUDAToolkit REQUIRED)

add_executable(fft_test fft_test.cu)
target_link_libraries(fft_test PUBLIC Phantom CUDA::cufft)
target_include_directories(fft_test PUBLIC ${CMAKE_SOURCE_DIR}/include)
