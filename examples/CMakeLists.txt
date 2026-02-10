add_executable(example_context
        1_bfv.cu
        2_bgv.cu
        3_ckks.cu
        4_kernel_fusing.cu
        example.cu
)
target_link_libraries(example_context PRIVATE Phantom)
target_include_directories(example_context PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_executable(example_serialization
        serialization.cu
)
target_link_libraries(example_serialization PRIVATE Phantom)
target_include_directories(example_serialization PUBLIC ${CMAKE_SOURCE_DIR}/include)

if (PHANTOM_ENABLE_BOOTSTRAP)
    add_executable(example_bootstrapping
            bootstrapping.cu
    )
    target_link_libraries(example_bootstrapping PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_bootstrapping PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_boot_test
            boot_test.cu
    )
    target_link_libraries(example_boot_test PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_boot_test PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_encode_coeffs
            encode_coeffs.cu
    )
    target_link_libraries(example_encode_coeffs PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_encode_coeffs PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_coeff_add_ctos
            coeff_add_ctos.cu
    )
    target_link_libraries(example_coeff_add_ctos PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_coeff_add_ctos PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_coeff_conv
            coeff_conv.cu
    )
    target_link_libraries(example_coeff_conv PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_coeff_conv PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_parms
            parms.cu
    )
    target_link_libraries(example_parms PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_parms PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_test_conv_bl
            test_conv_bl.cu
    )
    target_link_libraries(example_test_conv_bl PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_test_conv_bl PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_eval_conv_bnrelu_debug
            eval_conv_bnrelu_debug.cu
    )
    target_link_libraries(example_eval_conv_bnrelu_debug PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_eval_conv_bnrelu_debug PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_test_conv_in
            test_conv_in.cu
    )
    target_link_libraries(example_test_conv_in PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_test_conv_in PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_run_cnn
            run_cnn.cu
    )
    target_link_libraries(example_run_cnn PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_run_cnn PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_cnn_depth_30
            cnn_depth_30.cu
    )
    target_link_libraries(example_cnn_depth_30 PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_cnn_depth_30 PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_cnn_depth_34
            cnn_depth_34.cu
    )
    target_link_libraries(example_cnn_depth_34 PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_cnn_depth_34 PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_cnn_2_n_key_30
            cnn_2_n_key_30.cu
    )
    target_link_libraries(example_cnn_2_n_key_30 PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_cnn_2_n_key_30 PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_cnn_2_n_key_34
            cnn_2_n_key_34.cu
    )
    target_link_libraries(example_cnn_2_n_key_34 PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_cnn_2_n_key_34 PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})

    add_executable(example_resnet_crop_sparse
            resnet_crop_sparse.cu
    )
    target_link_libraries(example_resnet_crop_sparse PRIVATE Phantom ${NTL_LIBRARIES} ${GMP_LIBRARIES})
    target_include_directories(example_resnet_crop_sparse PUBLIC ${CMAKE_SOURCE_DIR}/include ${NTL_INCLUDE_DIRS} ${GMP_INCLUDE_DIRS})
endif ()
