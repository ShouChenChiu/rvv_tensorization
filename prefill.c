#include <riscv_vector.h>
void sgemm_7x32(float *out, const float *a, const float *b, const int K, const float *scale) {
    vfloat32m4_t  acc0, acc1, acc2, acc3, acc4, acc5, acc6;
    size_t vl = __riscv_vsetvl_e32m4(32);

    // Init OUT zeros 
    acc0 = __riscv_vfmv_v_f_f32m4(0.f, vl);
    acc1 = __riscv_vfmv_v_f_f32m4(0.f, vl);
    acc2 = __riscv_vfmv_v_f_f32m4(0.f, vl);
    acc3 = __riscv_vfmv_v_f_f32m4(0.f, vl);
    acc4 = __riscv_vfmv_v_f_f32m4(0.f, vl);
    acc5 = __riscv_vfmv_v_f_f32m4(0.f, vl);
    acc6 = __riscv_vfmv_v_f_f32m4(0.f, vl);
    // MATMUL outer product along K 
    // Take Query[0:7,0:K] cross Key[0:K, 0:vl]
    for(int k = 0 ; k < K; ++k){
        vfloat32m4_t vb = __riscv_vle32_v_f32m4(b + 32 * k, vl);
        acc0 = __riscv_vfmacc_vf_f32m4(acc0, *(a + k + K * 0), vb, vl);
        acc1 = __riscv_vfmacc_vf_f32m4(acc1, *(a + k + K * 1), vb, vl);
        acc2 = __riscv_vfmacc_vf_f32m4(acc2, *(a + k + K * 2), vb, vl);
        acc3 = __riscv_vfmacc_vf_f32m4(acc3, *(a + k + K * 3), vb, vl);
        acc4 = __riscv_vfmacc_vf_f32m4(acc4, *(a + k + K * 4), vb, vl);
        acc5 = __riscv_vfmacc_vf_f32m4(acc5, *(a + k + K * 5), vb, vl);
        acc6 = __riscv_vfmacc_vf_f32m4(acc6, *(a + k + K * 6), vb, vl);
    }
    
    // Mul with Scale
    acc0 = __riscv_vfmul_vf_f32m4(acc0, scale[0], vl);
    acc1 = __riscv_vfmul_vf_f32m4(acc1, scale[0], vl);
    acc2 = __riscv_vfmul_vf_f32m4(acc2, scale[0], vl);
    acc3 = __riscv_vfmul_vf_f32m4(acc3, scale[0], vl);
    acc4 = __riscv_vfmul_vf_f32m4(acc4, scale[0], vl);
    acc5 = __riscv_vfmul_vf_f32m4(acc5, scale[0], vl);
    acc6 = __riscv_vfmul_vf_f32m4(acc6, scale[0], vl); 
    
    // Store out
    const int out_strides = 32;
    __riscv_vse32_v_f32m4(out + out_strides * 0, acc0, vl);
    __riscv_vse32_v_f32m4(out + out_strides * 1, acc1, vl);
    __riscv_vse32_v_f32m4(out + out_strides * 2, acc2, vl);
    __riscv_vse32_v_f32m4(out + out_strides * 3, acc3, vl);
    __riscv_vse32_v_f32m4(out + out_strides * 4, acc4, vl);
    __riscv_vse32_v_f32m4(out + out_strides * 5, acc5, vl);
    __riscv_vse32_v_f32m4(out + out_strides * 6, acc6, vl);
}


void sgemm_8x16(float *out, const float *a, const float *b, const int K, const float *scale) {
    vfloat32m2_t  acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
    size_t vl = __riscv_vsetvl_e32m2(16);

    // Init OUT zeros 
    acc0 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc1 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc2 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc3 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc4 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc5 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc6 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc7 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    // MATMUL outer product along K 
    // Take Query[0:7,0:K] cross Key[0:K, 0:vl]
    for(int k = 0 ; k < K; ++k){
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + 16 * k, vl);
        acc0 = __riscv_vfmacc_vf_f32m2(acc0, *(a + k + K * 0), vb, vl);
        acc1 = __riscv_vfmacc_vf_f32m2(acc1, *(a + k + K * 1), vb, vl);
        acc2 = __riscv_vfmacc_vf_f32m2(acc2, *(a + k + K * 2), vb, vl);
        acc3 = __riscv_vfmacc_vf_f32m2(acc3, *(a + k + K * 3), vb, vl);
        acc4 = __riscv_vfmacc_vf_f32m2(acc4, *(a + k + K * 4), vb, vl);
        acc5 = __riscv_vfmacc_vf_f32m2(acc5, *(a + k + K * 5), vb, vl);
        acc6 = __riscv_vfmacc_vf_f32m2(acc6, *(a + k + K * 6), vb, vl);
        acc7 = __riscv_vfmacc_vf_f32m2(acc7, *(a + k + K * 7), vb, vl);
    }
    
    // Mul with Scale
    acc0 = __riscv_vfmul_vf_f32m2(acc0, scale[0], vl);
    acc1 = __riscv_vfmul_vf_f32m2(acc1, scale[0], vl);
    acc2 = __riscv_vfmul_vf_f32m2(acc2, scale[0], vl);
    acc3 = __riscv_vfmul_vf_f32m2(acc3, scale[0], vl);
    acc4 = __riscv_vfmul_vf_f32m2(acc4, scale[0], vl);
    acc5 = __riscv_vfmul_vf_f32m2(acc5, scale[0], vl);
    acc6 = __riscv_vfmul_vf_f32m2(acc6, scale[0], vl); 
    acc7 = __riscv_vfmul_vf_f32m2(acc7, scale[0], vl); 

    // Store out
    const int out_strides = 16;
    __riscv_vse32_v_f32m2(out + out_strides * 0, acc0, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 1, acc1, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 2, acc2, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 3, acc3, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 4, acc4, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 5, acc5, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 6, acc6, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 7, acc7, vl);
}


void sgemm_16x16(float *out, const float *a, const float *b, const int K, const float *scale) {
    vfloat32m2_t  acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11, acc12, acc13, acc14;
    size_t vl = __riscv_vsetvl_e32m2(16);
    acc0 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc1 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc2 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc3 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc4 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc5 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc6 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc7 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc8 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc9 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc10 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc11 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc12 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc13 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc14 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    // MATMUL outer product along K 
    // Take A[0:15,0:K] cross B[0:K, 0:vl]
    for(int k = 0 ; k < K; ++k){
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + 16 * k, vl);
        acc0 = __riscv_vfmacc_vf_f32m2(acc0, *(a + k + K * 0), vb, vl);
        acc1 = __riscv_vfmacc_vf_f32m2(acc1, *(a + k + K * 1), vb, vl);
        acc2 = __riscv_vfmacc_vf_f32m2(acc2, *(a + k + K * 2), vb, vl);
        acc3 = __riscv_vfmacc_vf_f32m2(acc3, *(a + k + K * 3), vb, vl);
        acc4 = __riscv_vfmacc_vf_f32m2(acc4, *(a + k + K * 4), vb, vl);
        acc5 = __riscv_vfmacc_vf_f32m2(acc5, *(a + k + K * 5), vb, vl);
        acc6 = __riscv_vfmacc_vf_f32m2(acc6, *(a + k + K * 6), vb, vl);
        acc7 = __riscv_vfmacc_vf_f32m2(acc7, *(a + k + K * 7), vb, vl);
        acc8 = __riscv_vfmacc_vf_f32m2(acc8, *(a + k + K * 8), vb, vl);
        acc9 = __riscv_vfmacc_vf_f32m2(acc9, *(a + k + K * 9), vb, vl);
        acc10 = __riscv_vfmacc_vf_f32m2(acc10, *(a + k + K * 10), vb, vl);
        acc11 = __riscv_vfmacc_vf_f32m2(acc11, *(a + k + K * 11), vb, vl);
        acc12 = __riscv_vfmacc_vf_f32m2(acc12, *(a + k + K * 12), vb, vl);
        acc13 = __riscv_vfmacc_vf_f32m2(acc13, *(a + k + K * 13), vb, vl);
        acc14 = __riscv_vfmacc_vf_f32m2(acc14, *(a + k + K * 14), vb, vl);
    }

    // Mul with Scale
    acc0 = __riscv_vfmul_vf_f32m2(acc0, scale[0], vl);
    acc1 = __riscv_vfmul_vf_f32m2(acc1, scale[0], vl);
    acc2 = __riscv_vfmul_vf_f32m2(acc2, scale[0], vl);
    acc3 = __riscv_vfmul_vf_f32m2(acc3, scale[0], vl);
    acc4 = __riscv_vfmul_vf_f32m2(acc4, scale[0], vl);
    acc5 = __riscv_vfmul_vf_f32m2(acc5, scale[0], vl);
    acc6 = __riscv_vfmul_vf_f32m2(acc6, scale[0], vl); 
    acc7 = __riscv_vfmul_vf_f32m2(acc7, scale[0], vl);
    acc8 = __riscv_vfmul_vf_f32m2(acc8, scale[0], vl);
    acc9 = __riscv_vfmul_vf_f32m2(acc9, scale[0], vl);
    acc10 = __riscv_vfmul_vf_f32m2(acc10, scale[0], vl);
    acc11 = __riscv_vfmul_vf_f32m2(acc11, scale[0], vl);
    acc12 = __riscv_vfmul_vf_f32m2(acc12, scale[0], vl);
    acc13 = __riscv_vfmul_vf_f32m2(acc13, scale[0], vl); 
    acc14 = __riscv_vfmul_vf_f32m2(acc14, scale[0], vl); 
    // Store out
    const int out_strides = 16;
    __riscv_vse32_v_f32m2(out + out_strides * 0, acc0, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 1, acc1, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 2, acc2, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 3, acc3, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 4, acc4, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 5, acc5, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 6, acc6, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 7, acc7, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 8, acc8, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 9, acc9, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 10, acc10, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 11, acc11, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 12, acc12, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 13, acc13, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 14, acc14, vl);

    acc0 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    for(int k = 0 ; k < K; ++k){
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + 16 * k, vl);
        acc0 = __riscv_vfmacc_vf_f32m2(acc0, *(a + k + K * 15), vb, vl);
    }
    acc0 = __riscv_vfmul_vf_f32m2(acc0, scale[0], vl);
    __riscv_vse32_v_f32m2(out + out_strides * 15, acc0, vl);
}

void sgemm_15x16(float *out, const float *a, const float *b, const int K, const float *scale) {
    vfloat32m2_t  acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7, acc8, acc9, acc10, acc11, acc12, acc13, acc14;
    size_t vl = __riscv_vsetvl_e32m2(16);
    acc0 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc1 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc2 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc3 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc4 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc5 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc6 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc7 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc8 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc9 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc10 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc11 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc12 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc13 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    acc14 = __riscv_vfmv_v_f_f32m2(0.f, vl);
    // MATMUL outer product along K 
    // Take A[0:15,0:K] cross B[0:K, 0:vl]
    for(int k = 0 ; k < K; ++k){
        vfloat32m2_t vb = __riscv_vle32_v_f32m2(b + 16 * k, vl);
        acc0 = __riscv_vfmacc_vf_f32m2(acc0, *(a + k + K * 0), vb, vl);
        acc1 = __riscv_vfmacc_vf_f32m2(acc1, *(a + k + K * 1), vb, vl);
        acc2 = __riscv_vfmacc_vf_f32m2(acc2, *(a + k + K * 2), vb, vl);
        acc3 = __riscv_vfmacc_vf_f32m2(acc3, *(a + k + K * 3), vb, vl);
        acc4 = __riscv_vfmacc_vf_f32m2(acc4, *(a + k + K * 4), vb, vl);
        acc5 = __riscv_vfmacc_vf_f32m2(acc5, *(a + k + K * 5), vb, vl);
        acc6 = __riscv_vfmacc_vf_f32m2(acc6, *(a + k + K * 6), vb, vl);
        acc7 = __riscv_vfmacc_vf_f32m2(acc7, *(a + k + K * 7), vb, vl);
        acc8 = __riscv_vfmacc_vf_f32m2(acc8, *(a + k + K * 8), vb, vl);
        acc9 = __riscv_vfmacc_vf_f32m2(acc9, *(a + k + K * 9), vb, vl);
        acc10 = __riscv_vfmacc_vf_f32m2(acc10, *(a + k + K * 10), vb, vl);
        acc11 = __riscv_vfmacc_vf_f32m2(acc11, *(a + k + K * 11), vb, vl);
        acc12 = __riscv_vfmacc_vf_f32m2(acc12, *(a + k + K * 12), vb, vl);
        acc13 = __riscv_vfmacc_vf_f32m2(acc13, *(a + k + K * 13), vb, vl);
        acc14 = __riscv_vfmacc_vf_f32m2(acc14, *(a + k + K * 14), vb, vl);
    }

    // Mul with Scale
    acc0 = __riscv_vfmul_vf_f32m2(acc0, scale[0], vl);
    acc1 = __riscv_vfmul_vf_f32m2(acc1, scale[0], vl);
    acc2 = __riscv_vfmul_vf_f32m2(acc2, scale[0], vl);
    acc3 = __riscv_vfmul_vf_f32m2(acc3, scale[0], vl);
    acc4 = __riscv_vfmul_vf_f32m2(acc4, scale[0], vl);
    acc5 = __riscv_vfmul_vf_f32m2(acc5, scale[0], vl);
    acc6 = __riscv_vfmul_vf_f32m2(acc6, scale[0], vl); 
    acc7 = __riscv_vfmul_vf_f32m2(acc7, scale[0], vl);
    acc8 = __riscv_vfmul_vf_f32m2(acc8, scale[0], vl);
    acc9 = __riscv_vfmul_vf_f32m2(acc9, scale[0], vl);
    acc10 = __riscv_vfmul_vf_f32m2(acc10, scale[0], vl);
    acc11 = __riscv_vfmul_vf_f32m2(acc11, scale[0], vl);
    acc12 = __riscv_vfmul_vf_f32m2(acc12, scale[0], vl);
    acc13 = __riscv_vfmul_vf_f32m2(acc13, scale[0], vl); 
    acc14 = __riscv_vfmul_vf_f32m2(acc14, scale[0], vl); 
    // Store out
    const int out_strides = 16;
    __riscv_vse32_v_f32m2(out + out_strides * 0, acc0, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 1, acc1, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 2, acc2, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 3, acc3, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 4, acc4, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 5, acc5, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 6, acc6, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 7, acc7, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 8, acc8, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 9, acc9, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 10, acc10, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 11, acc11, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 12, acc12, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 13, acc13, vl);
    __riscv_vse32_v_f32m2(out + out_strides * 14, acc14, vl);
}

void ogemm_2x64(float *out, const float *a, const float *b, const int K) {
    vfloat32m8_t  acc0, acc1;
    size_t vl = __riscv_vsetvl_e32m8(64);

    // Init OUT zeros 
    const int out_strides = 64;
    acc0 = __riscv_vle32_v_f32m8(out + out_strides * 0, vl);
    acc1 = __riscv_vle32_v_f32m8(out + out_strides * 1, vl);
    // MATMUL outer product along K 
    // Take Query[0:7,0:K] cross Key[0:K, 0:vl]
    for(int k = 0 ; k < K; ++k){
        vfloat32m8_t vb = __riscv_vle32_v_f32m8(b + 64 * k, vl);
        acc0 = __riscv_vfmacc_vf_f32m8(acc0, *(a + k + K * 0), vb, vl);
        acc1 = __riscv_vfmacc_vf_f32m8(acc1, *(a + k + K * 1), vb, vl);
    }    
    // Store out
    __riscv_vse32_v_f32m8(out + out_strides * 0, acc0, vl);
    __riscv_vse32_v_f32m8(out + out_strides * 1, acc1, vl);
}



