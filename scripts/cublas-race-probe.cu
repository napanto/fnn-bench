// Isolate the cuBLAS kernels reported by racecheck in small float parity cases.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <iostream>
#include <vector>
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess){std::cerr<<cudaGetErrorString(e)<<"\n";return 1;} } while(0)
#define BLAS(call) do { auto e=(call); if(e!=CUBLAS_STATUS_SUCCESS){std::cerr<<"cuBLAS status "<<int(e)<<"\n";return 1;} } while(0)
struct Shape { bool ta,tb; int m,n,k; };
int main() {
    cublasHandle_t h; BLAS(cublasCreate(&h)); int version; BLAS(cublasGetVersion(h,&version));
    std::cout<<"cuBLAS version "<<version<<"; one stream, synchronisation after each GEMM\n";
    const Shape shapes[]={{false,false,16,40,12},{false,false,16,40,16},{false,false,3,40,16},
        {true,false,16,40,3},{true,false,16,40,16},{false,true,16,12,40},{false,true,16,16,40},{false,true,3,16,40}};
    for(auto s:shapes) {
        std::vector<float> a(s.m*s.k,.125f),b(s.k*s.n,.25f),c(s.m*s.n);
        float *da,*db,*dc; CUDA(cudaMalloc(&da,a.size()*4));CUDA(cudaMalloc(&db,b.size()*4));CUDA(cudaMalloc(&dc,c.size()*4));
        CUDA(cudaMemcpy(da,a.data(),a.size()*4,cudaMemcpyHostToDevice));CUDA(cudaMemcpy(db,b.data(),b.size()*4,cudaMemcpyHostToDevice));
        float alpha=s.tb?1.f/40.f:1.f,beta=0;
        BLAS(cublasSgemm(h,s.ta?CUBLAS_OP_T:CUBLAS_OP_N,s.tb?CUBLAS_OP_T:CUBLAS_OP_N,
            s.m,s.n,s.k,&alpha,da,s.ta?s.k:s.m,db,s.tb?s.n:s.k,&beta,dc,s.m));
        CUDA(cudaDeviceSynchronize());CUDA(cudaMemcpy(c.data(),dc,c.size()*4,cudaMemcpyDeviceToHost));
        for(float v:c) if(std::abs(v-alpha*s.k/32.f)>1e-5f){std::cerr<<"value check failed\n";return 2;}
        std::cout<<(s.ta?'T':'N')<<(s.tb?'T':'N')<<" "<<s.m<<"x"<<s.n<<"x"<<s.k<<" checked\n";
        CUDA(cudaFree(da));CUDA(cudaFree(db));CUDA(cudaFree(dc));
    }
    BLAS(cublasDestroy(h));
}
