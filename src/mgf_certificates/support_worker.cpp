// Complete public-window support statistics; no floating exponential or scorer.
// Compiles on 64-bit GCC/Clang (uses unsigned __int128, OpenMP, OpenSSL libcrypto):
//   g++ -O3 -std=c++17 -fopenmp -Wno-deprecated-declarations support_worker.cpp -lcrypto -o support_worker
//
// Binary formats (little-endian except big-endian 32-bit multiplier limbs in MGFPUBL1):
//   1. PUBLIC input (MGFPUBL1):
//      [0..7]   magic = "MGFPUBL1"
//      [8..47]  10 x uint32: n (primes), W (windows), m (input bits), w (window width 1..12),
//               ell (prime bits 2..24), f (scale bits = 39), lambda (1..64),
//               lambda_pow (= 24), limbs (> 0), entries (sum of counts)
//      counts : W x uint32 (each a power of two; sum of log2 == m, sum == entries)
//      primes : n x uint32 (strictly increasing odd primes < 2^ell)
//      values : entries x limbs x big-endian uint32 (multipliers reduced mod p)
//   2. LUT input:
//      ((2^ell) - 1) x uint64 exact upward recurrence e[i] = e[i-1] + ceil(e[i-1]*delta / 2^39)
//   3. OUTPUT ledger (MGFLEDG1):
//      [0..7]   magic = "MGFLEDG1"
//      [8..31]  6 x uint32: f, W, count (last-first), first, last, n
//      [32..95] 32-byte SHA-256(PUBLIC) + 32-byte SHA-256(LUT)
//      records  (last - first) entries, each:
//               4 x u32 (index, p, g, center), u64 mean_numerator, u32 2^w, u32 0,
//               32-byte support_sha, W x u64 plus sums, W x u64 minus sums
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <openssl/sha.h>
#ifdef _OPENMP
#include <omp.h>
#endif

static uint32_t read32(std::istream& in) {
    uint8_t b[4]; in.read(reinterpret_cast<char*>(b), 4);
    if (!in) throw std::runtime_error("truncated uint32 input");
    return uint32_t(b[0]) | uint32_t(b[1])<<8 | uint32_t(b[2])<<16 | uint32_t(b[3])<<24;
}
static uint64_t read64(std::istream& in) {
    uint64_t lo=read32(in), hi=read32(in); return lo | hi<<32;
}
static void write32(std::ostream& out, uint32_t x) {
    uint8_t b[4]; for(int k=0;k<4;k++) b[k]=uint8_t(x>>(8*k));
    out.write(reinterpret_cast<char*>(b),4);
}
static void write64(std::ostream& out, uint64_t x) {
    write32(out,uint32_t(x)); write32(out,uint32_t(x>>32));
}
static std::array<unsigned char,32> file_sha(const std::string& path) {
    std::ifstream in(path,std::ios::binary); if(!in) throw std::runtime_error("missing input file");
    SHA256_CTX ctx; SHA256_Init(&ctx);
    std::array<char,1<<16> buffer;
    while(in) { in.read(buffer.data(),buffer.size()); auto n=in.gcount(); if(n) SHA256_Update(&ctx,buffer.data(),size_t(n)); }
    std::array<unsigned char,32> result; SHA256_Final(result.data(),&ctx); return result;
}
static uint32_t powmod(uint32_t a,uint32_t exponent,uint32_t p) {
    uint64_t result=1, base=a;
    while(exponent) { if(exponent&1) result=result*base%p; base=base*base%p; exponent>>=1; }
    return uint32_t(result);
}
static bool prime(uint32_t p) {
    if(p<2) return false; if(p%2==0) return p==2;
    for(uint32_t d=3;uint64_t(d)*d<=p;d+=2) if(p%d==0) return false;
    return true;
}
static uint32_t least_generator(uint32_t p) {
    uint32_t v=p-1; std::vector<uint32_t> factors;
    for(uint32_t q=2;uint64_t(q)*q<=v;q++) if(v%q==0) {
        factors.push_back(q); while(v%q==0) v/=q;
    }
    if(v>1) factors.push_back(v);
    for(uint32_t g=2;g<p;g++) {
        bool ok=true; for(auto q:factors) if(powmod(g,(p-1)/q,p)==1) {ok=false;break;}
        if(ok) return g;
    }
    throw std::runtime_error("no primitive root");
}
struct Input {
    uint32_t n,W,m,w,ell,f,lambda,lambda_pow,limbs,entries;
    std::vector<uint32_t> counts,primes,values;
};
static Input load_public(const std::string& path) {
    std::ifstream in(path,std::ios::binary); char magic[8];in.read(magic,8);
    if(std::string(magic,8)!="MGFPUBL1") throw std::runtime_error("public magic mismatch");
    Input d; uint32_t* fields[]={&d.n,&d.W,&d.m,&d.w,&d.ell,&d.f,&d.lambda,&d.lambda_pow,&d.limbs,&d.entries};
    for(auto p:fields) *p=read32(in);
    if(d.f!=39||!d.n||!d.W||!d.lambda||d.lambda>64||d.lambda_pow!=24||d.ell<2||d.ell>24||d.w<1||d.w>12||!d.limbs) throw std::runtime_error("unsupported exact input parameters");
    d.counts.resize(d.W);d.primes.resize(d.n); uint64_t entries=0,bits=0;
    for(auto& q:d.counts) {q=read32(in);if(!q||(q&(q-1)))throw std::runtime_error("bad address domain");entries+=q;for(auto v=q;v>1;v>>=1)bits++;}
    if(entries!=d.entries||bits!=d.m) throw std::runtime_error("address counts do not cover the actual input bits");
    for(auto& p:d.primes) p=read32(in);
    if(!std::is_sorted(d.primes.begin(),d.primes.end())||std::adjacent_find(d.primes.begin(),d.primes.end())!=d.primes.end()) throw std::runtime_error("RNS ordering/uniqueness failed");
    d.values.resize(size_t(d.entries)*d.limbs);
    for(auto& v:d.values) {uint8_t b[4];in.read(reinterpret_cast<char*>(b),4);if(!in)throw std::runtime_error("truncated multipliers");v=uint32_t(b[0])<<24|uint32_t(b[1])<<16|uint32_t(b[2])<<8|uint32_t(b[3]);}
    if(in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("extra public bytes");
    return d;
}
static uint64_t rate_delta(const Input& d) {
    // Exact ceil(F*(lambda + lambda^2/2 + lambda^3/(6*(1-lambda/4)))).
    // For 1<=a<=64, D=2^24 and F=2^39, every intermediate is <2^101.
    typedef unsigned __int128 Wide;
    const Wide a=d.lambda,D=Wide(1)<<24,F=Wide(1)<<39;
    const Wide denominator=6*D*D*(4*D-a);
    const Wide numerator=6*a*D*(4*D-a)+3*a*a*(4*D-a)+4*a*a*a;
    const Wide delta=(F*numerator+denominator-1)/denominator;
    if(delta>std::numeric_limits<uint64_t>::max())throw std::runtime_error("rate delta overflow");
    return uint64_t(delta);
}
static std::vector<uint64_t> load_lut(const std::string& path,const Input& d) {
    std::ifstream in(path,std::ios::binary); if(!in)throw std::runtime_error("missing exponential LUT");
    std::vector<uint64_t> e((uint64_t(1)<<d.ell)-1);
    for(auto& v:e) v=read64(in);
    if(in.peek()!=std::char_traits<char>::eof()) throw std::runtime_error("extra LUT bytes");
    const uint64_t F=uint64_t(1)<<39, delta=rate_delta(d), limit=std::numeric_limits<uint64_t>::max();
    if(e[0]!=F)throw std::runtime_error("bad exponential base");
    for(size_t i=1;i<e.size();i++) {
        if(e[i-1]>(limit-(F-1))/delta)throw std::runtime_error("native LUT product can overflow");
        uint64_t expected=e[i-1]+((e[i-1]*delta+F-1)>>39);
        if(e[i]!=expected)throw std::runtime_error("nonconforming exponential recurrence");
    }
    auto q=*std::max_element(d.counts.begin(),d.counts.end());
    if(e.back()>limit/q)throw std::runtime_error("a native window sum can overflow");
    return e;
}
struct Result {
    uint32_t index,p,g,center;
    uint64_t mean_numerator;
    std::array<unsigned char,32> support_sha;
    std::vector<uint64_t> plus,minus;
};
static Result compute(const Input& d,const std::vector<uint64_t>& e,uint32_t index,std::vector<uint32_t>& logs) {
    uint32_t p=d.primes[index]; if(!prime(p)||p<3||p>=(uint64_t(1)<<d.ell))throw std::runtime_error("nonprime/wrong support width");
    uint32_t g=least_generator(p);uint64_t value=1;
    for(uint32_t exponent=0;exponent<p-1;exponent++){logs[size_t(value)]=exponent;value=value*g%p;}
    if(value!=1)throw std::runtime_error("generator enumeration did not close");
    Result r;r.index=index;r.p=p;r.g=g;r.mean_numerator=0;r.plus.resize(d.W);r.minus.resize(d.W);
    SHA256_CTX digest;SHA256_Init(&digest);std::vector<uint8_t> raw(4*(*std::max_element(d.counts.begin(),d.counts.end())));
    uint64_t entry=0;const uint64_t limit=std::numeric_limits<uint64_t>::max();
    for(uint32_t w=0;w<d.W;w++) {
        uint64_t plus=0,minus=0,sum=0;
        for(uint32_t a=0;a<d.counts[w];a++,entry++) {
            uint64_t residue=0;const uint32_t* limbs=&d.values[size_t(entry)*d.limbs];
            for(uint32_t j=0;j<d.limbs;j++) residue=((residue<<32)|limbs[j])%p;
            if(!residue)throw std::runtime_error("public multiplier is zero modulo prime");
            uint32_t y=logs[size_t(residue)];if(y>p-2)throw std::runtime_error("dlog is outside canonical support");
            uint64_t yp=e[y],ym=e[p-2-y];
            if(plus>limit-yp||minus>limit-ym)throw std::runtime_error("window accumulation overflow");
            plus+=yp;minus+=ym;sum+=y;
            for(int b=0;b<4;b++)raw[4*a+b]=uint8_t(y>>(8*b));
        }
        SHA256_Update(&digest,raw.data(),4*d.counts[w]);
        uint64_t weight=(uint64_t(1)<<d.w)/d.counts[w];
        if(sum>limit/weight||r.mean_numerator>limit-sum*weight)throw std::runtime_error("mean overflow");
        r.mean_numerator+=sum*weight;r.plus[w]=plus;r.minus[w]=minus;
    }
    uint64_t center=(r.mean_numerator+(uint64_t(1)<<(d.w-1)))>>d.w;
    if(center>std::numeric_limits<uint32_t>::max())throw std::runtime_error("center overflow");
    r.center=uint32_t(center);SHA256_Final(r.support_sha.data(),&digest);return r;
}
int main(int argc,char** argv) {
    try {
        if(argc!=7) throw std::runtime_error("usage: support_worker PUBLIC LUT OUTPUT START STOP THREADS");
        auto started=std::chrono::steady_clock::now();
        std::string public_path=argv[1],lut_path=argv[2],output_path=argv[3];
        Input d=load_public(public_path);auto lut=load_lut(lut_path,d);
        uint32_t first=uint32_t(std::stoul(argv[4])),last=uint32_t(std::stoul(argv[5]));
        int threads=std::stoi(argv[6]);if(first>=last||last>d.n||threads<1)throw std::runtime_error("invalid work range");
        std::vector<Result> results(last-first);std::atomic<bool> failed{false};std::atomic<uint32_t> completed{0};std::string failure;
#ifdef _OPENMP
        omp_set_num_threads(threads);
#pragma omp parallel
#endif
        {
            std::vector<uint32_t> logs(uint64_t(1)<<d.ell);
#ifdef _OPENMP
#pragma omp for schedule(dynamic,1)
#endif
            for(int64_t index=first;index<last;index++) {
                if(failed.load())continue;
                try {
                    results[size_t(index-first)]=compute(d,lut,uint32_t(index),logs);
                    auto done=++completed;
                    if(done%256==0) {
#ifdef _OPENMP
#pragma omp critical
#endif
                        std::cerr<<"{\"done\":"<<done<<",\"total\":"<<(last-first)<<"}\n";
                    }
                }
                catch(const std::exception& error) {
#ifdef _OPENMP
#pragma omp critical
#endif
                    {failed=true;failure=error.what();}
                }
            }
        }
        if(failed)throw std::runtime_error(failure);
        auto input_digest=file_sha(public_path),lut_digest=file_sha(lut_path);
        std::ofstream out(output_path,std::ios::binary);if(!out)throw std::runtime_error("cannot create output");
        out.write("MGFLEDG1",8);for(auto v:{d.f,d.W,last-first,first,last,d.n})write32(out,v);
        out.write(reinterpret_cast<const char*>(input_digest.data()),32);out.write(reinterpret_cast<const char*>(lut_digest.data()),32);
        for(const auto& r:results) {
            for(auto v:{r.index,r.p,r.g,r.center})write32(out,v);
            write64(out,r.mean_numerator);write32(out,uint32_t(1)<<d.w);write32(out,0);
            out.write(reinterpret_cast<const char*>(r.support_sha.data()),32);
            for(auto v:r.plus)write64(out,v);for(auto v:r.minus)write64(out,v);
        }
        out.close();if(!out)throw std::runtime_error("ledger write failed");
        double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        std::cout<<"{\"status\":\"complete\",\"first\":"<<first<<",\"last\":"<<last<<",\"primes\":"<<(last-first)<<",\"threads\":"<<threads<<",\"seconds\":"<<seconds<<"}\n";
        return 0;
    } catch(const std::exception& error) {std::cerr<<"support worker: "<<error.what()<<"\n";return 2;}
}
