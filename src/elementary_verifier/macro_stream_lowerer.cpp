// Stateless Macro-to-Elementary Gate Stream Lowerer (portable ISO C++17 binary).
// Compiles on Linux / macOS / Windows with zero external dependencies:
//   g++ -O3 -std=c++17 macro_stream_lowerer.cpp -o macro_stream_lowerer
//
// Reads wire-ID macro instructions from stdin and writes the exact bit-for-bit
// EGKMV1 flat 32-bit elementary gate stream to stdout.
//
// Input macro binary protocol (little-endian):
//   Prefix: uint32 hdr_len (>= 28), followed by hdr_len raw EGKMV1 header bytes
//           (passed verbatim to stdout; bytes [8..15] supply the SplitMix64 MX seed).
//   Records: 1-byte macro opcode followed by its payload:
//     0  = M_EOF             : (no payload) emits EGKMV1 OP_EOF and exits 0
//     1  = M_RAW_WORDS       : uint32 cnt + cnt x uint32 pre-encoded EGKMV1 words
//     2  = M_ADD_QQ          : uint16 n, uint16 sc, n x u16 a, n x u16 b
//     3  = M_SUB_QQ          : uint16 n, uint16 sc, n x u16 a, n x u16 b
//     4  = M_ADD_CONST       : uint16 n (<=64), uint16 sc, uint64 c, n x u16 a
//     5  = M_CMP_GE_PHASE    : uint16 n, uint16 sc, n x u16 a, n x u16 b
//     6  = M_QROM_INIT       : uint16 a (<=20), uint16 w, uint16 n_limbs, uint16 sc,
//                              a x u16 addr, w x u16 out_w, (2^a * n_limbs) x u64 table
//     7  = M_SQRT_PHASEUP    : uint16 a (<=24), uint16 sc, a x u16 addr, 2^a x u8 truth
//     8  = M_CX_VEC          : uint16 n, n x u16 ctrl, n x u16 tgt
//     9  = M_X_MASK          : uint16 n, uint16 n_limbs, n x u16 w, n_limbs x u64 mask
//     10 = M_Z_VEC           : uint16 n, n x u16 w
//     11 = M_CZ_VEC          : uint16 n, n x u16 a, n x u16 b
//     12 = M_ALLOC_VEC       : uint16 n, n x u16 w
//     13 = M_FREE_ZERO_VEC   : uint16 n, n x u16 w
//     14 = M_MX_FREE_VEC     : uint16 n, n x u16 w
//     15 = M_MX_REALLOC_VEC  : uint16 n, n x u16 w

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

constexpr uint32_t OP_EOF = 0;
constexpr uint32_t OP_ALLOC = 1;
constexpr uint32_t OP_FREE_ZERO = 2;
constexpr uint32_t OP_X = 3;
constexpr uint32_t OP_CX = 4;
constexpr uint32_t OP_Z = 5;
constexpr uint32_t OP_CZ = 6;
constexpr uint32_t OP_GLOBAL_Z = 7;
constexpr uint32_t OP_AND = 8;
constexpr uint32_t OP_MX_FREE = 9;

inline int tzcnt64(uint64_t v) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_ctzll(v);
#else
    int c = 0;
    while ((v & 1ULL) == 0ULL) { v >>= 1; ++c; }
    return c;
#endif
}

inline int bit_length32(int v) {
    int r = 0;
    while (v > 0) { v >>= 1; ++r; }
    return r;
}

void read_exact(std::FILE* fp, void* dst, size_t count) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    while (count > 0) {
        size_t n = std::fread(p, 1, count, fp);
        if (n == 0) throw std::runtime_error("Unexpected EOF in macro stream");
        p += n;
        count -= n;
    }
}

inline uint16_t read_u16_le(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

inline uint32_t read_u32_le(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

inline uint64_t read_u64_le(const uint8_t* p) {
    return uint64_t(read_u32_le(p)) | (uint64_t(read_u32_le(p + 4)) << 32);
}

struct OutBuffer {
    std::FILE* out;
    std::vector<uint8_t> buf;
    size_t pos = 0;
    uint64_t rng_state;

    OutBuffer(std::FILE* fp, uint64_t seed) : out(fp), buf(1 << 20), rng_state(seed) {}

    void flush() {
        if (pos > 0) {
            if (std::fwrite(buf.data(), 1, pos, out) != pos)
                throw std::runtime_error("Write failed");
            pos = 0;
        }
    }

    void write_raw(const void* data, size_t n) {
        flush();
        if (std::fwrite(data, 1, n, out) != n)
            throw std::runtime_error("Write raw failed");
    }

    inline void emit1(uint32_t w) {
        if (pos + 4 > buf.size()) flush();
        buf[pos] = uint8_t(w);
        buf[pos + 1] = uint8_t(w >> 8);
        buf[pos + 2] = uint8_t(w >> 16);
        buf[pos + 3] = uint8_t(w >> 24);
        pos += 4;
    }

    inline void emit2(uint32_t w1, uint32_t w2) {
        if (pos + 8 > buf.size()) flush();
        buf[pos] = uint8_t(w1);
        buf[pos + 1] = uint8_t(w1 >> 8);
        buf[pos + 2] = uint8_t(w1 >> 16);
        buf[pos + 3] = uint8_t(w1 >> 24);
        buf[pos + 4] = uint8_t(w2);
        buf[pos + 5] = uint8_t(w2 >> 8);
        buf[pos + 6] = uint8_t(w2 >> 16);
        buf[pos + 7] = uint8_t(w2 >> 24);
        pos += 8;
    }

    inline uint32_t next_bit() {
        rng_state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = rng_state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        return uint32_t((z ^ (z >> 31)) & 1ULL);
    }

    inline void alloc(int w) { emit1(OP_ALLOC | (uint32_t(w) << 4)); }
    inline void free_zero(int w) { emit1(OP_FREE_ZERO | (uint32_t(w) << 4)); }
    inline void x(int w) { emit1(OP_X | (uint32_t(w) << 4)); }
    inline void cx(int c, int t) { emit1(OP_CX | (uint32_t(c) << 4) | (uint32_t(t) << 18)); }
    inline void z(int w) { emit1(OP_Z | (uint32_t(w) << 4)); }
    inline void cz(int a, int b) { emit1(OP_CZ | (uint32_t(a) << 4) | (uint32_t(b) << 18)); }
    inline void global_z() { emit1(OP_GLOBAL_Z); }
    inline void and_gate(int c1, int c2, int t) { emit2(OP_AND | (uint32_t(c1) << 4) | (uint32_t(c2) << 18), uint32_t(t)); }
    inline uint32_t mx_free(int w) {
        uint32_t m = next_bit();
        emit1(OP_MX_FREE | (uint32_t(w) << 4) | (m << 18));
        return m;
    }
};

void read_wires(std::FILE* fp, std::vector<uint8_t>& io_buf, uint16_t* dst, int n) {
    if (n < 0 || n > 4096) throw std::runtime_error("Wire vector length out of range (max 4096)");
    size_t bytes = size_t(n) * 2;
    if (io_buf.size() < bytes) io_buf.resize(bytes);
    read_exact(fp, io_buf.data(), bytes);
    for (int i = 0; i < n; ++i) dst[i] = read_u16_le(io_buf.data() + 2 * i);
}

void emit_add_qq(OutBuffer& ob, const uint16_t* a, const uint16_t* b, int n, int sc) {
    if (n == 0) return;
    if (n == 1) { ob.cx(b[0], a[0]); return; }
    int c0 = sc;
    ob.alloc(c0);
    ob.and_gate(a[0], b[0], c0);
    for (int k = 1; k < n - 1; ++k) {
        int c_prev = sc + k - 1, ck = sc + k;
        ob.cx(c_prev, a[k]);
        ob.cx(c_prev, b[k]);
        ob.alloc(ck);
        ob.and_gate(a[k], b[k], ck);
        ob.cx(c_prev, ck);
    }
    ob.cx(b[n - 1], a[n - 1]);
    ob.cx(sc + n - 2, a[n - 1]);
    for (int k = n - 2; k >= 1; --k) {
        int c_prev = sc + k - 1, ck = sc + k;
        ob.cx(c_prev, ck);
        if (ob.mx_free(ck)) ob.cz(a[k], b[k]);
        ob.cx(c_prev, b[k]);
        ob.cx(b[k], a[k]);
    }
    if (ob.mx_free(c0)) ob.cz(a[0], b[0]);
    ob.cx(b[0], a[0]);
}

void emit_add_const(OutBuffer& ob, const uint16_t* a, int n, uint64_t c_val, int sc, int* carries) {
    if (n == 0) return;
    uint64_t mask = (n == 64) ? ~0ULL : ((1ULL << n) - 1ULL);
    uint64_t c = c_val & mask;
    if (c == 0ULL) return;
    int k0 = tzcnt64(c);
    if (k0 == n - 1) { ob.x(a[n - 1]); return; }
    if (k0 == n - 2) {
        ob.cx(a[n - 2], a[n - 1]);
        if ((c >> (n - 1)) & 1ULL) ob.x(a[n - 1]);
        ob.x(a[n - 2]);
        return;
    }
    int num_anc = n - 2 - k0;
    carries[0] = a[k0];
    for (int i = 0; i < num_anc; ++i) carries[i + 1] = sc + i;
    for (int idx = 1; idx <= num_anc; ++idx) {
        int k = k0 + idx;
        int prev_c = carries[idx - 1], next_c = carries[idx];
        if (((c >> k) & 1ULL) == 0ULL) {
            ob.alloc(next_c);
            ob.and_gate(a[k], prev_c, next_c);
        } else {
            ob.x(a[k]); ob.x(prev_c);
            ob.alloc(next_c);
            ob.and_gate(a[k], prev_c, next_c);
            ob.x(next_c); ob.x(prev_c); ob.x(a[k]);
        }
    }
    ob.cx(carries[num_anc], a[n - 1]);
    if ((c >> (n - 1)) & 1ULL) ob.x(a[n - 1]);
    for (int idx = num_anc; idx >= 1; --idx) {
        int k = k0 + idx;
        int prev_c = carries[idx - 1], next_c = carries[idx];
        bool bk = ((c >> k) & 1ULL) != 0ULL;
        if (!bk) {
            if (ob.mx_free(next_c)) ob.cz(a[k], prev_c);
        } else {
            ob.x(next_c); ob.x(a[k]); ob.x(prev_c);
            if (ob.mx_free(next_c)) ob.cz(a[k], prev_c);
            ob.x(prev_c); ob.x(a[k]);
        }
        ob.cx(prev_c, a[k]);
        if (bk) ob.x(a[k]);
    }
    ob.x(a[k0]);
}

void emit_cmp_ge_phase(OutBuffer& ob, const uint16_t* a, const uint16_t* b, int n, int sc) {
    if (n == 0) { ob.global_z(); return; }
    for (int i = 0; i < n; ++i) ob.x(b[i]);
    if (n == 1) {
        ob.global_z();
        ob.x(a[0]); ob.x(b[0]);
        ob.cz(a[0], b[0]);
        ob.x(b[0]); ob.x(a[0]);
        ob.x(b[0]);
        return;
    }
    int c0 = sc, top = sc + n - 1;
    ob.x(a[0]); ob.x(b[0]);
    ob.alloc(c0);
    ob.and_gate(a[0], b[0], c0);
    ob.x(c0);
    ob.x(b[0]); ob.x(a[0]);
    for (int k = 1; k < n - 1; ++k) {
        int c_prev = sc + k - 1, ck = sc + k;
        ob.cx(c_prev, a[k]);
        ob.cx(c_prev, b[k]);
        ob.alloc(ck);
        ob.and_gate(a[k], b[k], ck);
        ob.cx(c_prev, ck);
    }
    int c_last = sc + n - 2;
    ob.z(c_last);
    ob.cx(c_last, a[n - 1]);
    ob.cx(c_last, b[n - 1]);
    ob.alloc(top);
    ob.and_gate(a[n - 1], b[n - 1], top);
    ob.z(top);
    if (ob.mx_free(top)) ob.cz(a[n - 1], b[n - 1]);
    ob.cx(c_last, b[n - 1]);
    ob.cx(c_last, a[n - 1]);
    for (int k = n - 2; k >= 1; --k) {
        int c_prev = sc + k - 1, ck = sc + k;
        ob.cx(c_prev, ck);
        if (ob.mx_free(ck)) ob.cz(a[k], b[k]);
        ob.cx(c_prev, b[k]);
        ob.cx(c_prev, a[k]);
    }
    ob.x(c0);
    ob.x(a[0]); ob.x(b[0]);
    if (ob.mx_free(c0)) ob.cz(a[0], b[0]);
    ob.x(b[0]); ob.x(a[0]);
    for (int i = 0; i < n; ++i) ob.x(b[i]);
}

void qrom_dfs(
    OutBuffer& ob, const uint16_t* addr, int a, const uint16_t* out_w, int w, int n_limbs,
    const uint64_t* mob, int sc, int parent_wire, int parent_mask, int min_b, int depth) {
    int u = sc + depth;
    for (int b = min_b; b < a; ++b) {
        int cur_mask = parent_mask | (1 << b);
        int xb = addr[b];
        ob.alloc(u);
        ob.and_gate(parent_wire, xb, u);
        int base_idx = cur_mask * n_limbs;
        for (int li = 0; li < n_limbs; ++li) {
            uint64_t vm = mob[base_idx + li];
            int bit_base = li << 6;
            while (vm != 0ULL) {
                int bit = tzcnt64(vm);
                if (bit_base + bit < w) ob.cx(u, out_w[bit_base + bit]);
                vm &= (vm - 1ULL);
            }
        }
        if (b + 1 < a)
            qrom_dfs(ob, addr, a, out_w, w, n_limbs, mob, sc, u, cur_mask, b + 1, depth + 1);
        if (ob.mx_free(u)) ob.cz(parent_wire, xb);
    }
}

void emit_qrom_init(
    OutBuffer& ob, const uint16_t* addr, int a, const uint16_t* out_w, int w,
    int n_limbs, uint64_t* mob, int sc) {
    int n_entries = 1 << a;
    for (int j = 0; j < w; ++j) {
        ob.free_zero(out_w[j]);
        ob.alloc(out_w[j]);
    }
    for (int bit = 0; bit < a; ++bit) {
        int step = 1 << bit, step2 = step << 1;
        for (int idx = 0; idx < n_entries; idx += step2) {
            for (int j = idx; j < idx + step; ++j) {
                int s_off = j * n_limbs, d_off = (j + step) * n_limbs;
                for (int li = 0; li < n_limbs; ++li) mob[d_off + li] ^= mob[s_off + li];
            }
        }
    }
    for (int li = 0; li < n_limbs; ++li) {
        uint64_t v0 = mob[li];
        int bit_base = li << 6;
        while (v0 != 0ULL) {
            int bit = tzcnt64(v0);
            if (bit_base + bit < w) ob.x(out_w[bit_base + bit]);
            v0 &= (v0 - 1ULL);
        }
    }
    for (int i = 0; i < a; ++i) {
        int ai = addr[i], off = (1 << i) * n_limbs;
        for (int li = 0; li < n_limbs; ++li) {
            uint64_t vm = mob[off + li];
            int bit_base = li << 6;
            while (vm != 0ULL) {
                int bit = tzcnt64(vm);
                if (bit_base + bit < w) ob.cx(ai, out_w[bit_base + bit]);
                vm &= (vm - 1ULL);
            }
        }
    }
    if (a >= 2) {
        for (int i1 = 0; i1 < a - 1; ++i1)
            qrom_dfs(ob, addr, a, out_w, w, n_limbs, mob, sc, addr[i1], 1 << i1, i1 + 1, 0);
    }
}

void emit_sqrt_phaseup(OutBuffer& ob, const uint16_t* addr, int a, uint8_t* mob, int sc, int* p1, int* p2) {
    int n_entries = 1 << a;
    for (int bit = 0; bit < a; ++bit) {
        int step = 1 << bit, step2 = step << 1;
        for (int idx = 0; idx < n_entries; idx += step2)
            for (int j = idx; j < idx + step; ++j) mob[j + step] ^= mob[j];
    }
    if (a == 0) { if (mob[0]) ob.global_z(); return; }
    if (a == 1) {
        if (mob[0]) ob.global_z();
        if (mob[1]) ob.z(addr[0]);
        return;
    }
    int n1 = a >> 1, n2 = a - n1, m1 = 1 << n1, m2 = 1 << n2;
    int next_sc = sc;
    for (int col = 1; col < m1; ++col) {
        int msb = bit_length32(col) - 1, prev = col ^ (1 << msb);
        if (prev == 0) p1[col] = addr[msb];
        else { int w = next_sc++; p1[col] = w; ob.alloc(w); ob.and_gate(p1[prev], addr[msb], w); }
    }
    for (int col = 1; col < m2; ++col) {
        int msb = bit_length32(col) - 1, prev = col ^ (1 << msb);
        if (prev == 0) p2[col] = addr[n1 + msb];
        else { int w = next_sc++; p2[col] = w; ob.alloc(w); ob.and_gate(p2[prev], addr[n1 + msb], w); }
    }
    if (mob[0]) ob.global_z();
    for (int k2 = 1; k2 < m2; ++k2) if (mob[k2 << n1]) ob.z(p2[k2]);
    for (int k1 = 1; k1 < m1; ++k1) {
        if (mob[k1]) ob.z(p1[k1]);
        int w1 = p1[k1];
        for (int k2 = 1; k2 < m2; ++k2) if (mob[(k2 << n1) | k1]) ob.cz(w1, p2[k2]);
    }
    for (int col = m2 - 1; col >= 1; --col) {
        int msb = bit_length32(col) - 1, prev = col ^ (1 << msb);
        if (prev != 0 && ob.mx_free(p2[col])) ob.cz(p2[prev], addr[n1 + msb]);
    }
    for (int col = m1 - 1; col >= 1; --col) {
        int msb = bit_length32(col) - 1, prev = col ^ (1 << msb);
        if (prev != 0 && ob.mx_free(p1[col])) ob.cz(p1[prev], addr[msb]);
    }
}

} // namespace

int main() {
    try {
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
        _setmode(_fileno(stdout), _O_BINARY);
#endif
        uint8_t small[64];
        read_exact(stdin, small, 4);
        uint32_t hdr_len = read_u32_le(small);
        if (hdr_len < 28 || hdr_len > (1U << 24))
            throw std::runtime_error("Invalid EGKMV1 header length");
        std::vector<uint8_t> hdr_bytes(hdr_len);
        read_exact(stdin, hdr_bytes.data(), hdr_len);
        uint64_t mx_seed = read_u64_le(hdr_bytes.data() + 8);

        OutBuffer ob(stdout, mx_seed);
        ob.write_raw(hdr_bytes.data(), hdr_len);

        std::vector<uint8_t> io_buf(1 << 20);
        std::vector<uint16_t> w_a(4096), w_b(4096);
        std::vector<int> carries(128), p1(4096), p2(4096);
        std::vector<uint64_t> mob_limbs(65536);
        std::vector<uint8_t> mob_bool(65536);
        auto ensure_wires = [&](int need) {
            if (need < 0) throw std::runtime_error("Negative wire count");
            if (w_a.size() < size_t(need)) w_a.resize(size_t(need));
            if (w_b.size() < size_t(need)) w_b.resize(size_t(need));
        };

        while (true) {
            int op = std::fgetc(stdin);
            if (op == EOF) throw std::runtime_error("Macro stream ended without M_EOF");
            switch (op) {
                case 0:
                    ob.emit1(OP_EOF);
                    ob.flush();
                    std::fflush(stdout);
                    return 0;
                case 1: {
                    read_exact(stdin, small, 4);
                    uint32_t cnt = read_u32_le(small);
                    size_t bytes = size_t(cnt) * 4;
                    if (io_buf.size() < bytes) io_buf.resize(bytes);
                    read_exact(stdin, io_buf.data(), bytes);
                    uint32_t idx = 0;
                    while (idx < cnt) {
                        uint32_t w = read_u32_le(io_buf.data() + idx * 4);
                        ++idx;
                        uint32_t eop = w & 0xFU;
                        if (eop == OP_AND) {
                            if (idx >= cnt) throw std::runtime_error("Truncated OP_AND in M_RAW_WORDS");
                            uint32_t w2 = read_u32_le(io_buf.data() + idx * 4);
                            ++idx;
                            ob.emit2(w, w2);
                        } else {
                            if (eop == OP_MX_FREE) ob.next_bit();
                            ob.emit1(w);
                        }
                    }
                    break;
                }
                case 2: {
                    read_exact(stdin, small, 4);
                    int n = read_u16_le(small), sc = read_u16_le(small + 2);
                    ensure_wires(n);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    read_wires(stdin, io_buf, w_b.data(), n);
                    emit_add_qq(ob, w_a.data(), w_b.data(), n, sc);
                    break;
                }
                case 3: {
                    read_exact(stdin, small, 4);
                    int n = read_u16_le(small), sc = read_u16_le(small + 2);
                    ensure_wires(n);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    read_wires(stdin, io_buf, w_b.data(), n);
                    for (int i = 0; i < n; ++i) ob.x(w_a[i]);
                    emit_add_qq(ob, w_a.data(), w_b.data(), n, sc);
                    for (int i = 0; i < n; ++i) ob.x(w_a[i]);
                    break;
                }
                case 4: {
                    read_exact(stdin, small, 12);
                    int n = read_u16_le(small), sc = read_u16_le(small + 2);
                    if (n > 64) throw std::runtime_error("M_ADD_CONST width exceeds 64 bits");
                    ensure_wires(n);
                    if (carries.size() < size_t(n + 2)) carries.resize(size_t(n + 2));
                    uint64_t c = read_u64_le(small + 4);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    emit_add_const(ob, w_a.data(), n, c, sc, carries.data());
                    break;
                }
                case 5: {
                    read_exact(stdin, small, 4);
                    int n = read_u16_le(small), sc = read_u16_le(small + 2);
                    ensure_wires(n);
                    if (n > 0) {
                        read_wires(stdin, io_buf, w_a.data(), n);
                        read_wires(stdin, io_buf, w_b.data(), n);
                    }
                    emit_cmp_ge_phase(ob, w_a.data(), w_b.data(), n, sc);
                    break;
                }
                case 6: {
                    read_exact(stdin, small, 8);
                    int a = read_u16_le(small), w = read_u16_le(small + 2);
                    int n_limbs = read_u16_le(small + 4), sc = read_u16_le(small + 6);
                    if (a > 20) throw std::runtime_error("M_QROM_INIT address width exceeds 20 bits");
                    if (w > 4096 || n_limbs > 64) throw std::runtime_error("M_QROM_INIT output width out of range");
                    ensure_wires(a > w ? a : w);
                    if (a > 0) read_wires(stdin, io_buf, w_a.data(), a);
                    read_wires(stdin, io_buf, w_b.data(), w);
                    size_t total_u64 = (size_t(1) << a) * size_t(n_limbs);
                    if (mob_limbs.size() < total_u64) mob_limbs.resize(total_u64);
                    size_t total_bytes = total_u64 * 8;
                    if (io_buf.size() < total_bytes) io_buf.resize(total_bytes);
                    read_exact(stdin, io_buf.data(), total_bytes);
                    for (size_t k = 0; k < total_u64; ++k) mob_limbs[k] = read_u64_le(io_buf.data() + k * 8);
                    emit_qrom_init(ob, w_a.data(), a, w_b.data(), w, n_limbs, mob_limbs.data(), sc);
                    break;
                }
                case 7: {
                    read_exact(stdin, small, 4);
                    int a = read_u16_le(small), sc = read_u16_le(small + 2);
                    if (a > 24) throw std::runtime_error("M_SQRT_PHASEUP address width exceeds 24 bits");
                    ensure_wires(a);
                    size_t m1_sz = size_t(1) << (a >> 1);
                    size_t m2_sz = size_t(1) << (a - (a >> 1));
                    if (p1.size() < m1_sz) p1.resize(m1_sz);
                    if (p2.size() < m2_sz) p2.resize(m2_sz);
                    if (a > 0) read_wires(stdin, io_buf, w_a.data(), a);
                    size_t n_entries = size_t(1) << a;
                    if (mob_bool.size() < n_entries) mob_bool.resize(n_entries);
                    read_exact(stdin, mob_bool.data(), n_entries);
                    emit_sqrt_phaseup(ob, w_a.data(), a, mob_bool.data(), sc, p1.data(), p2.data());
                    break;
                }
                case 8: {
                    read_exact(stdin, small, 2);
                    int n = read_u16_le(small);
                    ensure_wires(n);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    read_wires(stdin, io_buf, w_b.data(), n);
                    for (int i = 0; i < n; ++i) ob.cx(w_a[i], w_b[i]);
                    break;
                }
                case 9: {
                    read_exact(stdin, small, 4);
                    int n = read_u16_le(small), n_limbs = read_u16_le(small + 2);
                    ensure_wires(n);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    size_t limb_bytes = size_t(n_limbs) * 8;
                    if (io_buf.size() < limb_bytes) io_buf.resize(limb_bytes);
                    read_exact(stdin, io_buf.data(), limb_bytes);
                    for (int li = 0; li < n_limbs; ++li) {
                        uint64_t v = read_u64_le(io_buf.data() + li * 8);
                        int base_bit = li << 6;
                        while (v != 0ULL) {
                            int bit = tzcnt64(v);
                            if (base_bit + bit < n) ob.x(w_a[base_bit + bit]);
                            v &= (v - 1ULL);
                        }
                    }
                    break;
                }
                case 10: {
                    read_exact(stdin, small, 2);
                    int n = read_u16_le(small);
                    ensure_wires(n);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    for (int i = 0; i < n; ++i) ob.z(w_a[i]);
                    break;
                }
                case 11: {
                    read_exact(stdin, small, 2);
                    int n = read_u16_le(small);
                    ensure_wires(n);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    read_wires(stdin, io_buf, w_b.data(), n);
                    for (int i = 0; i < n; ++i) ob.cz(w_a[i], w_b[i]);
                    break;
                }
                case 12: {
                    read_exact(stdin, small, 2);
                    int n = read_u16_le(small);
                    ensure_wires(n);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    for (int i = 0; i < n; ++i) ob.alloc(w_a[i]);
                    break;
                }
                case 13: {
                    read_exact(stdin, small, 2);
                    int n = read_u16_le(small);
                    ensure_wires(n);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    for (int i = 0; i < n; ++i) ob.free_zero(w_a[i]);
                    break;
                }
                case 14: {
                    read_exact(stdin, small, 2);
                    int n = read_u16_le(small);
                    ensure_wires(n);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    for (int i = 0; i < n; ++i) ob.mx_free(w_a[i]);
                    break;
                }
                case 15: {
                    read_exact(stdin, small, 2);
                    int n = read_u16_le(small);
                    ensure_wires(n);
                    read_wires(stdin, io_buf, w_a.data(), n);
                    for (int i = 0; i < n; ++i) { ob.mx_free(w_a[i]); ob.alloc(w_a[i]); }
                    break;
                }
                default:
                    throw std::runtime_error("Unknown macro opcode");
            }
        }
    } catch (const std::exception& ex) {
        std::cerr << "macro_stream_lowerer error: " << ex.what() << "\n";
        return 2;
    }
}
