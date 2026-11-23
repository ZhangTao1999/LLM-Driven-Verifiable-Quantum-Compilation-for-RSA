// Independent macro-free elementary-gate kickmix verifier (portable ISO C++17 binary).
// Compiles on Linux / macOS / Windows with zero external dependencies:
//   g++ -O3 -std=c++17 elementary_kickmix_verifier.cpp -o elementary_kickmix_verifier
//
// Knows ZERO domain concepts (no RNS, primes, QROM, adders, comparators, or LKS).
// Enforces wire liveness, disjointness, clean |0> AND targets, clean |0> frees,
// readonly input preservation, and 64-branch phase cancellation on flat 1-bit opcodes.
//
// Binary stream protocol (EGKMV1, little-endian):
//   Header (28 bytes):
//     [0..7]   magic            = "EGKMV1\0\0" (8 bytes)
//     [8..15]  rng_state        = uint64 SplitMix64 seed for MX measurement outcomes
//     [16..19] num_branches     = uint32 in [1, 64]
//     [20..23] max_wires        = uint32 in [1, 16384]
//     [24..27] num_init_wires   = uint32 <= max_wires
//   Init wire records (num_init_wires x 12 bytes):
//     uint16 wire_id, uint16 flags (bit 0 = readonly; bits 1..15 must be 0), uint64 branch_bitplane
//   Output groups:
//     uint32 num_output_groups; then for each group: uint32 glen (1..64) + glen x uint16 wire_id
//   Opcode stream (32-bit little-endian words; op = word & 0xF, 14-bit wire fields):
//     0 = OP_EOF       : word must be 0x00000000; no trailing bytes permitted
//     1 = OP_ALLOC     : w = (word >> 4) & 0x3FFF (bits 18..31 = 0)
//     2 = OP_FREE_ZERO : w = (word >> 4) & 0x3FFF (bits 18..31 = 0; asserts val[w] == 0)
//     3 = OP_X         : w = (word >> 4) & 0x3FFF (bits 18..31 = 0)
//     4 = OP_CX        : c = (word >> 4) & 0x3FFF, t = (word >> 18) & 0x3FFF (c != t)
//     5 = OP_Z         : w = (word >> 4) & 0x3FFF (bits 18..31 = 0)
//     6 = OP_CZ        : a = (word >> 4) & 0x3FFF, b = (word >> 18) & 0x3FFF (a != b)
//     7 = OP_GLOBAL_Z  : bits 4..31 = 0
//     8 = OP_AND       : 2 words: word1 has c1=(>>4), c2=(>>18); word2 has t=(&0x3FFF, bits 14..31=0)
//     9 = OP_MX_FREE   : w = (word >> 4) & 0x3FFF, m = (word >> 18) & 1 (must match SplitMix64)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

namespace {

struct GateViolation : public std::runtime_error {
    explicit GateViolation(const std::string& msg) : std::runtime_error(msg) {}
};

inline uint64_t next_u64(uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

void read_exact(std::FILE* fp, void* dst, size_t count) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    while (count > 0) {
        size_t n = std::fread(p, 1, count, fp);
        if (n == 0) {
            throw GateViolation("TRUNCATED_STREAM_WITHOUT_EOF");
        }
        p += n;
        count -= n;
    }
}

uint16_t read_u16_le(const uint8_t* p) {
    return uint16_t(p[0]) | (uint16_t(p[1]) << 8);
}

uint32_t read_u32_le(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

uint64_t read_u64_le(const uint8_t* p) {
    return uint64_t(read_u32_le(p)) | (uint64_t(read_u32_le(p + 4)) << 32);
}

std::string hex16(uint64_t x) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setw(16) << std::setfill('0') << x;
    return oss.str();
}

std::string verify_stream(std::FILE* fp) {
    uint8_t hdr[28];
    read_exact(fp, hdr, 28);
    if (std::memcmp(hdr, "EGKMV1\0\0", 8) != 0) {
        throw GateViolation("Invalid magic header");
    }
    uint64_t rng_state = read_u64_le(hdr + 8);
    uint32_t num_branches = read_u32_le(hdr + 16);
    uint32_t max_wires = read_u32_le(hdr + 20);
    uint32_t num_init_wires = read_u32_le(hdr + 24);

    if (num_branches < 1 || num_branches > 64) {
        throw GateViolation("Invalid num_branches");
    }
    if (max_wires < 1 || max_wires > 16384) {
        throw GateViolation("Invalid max_wires");
    }
    if (num_init_wires > max_wires) {
        throw GateViolation("num_init_wires exceeds max_wires");
    }
    const uint64_t branch_mask = (num_branches == 64) ? ~0ULL : ((1ULL << num_branches) - 1ULL);

    std::vector<uint8_t> live(max_wires, 0);
    std::vector<uint64_t> val(max_wires, 0ULL);
    std::vector<uint8_t> is_init(max_wires, 0);
    std::vector<std::pair<uint16_t, uint64_t>> readonly_init;
    readonly_init.reserve(num_init_wires);

    int live_wires = 0;
    int peak_live_wires = 0;

    uint64_t stream_bytes = 28ULL;
    uint8_t rec[12];
    for (uint32_t i = 0; i < num_init_wires; ++i) {
        read_exact(fp, rec, 12);
        stream_bytes += 12ULL;
        uint16_t w = read_u16_le(rec);
        uint16_t flags = read_u16_le(rec + 2);
        uint64_t raw_bp = read_u64_le(rec + 4);
        if ((flags & ~uint16_t(1U)) != 0U) {
            throw GateViolation("NONZERO_HEADER_FLAGS");
        }
        if ((raw_bp & ~branch_mask) != 0ULL) {
            throw GateViolation("NONZERO_UNUSED_BRANCH_BITS");
        }
        uint64_t init_bp = raw_bp & branch_mask;
        if (w >= max_wires || live[w]) {
            throw GateViolation("Invalid or duplicate init wire");
        }
        live[w] = 1;
        is_init[w] = 1;
        val[w] = init_bp;
        ++live_wires;
        if (live_wires > peak_live_wires) peak_live_wires = live_wires;
        if (flags & 1U) {
            readonly_init.emplace_back(w, init_bp);
        }
    }

    read_exact(fp, rec, 4);
    stream_bytes += 4ULL;
    uint32_t num_output_groups = read_u32_le(rec);
    if (num_output_groups > max_wires) {
        throw GateViolation("num_output_groups exceeds max_wires");
    }
    std::vector<std::vector<uint16_t>> output_groups(num_output_groups);
    for (uint32_t g = 0; g < num_output_groups; ++g) {
        read_exact(fp, rec, 4);
        stream_bytes += 4ULL;
        uint32_t glen = read_u32_le(rec);
        if (glen < 1 || glen > 64) {
            throw GateViolation("Invalid output group length");
        }
        std::vector<uint8_t> gbytes(glen * 2);
        read_exact(fp, gbytes.data(), glen * 2);
        stream_bytes += uint64_t(glen * 2);
        output_groups[g].resize(glen);
        for (uint32_t j = 0; j < glen; ++j) {
            uint16_t wid = read_u16_le(gbytes.data() + j * 2);
            if (wid >= max_wires || !is_init[wid]) {
                throw GateViolation("Output group wire not in init wires");
            }
            output_groups[g][j] = wid;
        }
    }

    uint64_t phase_sign = 0ULL;
    uint64_t n_alloc = 0, n_free_zero = 0, n_x = 0, n_cx = 0, n_z = 0, n_cz = 0, n_global_z = 0, n_and = 0, n_mx = 0;

    constexpr size_t BUF_BYTES = 1 << 20;
    std::vector<uint8_t> buf(BUF_BYTES + 4);
    size_t carry = 0;
    bool eof_seen = false;

    while (!eof_seen) {
        size_t n_read = std::fread(buf.data() + carry, 1, BUF_BYTES - carry, fp);
        if (n_read == 0) {
            throw GateViolation("TRUNCATED_STREAM_WITHOUT_EOF");
        }
        stream_bytes += uint64_t(n_read);
        size_t total = carry + n_read;
        size_t limit = total & ~size_t(3);
        size_t pos = 0;

        while (pos < limit) {
            uint32_t word = read_u32_le(buf.data() + pos);
            pos += 4;
            uint32_t op = word & 0xFU;
            switch (op) {
                case 4: { // OP_CX
                    uint32_t c = (word >> 4) & 0x3FFFU;
                    uint32_t t = (word >> 18) & 0x3FFFU;
                    if (c >= max_wires || t >= max_wires || !live[c] || !live[t] || c == t)
                        throw GateViolation("INVALID_CX");
                    val[t] ^= val[c];
                    ++n_cx;
                    break;
                }
                case 6: { // OP_CZ
                    uint32_t a = (word >> 4) & 0x3FFFU;
                    uint32_t b = (word >> 18) & 0x3FFFU;
                    if (a >= max_wires || b >= max_wires || !live[a] || !live[b] || a == b)
                        throw GateViolation("INVALID_CZ");
                    phase_sign ^= (val[a] & val[b]);
                    ++n_cz;
                    break;
                }
                case 3: { // OP_X
                    if ((word >> 18) != 0U) throw GateViolation("NONZERO_RESERVED_BITS");
                    uint32_t w = (word >> 4) & 0x3FFFU;
                    if (w >= max_wires || !live[w]) throw GateViolation("INVALID_X");
                    val[w] ^= branch_mask;
                    ++n_x;
                    break;
                }
                case 5: { // OP_Z
                    if ((word >> 18) != 0U) throw GateViolation("NONZERO_RESERVED_BITS");
                    uint32_t w = (word >> 4) & 0x3FFFU;
                    if (w >= max_wires || !live[w]) throw GateViolation("INVALID_Z");
                    phase_sign ^= val[w];
                    ++n_z;
                    break;
                }
                case 8: { // OP_AND
                    uint32_t word2;
                    if (pos + 4 <= limit) {
                        word2 = read_u32_le(buf.data() + pos);
                        pos += 4;
                    } else {
                        size_t rem = total - pos;
                        std::memcpy(rec, buf.data() + pos, rem);
                        read_exact(fp, rec + rem, 4 - rem);
                        stream_bytes += uint64_t(4 - rem);
                        pos = total;
                        limit = total;
                        word2 = read_u32_le(rec);
                    }
                    if ((word2 >> 14) != 0U) throw GateViolation("NONZERO_RESERVED_BITS");
                    uint32_t c1 = (word >> 4) & 0x3FFFU;
                    uint32_t c2 = (word >> 18) & 0x3FFFU;
                    uint32_t t = word2 & 0x3FFFU;
                    if (c1 >= max_wires || c2 >= max_wires || t >= max_wires)
                        throw GateViolation("INVALID_AND_RANGE");
                    if (!live[c1] || !live[c2] || !live[t])
                        throw GateViolation("DEAD_WIRE_IN_AND");
                    if (c1 == c2 || t == c1 || t == c2)
                        throw GateViolation("WIRE_ALIAS_IN_AND");
                    if ((val[t] & branch_mask) != 0ULL)
                        throw GateViolation("DIRTY_AND_TARGET");
                    val[t] = val[c1] & val[c2];
                    ++n_and;
                    break;
                }
                case 9: { // OP_MX_FREE
                    if ((word >> 19) != 0U) throw GateViolation("NONZERO_RESERVED_BITS");
                    uint32_t w = (word >> 4) & 0x3FFFU;
                    uint32_t expected_m = (word >> 18) & 1U;
                    if (w >= max_wires || !live[w]) throw GateViolation("INVALID_MX");
                    uint32_t m = uint32_t(next_u64(rng_state) & 1ULL);
                    if (m != expected_m) throw GateViolation("MX_SYNC_MISMATCH");
                    if (m) phase_sign ^= val[w];
                    val[w] = 0ULL;
                    live[w] = 0;
                    --live_wires;
                    ++n_mx;
                    break;
                }
                case 1: { // OP_ALLOC
                    if ((word >> 18) != 0U) throw GateViolation("NONZERO_RESERVED_BITS");
                    uint32_t w = (word >> 4) & 0x3FFFU;
                    if (w >= max_wires || live[w]) throw GateViolation("INVALID_ALLOC");
                    live[w] = 1;
                    val[w] = 0ULL;
                    ++live_wires;
                    if (live_wires > peak_live_wires) peak_live_wires = live_wires;
                    ++n_alloc;
                    break;
                }
                case 2: { // OP_FREE_ZERO
                    if ((word >> 18) != 0U) throw GateViolation("NONZERO_RESERVED_BITS");
                    uint32_t w = (word >> 4) & 0x3FFFU;
                    if (w >= max_wires || !live[w]) throw GateViolation("INVALID_FREE");
                    if ((val[w] & branch_mask) != 0ULL)
                        throw GateViolation("NONZERO_FREE");
                    live[w] = 0;
                    val[w] = 0ULL;
                    --live_wires;
                    ++n_free_zero;
                    break;
                }
                case 7: { // OP_GLOBAL_Z
                    if ((word >> 4) != 0U) throw GateViolation("NONZERO_RESERVED_BITS");
                    phase_sign ^= branch_mask;
                    ++n_global_z;
                    break;
                }
                case 0: { // OP_EOF
                    if (word != 0U) throw GateViolation("NONZERO_EOF_WORD");
                    if (pos != total) throw GateViolation("TRAILING_BYTES_AFTER_EOF");
                    if (std::fread(rec, 1, 1, fp) > 0) throw GateViolation("TRAILING_BYTES_AFTER_EOF");
                    eof_seen = true;
                    break;
                }
                default:
                    throw GateViolation("UNKNOWN_OPCODE");
            }
        }
        if (!eof_seen) {
            carry = total - limit;
            if (carry > 0) std::memmove(buf.data(), buf.data() + limit, carry);
        }
    }

    if (live_wires != int(num_init_wires))
        throw GateViolation("LEAKED_OR_MISSING_WIRES");
    for (uint32_t w = 0; w < max_wires; ++w) {
        if (live[w] != is_init[w])
            throw GateViolation("FINAL_LIVENESS_MISMATCH");
    }
    for (const auto& kv : readonly_init) {
        if ((val[kv.first] & branch_mask) != kv.second)
            throw GateViolation("READONLY_INPUT_MUTATED");
    }
    if ((phase_sign & branch_mask) != 0ULL)
        throw GateViolation("NONZERO_RESIDUAL_PHASE");

    uint64_t total_ops = n_alloc + n_free_zero + n_x + n_cx + n_z + n_cz + n_global_z + n_and + n_mx;
    std::ostringstream out;
    out << "{\"status\": \"PASS\", \"verifier\": \"cpp17_egkmv1\", "
        << "\"num_branches\": " << num_branches << ", "
        << "\"peak_live_wires\": " << peak_live_wires << ", "
        << "\"final_live_wires\": " << live_wires << ", "
        << "\"stream_bytes\": " << stream_bytes << ", "
        << "\"residual_phase_mask_u64\": \"" << hex16(phase_sign & branch_mask) << "\", "
        << "\"final_rng_state_u64\": \"" << hex16(rng_state) << "\", "
        << "\"gate_counts\": {"
        << "\"alloc\": " << n_alloc << ", "
        << "\"free_zero\": " << n_free_zero << ", "
        << "\"X\": " << n_x << ", "
        << "\"CX\": " << n_cx << ", "
        << "\"Z\": " << n_z << ", "
        << "\"CZ\": " << n_cz << ", "
        << "\"global_Z\": " << n_global_z << ", "
        << "\"AND\": " << n_and << ", "
        << "\"MX\": " << n_mx << ", "
        << "\"total_stream_ops\": " << total_ops
        << "}, \"output_groups\": [";
    for (size_t g = 0; g < output_groups.size(); ++g) {
        if (g > 0) out << ", ";
        out << "[";
        const auto& gw = output_groups[g];
        for (uint32_t b = 0; b < num_branches; ++b) {
            if (b > 0) out << ", ";
            uint64_t oval = 0ULL;
            for (size_t bit = 0; bit < gw.size(); ++bit) {
                if ((val[gw[bit]] >> b) & 1ULL) oval |= (1ULL << bit);
            }
            out << oval;
        }
        out << "]";
    }
    out << "]}";
    return out.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::FILE* fp = stdin;
#ifdef _WIN32
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        if (argc > 1 && std::strcmp(argv[1], "-") != 0) {
            fp = std::fopen(argv[1], "rb");
            if (!fp) throw GateViolation("Cannot open input file");
        }
        std::string json = verify_stream(fp);
        if (fp != stdin) std::fclose(fp);
        std::cout << json << std::endl;
        return 0;
    } catch (const GateViolation& ex) {
        std::cout << "{\"status\": \"FAIL\", \"error\": \"" << ex.what() << "\"}" << std::endl;
        return 2;
    } catch (const std::exception& ex) {
        std::cout << "{\"status\": \"ERROR\", \"error\": \"" << ex.what() << "\"}" << std::endl;
        return 3;
    }
}
