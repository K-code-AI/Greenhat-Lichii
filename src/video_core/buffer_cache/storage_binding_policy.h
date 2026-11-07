// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"

namespace VideoCommon {

/// How a storage-buffer slot's size is decided. The caller knows what the qword at the
/// descriptor address is: a bare native pointer, an entry of a table held in memory, or an NVN
/// buffer descriptor whose next qword carries the size.
struct StorageBindingPolicy {
    bool native_pointer;
    bool size_from_mapping;
    bool require_contiguous;
    bool table_pointer;

    static constexpr StorageBindingPolicy Graphics(bool native, bool table) {
        return {native, native, native || table, table};
    }

    static constexpr StorageBindingPolicy Compute(bool native, bool dlss, bool table) {
        return {native, dlss || native, native || table, table};
    }

    /// The qword after the pointer is an NVN buffer size only when nothing else explains it.
    constexpr bool ReadsSizeWord(bool written) const {
        return !size_from_mapping && !(table_pointer && !written);
    }

    constexpr u32 Size(u32 mapped, u64 next_qword, bool written, u64 max_range) const {
        // A pointer read from a table in memory is the base of an object pool, read anywhere in
        // it: bound to the end of its mapping, as far as a storage buffer may reach (at most
        // 1 GiB). Ocarina of Time's grass instances lie 12-18 MiB past theirs (an 8 MiB cap
        // culled every blade), and its light probe records reach about 130 MiB past theirs: a
        // 64 MiB cap left the far probes outside the binding, and the fog took one overbright
        // constant from them. Written pools keep a cap: a write marks the whole binding
        // GPU-modified.
        if (table_pointer && !written) {
            const u64 reach = (std::min)(max_range, u64{1024 * 1024 * 1024});
            const u32 limit = static_cast<u32>(reach - (std::min)(reach, u64{64 * 1024}));
            return (std::min)(mapped, limit);
        }
        // A native (Switch 2) program's pointer is bare: the word after it is another parameter
        // (a dimension, stride or texture handle), not an NVN buffer size. Taking it as one bound
        // Fortune's Weave's DLSS input features to 266 bytes and dropped the rest.
        if (native_pointer) {
            return (std::min)(mapped, u32{8 * 1024 * 1024});
        }
        const u32 packed_size = static_cast<u32>(next_qword);
        if (!size_from_mapping && (next_qword >> 32) == 0 && packed_size != 0 &&
            packed_size <= mapped) {
            return packed_size;
        }
        return (std::min)(mapped, u32{8 * 1024 * 1024});
    }
};

} // namespace VideoCommon
