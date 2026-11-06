// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <algorithm>

#include "common/common_types.h"

namespace VideoCommon {

struct StorageBindingPolicy {
    bool size_from_mapping;
    bool require_contiguous;
    bool table_pointer;

    static constexpr StorageBindingPolicy Graphics(bool native, bool table) {
        return {native || table, native || table, table};
    }

    static constexpr StorageBindingPolicy Compute(bool native, bool dlss, bool table,
                                                  bool force_mapping_sized = false) {
        return {dlss || (native && (table || force_mapping_sized)), native || table, table};
    }

    constexpr bool ReadsSizeWord(bool written) const {
        return !size_from_mapping && !(table_pointer && !written);
    }

    constexpr u32 Size(u32 mapped, u64 next_qword, bool written) const {
        if (table_pointer && !written) {
            // A pointer read from a table in memory is the base of an object pool, read anywhere in
            // it (Ocarina of Time's grass instances lie 12-18 MiB past theirs): bound to the end of
            // its mapping, up to 64 MiB. The 8 MiB cap made those reads return 0 and culled every
            // blade. Written pools keep the cap: a write marks the whole binding GPU-modified.
            return (std::min)(mapped, u32{64 * 1024 * 1024});
        }
        const u32 packed_size = static_cast<u32>(next_qword);
        if (!size_from_mapping && (next_qword >> 32) == 0 && packed_size != 0 &&
            packed_size <= mapped) {
            return packed_size;
        }
        // A bare native pointer has another parameter (a dimension, stride or texture handle) in the
        // word after it, not an NVN buffer size: taking it as one bound Fortune's Weave's DLSS
        // input features to 266 bytes and dropped the rest.
        return (std::min)(mapped, u32{8 * 1024 * 1024});
    }
};

} // namespace VideoCommon
