// SPDX-License-Identifier: GPL-3.0-or-later

#include <catch2/catch_test_macros.hpp>

#include "video_core/buffer_cache/storage_binding_policy.h"

using VideoCommon::StorageBindingPolicy;

TEST_CASE("Compute storage sizing preserves DLSS and native table contracts",
          "[video_core][storage_binding_policy]") {
    constexpr u32 MiB = 1024 * 1024;
    constexpr u64 max_range = 1024ULL * MiB; // what a desktop Vulkan device reports

    // A native compute program's pointer is bare: the word after it is another parameter, never
    // an NVN size, so it is never read and the binding is capped instead.
    const auto direct = StorageBindingPolicy::Compute(true, false, false);
    CHECK_FALSE(direct.ReadsSizeWord(true));
    CHECK_FALSE(direct.ReadsSizeWord(false));
    CHECK(direct.Size(32 * MiB, 256, true, max_range) == 8 * MiB);
    CHECK(direct.Size(32 * MiB, 0, false, max_range) == 8 * MiB);
    CHECK(direct.require_contiguous);

    const auto dlss = StorageBindingPolicy::Compute(true, true, false);
    CHECK(dlss.Size(32 * MiB, 0x10a, false, max_range) == 8 * MiB);
    CHECK_FALSE(dlss.ReadsSizeWord(false));
    CHECK(dlss.require_contiguous);

    // An unwritten table entry is the base of a pool read anywhere in it: it reaches to the end
    // of its mapping, as far as a storage buffer may reach (at most 1 GiB, less a margin).
    const auto indirect = StorageBindingPolicy::Compute(true, false, true);
    CHECK(indirect.Size(32 * MiB, 256, false, max_range) == 32 * MiB);
    CHECK(indirect.Size(80 * MiB, 256, false, max_range) == 80 * MiB);
    CHECK(indirect.Size(2048 * MiB, 256, false, max_range) == 1024 * MiB - 64 * 1024);
    CHECK(indirect.Size(80 * MiB, 256, false, 16 * MiB) == 16 * MiB - 64 * 1024);
    // A written pool keeps a cap: the whole binding would be marked GPU-modified.
    CHECK(indirect.Size(32 * MiB, 256, true, max_range) == 8 * MiB);
    CHECK_FALSE(indirect.ReadsSizeWord(true));
    CHECK_FALSE(indirect.ReadsSizeWord(false));
    CHECK(indirect.require_contiguous);

    // A legacy (non-native) descriptor's next qword is the NVN size.
    const auto legacy = StorageBindingPolicy::Compute(false, false, false);
    CHECK(legacy.ReadsSizeWord(true));
    CHECK(legacy.Size(32 * MiB, 256, true, max_range) == 256);
    for (u64 invalid : {u64{0}, u64{33 * MiB}, u64{1} << 32}) {
        CHECK(legacy.Size(32 * MiB, invalid, true, max_range) == 8 * MiB);
    }
    CHECK_FALSE(legacy.require_contiguous);

    for (bool table : {false, true}) {
        const auto graphics = StorageBindingPolicy::Graphics(true, table);
        CHECK(graphics.require_contiguous);
        CHECK(graphics.Size(32 * MiB, 256, true, max_range) == 8 * MiB);
        CHECK(graphics.Size(32 * MiB, 256, false, max_range) == (table ? 32 : 8) * MiB);
    }
    // A written graphics table entry is an NVN descriptor again: its size word is read.
    const auto written_table = StorageBindingPolicy::Graphics(false, true);
    CHECK(written_table.ReadsSizeWord(true));
    CHECK_FALSE(written_table.ReadsSizeWord(false));
    CHECK(written_table.Size(32 * MiB, 256, true, max_range) == 256);
    CHECK(written_table.Size(32 * MiB, 256, false, max_range) == 32 * MiB);
}
