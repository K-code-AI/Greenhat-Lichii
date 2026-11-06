// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <algorithm>
#include "common/common_types.h"

// Ocarina of Time shows its pre-rendered movies through a compute chain: one pass turns the
// decoded NV12 frame into a 1920x1080 RGBA8 image, a tensor network (HMMA) refines it, and a last
// pass upscales the result to the 2560x1440 RGBA8 image the scene composites. The network only
// sharpens the frame, and emulating its matrix multiplies costs a phone GPU ~50 ms a frame, so the
// RGB frame is scaled into the output image instead.
namespace VideoCommon::OcarinaMovie {

constexpr u64 ProgramId = 0x0400486024DEC000ULL;

// Native program keys.
constexpr u64 ConvertKey = 0xf9e5b02b5fabff78ULL;
constexpr std::array NetworkKeys{
    0x2f5791cea2a85397ULL, 0x313d724f3e214203ULL, 0xdf2c83cf2ad70790ULL,
    0x1fdb95ea2565f49cULL, 0x0be2f024aea21a6cULL, 0x0618cc4a3ac8f33aULL,
    0xe27336d38f1cc8f4ULL, 0xecebb02dec6faa17ULL, 0x2a366b38546a4d5eULL,
};
constexpr u64 OutputKey = 0x6093c9143ba0904dULL;

// Constant bank 0 offsets of the image handles the passes write.
constexpr u32 ConvertImageOffset = 0x68;
constexpr u32 OutputImageOffset = 0xc8;

constexpr bool IsNetwork(u64 native_key) {
    return std::ranges::find(NetworkKeys, native_key) != NetworkKeys.end();
}

} // namespace VideoCommon::OcarinaMovie
