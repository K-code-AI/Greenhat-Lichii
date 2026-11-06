// SPDX-License-Identifier: GPL-3.0-or-later

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <span>
#include <string>
#include <vector>
#include <catch2/catch_test_macros.hpp>

#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/exception.h"
#include "shader_recompiler/frontend/ampere/lower_to_maxwell.h"
#include "shader_recompiler/frontend/ir/ir_emitter.h"
#include "shader_recompiler/frontend/maxwell/decode.h"

namespace {
using namespace Shader;

struct TensorMode {
    std::string previous;
    bool existed{};

    explicit TensorMode(const char* mode) {
        if (const char* value = std::getenv("EDEN_NX2_TENSOR")) {
            previous = value;
            existed = true;
        }
        Set(mode);
    }

    static void Set(const char* value) {
#ifdef _WIN32
        _putenv_s("EDEN_NX2_TENSOR", value ? value : "");
#else
        if (value) {
            setenv("EDEN_NX2_TENSOR", value, 1);
        } else {
            unsetenv("EDEN_NX2_TENSOR");
        }
#endif
    }

    ~TensorMode() {
        Set(existed ? previous.c_str() : nullptr);
    }
};

struct TensorProgram {
    ObjectPool<IR::Inst> pool;
    IR::Block block{pool};
    IR::Program program;
    Profile profile;

    explicit TensorProgram(u32 count = 1, u32 software_depth = 0) {
        program.stage = Stage::Compute;
        program.workgroup_size = {64, 1, 1};
        program.blocks = {&block};
        program.post_order_blocks = {&block};
        program.syntax_list = {{{.block = &block}, IR::AbstractSyntaxNode::Type::Block},
                               {{}, IR::AbstractSyntaxNode::Type::Return}};
        program.info.uses_subgroup_shuffles = count != 0 && software_depth != 0;
        program.info.uses_subgroup_invocation_id = count != 0;
        profile.supported_spirv = 0x00010500;
        profile.supported_subgroup_stages = 1U << static_cast<u32>(Stage::Compute);
        profile.max_subgroup_size = 32;
        IR::IREmitter ir{block};
        if (count == 0) {
            return;
        }
        const auto lane = ir.LaneId();
        std::array<IR::U32, 8> inputs;
        for (u32 i = 0; i < inputs.size(); ++i) {
            inputs[i] = software_depth ? ir.IAdd(ir.Imm32(0x3c003c00U + i * 0x01000100U),
                ir.ShiftLeftLogical(lane, ir.Imm32(3))) : ir.IAdd(ir.Imm32((i + 1) * 0x01000100U), lane);
        }
        for (u32 i = 0; i < count; ++i) {
            const auto result = ir.TensorMulAdd(
                                       ir.CompositeConstruct(inputs[0], inputs[1], inputs[2], inputs[3]),
                                       ir.CompositeConstruct(inputs[4], inputs[5]),
                                       ir.CompositeConstruct(inputs[6], inputs[7]), software_depth);
            inputs[6] = IR::U32{ir.CompositeExtract(result, 0)};
            inputs[7] = IR::U32{ir.CompositeExtract(result, 1)};
        }
        if (software_depth) {
            program.info.used_storage_buffer_types = IR::Type::U32;
            program.info.storage_buffers_descriptors = {{0, 0, 1, true}};
            program.info.uses_local_invocation_id = true;
            const auto offset = ir.ShiftLeftLogical(ir.LocalInvocationIdX(), ir.Imm32(3));
            block.AppendNewInst(IR::Opcode::WriteStorage32, {ir.Imm32(0), offset, inputs[6]});
            block.AppendNewInst(IR::Opcode::WriteStorage32,
                {ir.Imm32(0), ir.IAdd(offset, ir.Imm32(4)), inputs[7]});
        }
    }

    std::vector<u32> Emit(std::string_view name = {}) {
        auto code = Backend::SPIRV::EmitSPIRV(profile, program);
        if (const char* directory = std::getenv("EDEN_TENSOR_TEST_OUTPUT");
            directory && !name.empty()) {
            std::filesystem::create_directories(directory);
            std::ofstream file{std::filesystem::path{directory} / (std::string{name} + ".spv"),
                               std::ios::binary};
            file.write(reinterpret_cast<const char*>(code.data()),
                       static_cast<std::streamsize>(code.size() * sizeof(u32)));
            REQUIRE(file.good());
        }
        return code;
    }
};

std::vector<std::vector<u32>> Instructions(const std::vector<u32>& code, u32 opcode) {
    std::vector<std::vector<u32>> result;
    for (size_t offset = 5; offset < code.size();) {
        const u32 size = code[offset] >> 16;
        REQUIRE(size != 0);
        REQUIRE(offset + size <= code.size());
        if ((code[offset] & 0xffffU) == opcode) {
            result.emplace_back(code.begin() + offset + 1, code.begin() + offset + size);
        }
        offset += size;
    }
    return result;
}

bool HasCapability(const std::vector<u32>& code, u32 capability) {
    const auto capabilities = Instructions(code, 17);
    return std::ranges::any_of(capabilities, [=](const auto& words) {
        return words[0] == capability;
    });
}

}

TEST_CASE("Software HMMA shares one helper without tensor extensions", "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram single{1, depth};
        const auto first = single.Emit(depth == 8 ? "software-k8" : "software-k16");
        REQUIRE_FALSE(HasCapability(first, 5357));
        REQUIRE(Instructions(first, 246).size() == 1); // OpLoopMerge: one loop over K8 groups
        REQUIRE(Instructions(first, 54).size() == 2); // OpFunction: main and the helper
        REQUIRE(Instructions(first, 57).size() == 1); // OpFunctionCall
        TensorProgram repeated{64, depth};
        const auto many = repeated.Emit(depth == 8 ? "software-k8-many" : "software-k16-many");
        REQUIRE(Instructions(many, 246).size() == 1);
        REQUIRE(Instructions(many, 57).size() == 64);
        // Three fragment-vector constructions, the call and two result extracts
        // take at most 40 words per site; the matrix helper itself is never duplicated.
        REQUIRE(many.size() < first.size() + 64 * 40);
    }
}

namespace {
struct HelperCounts {
    u32 fma{};
    u32 shuffles{};
    u32 quad_broadcasts{};
    u32 conversions{};
    std::vector<u32> fma_before_rounding; ///< FMAs emitted before each RoundEven
    std::vector<u32> quad_indices;
    std::vector<u32> conversion_ids;
};

HelperCounts CountHelper(const std::vector<u32>& code) {
    std::map<u32, u32> constants;
    for (const auto& words : Instructions(code, 43)) { // OpConstant
        constants.emplace(words[1], words[2]);
    }
    HelperCounts counts;
    for (size_t offset = 5; offset < code.size();) {
        const u32 length = code[offset] >> 16;
        const u32 opcode = code[offset] & 0xffffU;
        REQUIRE(length != 0);
        REQUIRE(offset + length <= code.size());
        const auto words = std::span{code}.subspan(offset + 1, length - 1);
        if (opcode == 12 && words[3] == 2) { // GLSL.std.450 RoundEven
            counts.fma_before_rounding.push_back(counts.fma);
        } else if (opcode == 12 && words[3] == 50) { // GLSL.std.450 Fma
            ++counts.fma;
        } else if (opcode == 345) { // OpGroupNonUniformShuffle
            ++counts.shuffles;
        } else if (opcode == 365) { // OpGroupNonUniformQuadBroadcast
            ++counts.quad_broadcasts;
            REQUIRE(constants.contains(words[4]));
            counts.quad_indices.push_back(constants.at(words[4]));
        } else if (opcode == 115) { // OpFConvert
            ++counts.conversions;
            counts.conversion_ids.push_back(words[1]);
        }
        offset += length;
    }
    return counts;
}
} // namespace

TEST_CASE("Software HMMA rounds once per K8 group", "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        for (const bool quad : {false, true}) {
            TensorProgram fixture{1, depth};
            fixture.profile.support_quad_shuffles = quad;
            const auto code = fixture.Emit();
            std::map<u32, u32> constants;
            for (const auto& words : Instructions(code, 43)) { // OpConstant
                constants.emplace(words[1], words[2]);
            }
            REQUIRE(Instructions(code, 247).empty()); // OpSelectionMerge: no lane-dependent branch
            REQUIRE(HasCapability(code, 68) == quad); // GroupNonUniformQuad
            // The loop runs once per K8 group.
            const auto limits = Instructions(code, 176); // OpULessThan
            REQUIRE(limits.size() == 1);
            REQUIRE(constants.at(limits.front()[3]) == depth / 8);
            const HelperCounts counts = CountHelper(code);
            // Four K pairs x 2 rows x 2 columns x 2 halves, then the four accumulators rounded.
            REQUIRE(counts.fma == 32);
            REQUIRE(counts.fma_before_rounding == std::vector<u32>(4, 32));
            // B always needs a general shuffle (two columns); A comes from its own quad.
            REQUIRE(counts.shuffles == (quad ? 8U : 16U));
            REQUIRE(counts.quad_broadcasts == (quad ? 8U : 0U));
            for (u32 i = 0; i < counts.quad_indices.size(); ++i) {
                REQUIRE(counts.quad_indices[i] == i / 2);
            }
        }
    }
}

TEST_CASE("Software HMMA rounds through a half when the shader has one",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram fixture{1, depth};
        fixture.program.info.uses_fp16 = true;
        const auto code = fixture.Emit();
        const HelperCounts counts = CountHelper(code);
        REQUIRE(counts.fma_before_rounding.empty());
        // Narrow and widen each of the four accumulators, both kept exact.
        REQUIRE(counts.conversions == 8);
        const auto decorations = Instructions(code, 71); // OpDecorate
        for (const u32 id : counts.conversion_ids) {
            REQUIRE(std::ranges::any_of(decorations, [=](const auto& words) {
                return words[0] == id && words[1] == 42; // NoContraction
            }));
        }
    }
}

TEST_CASE("Software matrix helpers also emit for 64-lane host subgroups",
          "[video_core][tensor][ampere]") {
    for (const u32 depth : {8U, 16U}) {
        TensorProgram fixture{1, depth};
        fixture.profile.max_subgroup_size = 64;
        const auto code = fixture.Emit(depth == 8 ? "software-k8-wave64" : "software-k16-wave64");
        REQUIRE_FALSE(HasCapability(code, 5357));
        REQUIRE(Instructions(code, 246).size() == 1);
        REQUIRE(Instructions(code, 345).size() == 16); // OpGroupNonUniformShuffle
    }
}

TEST_CASE("Captured Bananza matrix shaders lower without an environment override",
          "[video_core][tensor][ampere]") {
    TensorMode mode{nullptr};
    for (const Ampere::Instruction instruction : {
             Ampere::Instruction{{0x102c723c, 0x30, 0x8ff, 0x04aff000}},
             Ampere::Instruction{{0x1020723c, 0x24, 0x8ff, 0x04aff000}},
             Ampere::Instruction{{0x1418723c, 0x1c, 0x8ff, 0x04aff000}}}) {
        const std::array code{instruction,
                              Ampere::Instruction{{0x794d, 0, 0x03800000, 0}}};
        const auto lowered = Ampere::LowerToMaxwell(code, false, true);
        REQUIRE(lowered);
        REQUIRE(lowered.instructions.size() == 2);
        REQUIRE(Maxwell::Decode(lowered.instructions[0]) == Maxwell::Opcode::TENSOR_MUL_ADD);
    }
}

TEST_CASE("Native matrix lowering selects compact software helpers without legacy toggles", "[video_core][tensor][ampere]") {
    TensorMode mode{nullptr};
    std::array<Ampere::Instruction, 2> code{{
        {{0x0810723c, 12, 0x8ff, 0}}, {{0x0000794d, 0, 0x03800000, 0}}}};
    const auto supported = Ampere::LowerToMaxwell(code, false, false);
    REQUIRE(static_cast<bool>(supported));
    REQUIRE(supported.instructions.size() > 100);
    const auto compact = Ampere::LowerToMaxwell(code, false, true);
    REQUIRE(compact);
    REQUIRE(compact.instructions.size() == 2);
    REQUIRE(Maxwell::Decode(compact.instructions.front()) == Maxwell::Opcode::TENSOR_MUL_ADD);
    REQUIRE((compact.instructions.front() & (1ULL << 48)) != 0);
    for (const char* legacy : {"fast", "stub", "off"}) {
        TensorMode::Set(legacy);
        REQUIRE(Ampere::LowerToMaxwell(code, false, true).instructions == compact.instructions);
        REQUIRE(Ampere::LowerToMaxwell(code).instructions == supported.instructions);
    }
    code[0].words[0] &= ~0x7000U;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
    code[0].words[0] |= 0x7000;
    code[0].words[3] = 1;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
    code[0].words[3] = 0;
    code[0].words[0] += 1U << 24;
    REQUIRE_FALSE(static_cast<bool>(Ampere::LowerToMaxwell(code, false, true)));
}
