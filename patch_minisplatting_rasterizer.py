#!/usr/bin/env python3
"""Apply the contribution-area instrumentation to the project's pinned dr_aa rasterizer.

Run from the gaussian-splatting repository root in Colab:
    %cd /content/gaussian-splatting-ecng3020
    !python patch_minisplatting_rasterizer.py

This targets submodule commit 9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0.
It stops on mismatched source text rather than applying a partial/unsafe patch.
"""
from pathlib import Path
import subprocess

ROOT = Path("submodules/diff-gaussian-rasterization")
EXPECTED = "9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0"

if not ROOT.is_dir():
    raise SystemExit(f"Missing submodule: {ROOT.resolve()}")
head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
if head != EXPECTED:
    raise SystemExit(f"Wrong rasterizer revision: {head}; expected {EXPECTED}")

def replace(path, pairs):
    p = ROOT / path
    s = p.read_text()
    for old, new, label in pairs:
        n = s.count(old)
        if n != 1:
            raise RuntimeError(f"{path}: expected one match for {label}, found {n}; no write made to this file.")
        s = s.replace(old, new, 1)
    p.write_text(s)
    print("Patched", path)

replace("cuda_rasterizer/forward.h", [
    ("\t\tfloat* out_color,\n\t\tfloat* depths,\n\t\tfloat* depth);",
     "\t\tfloat* out_color,\n\t\tfloat* depths,\n\t\tfloat* depth,\n\t\tint* max_contribution_area);", "FORWARD::render declaration"),
])
replace("cuda_rasterizer/forward.cu", [
    ("\tconst float* __restrict__ depths,\n\tfloat* __restrict__ invdepth)",
     "\tconst float* __restrict__ depths,\n\tfloat* __restrict__ invdepth,\n\tint* __restrict__ max_contribution_area)", "renderCUDA signature"),
    ("\tfloat C[CHANNELS] = { 0 };\n\n\tfloat expected_invdepth = 0.0f;",
     "\tfloat C[CHANNELS] = { 0 };\n\tfloat max_contribution_weight = 0.0f;\n\tint max_contribution_gaussian = -1;\n\n\tfloat expected_invdepth = 0.0f;", "pixel contribution state"),
    ("\t\t\tif (test_T < 0.0001f)\n\t\t\t{\n\t\t\t\tdone = true;\n\t\t\t\tcontinue;\n\t\t\t}\n\n\t\t\t// Eq. (3) from 3D Gaussian splatting paper.",
     "\t\t\tif (test_T < 0.0001f)\n\t\t\t{\n\t\t\t\tdone = true;\n\t\t\t\tcontinue;\n\t\t\t}\n\n\t\t\t// Track the strongest alpha-blending weight alpha*T for this pixel.\n\t\t\tfloat contribution_weight = alpha * T;\n\t\t\tif (contribution_weight > max_contribution_weight)\n\t\t\t{\n\t\t\t\tmax_contribution_weight = contribution_weight;\n\t\t\t\tmax_contribution_gaussian = collected_id[j];\n\t\t\t}\n\n\t\t\t// Eq. (3) from 3D Gaussian splatting paper.", "winner update"),
    ("\tif (inside)\n\t{\n\t\tfinal_T[pix_id] = T;",
     "\tif (inside)\n\t{\n\t\t// Each pixel votes for at most one Gaussian: its largest alpha*T contributor.\n\t\tif (max_contribution_gaussian >= 0)\n\t\t\tatomicAdd(&max_contribution_area[max_contribution_gaussian], 1);\n\t\tfinal_T[pix_id] = T;", "per-pixel atomic count"),
    ("\tfloat* out_color,\n\tfloat* depths,\n\tfloat* depth)\n{\n\trenderCUDA<NUM_CHANNELS>",
     "\tfloat* out_color,\n\tfloat* depths,\n\tfloat* depth,\n\tint* max_contribution_area)\n{\n\trenderCUDA<NUM_CHANNELS>", "FORWARD::render definition"),
    ("\t\tdepths, \n\t\tdepth);",
     "\t\tdepths, \n\t\tdepth,\n\t\tmax_contribution_area);", "renderCUDA launch"),
])
replace("cuda_rasterizer/rasterizer.h", [
    ("\t\t\tfloat* depth,\n\t\t\tbool antialiasing,",
     "\t\t\tfloat* depth,\n\t\t\tint* max_contribution_area,\n\t\t\tbool antialiasing,", "Rasterizer::forward declaration"),
])
replace("cuda_rasterizer/rasterizer_impl.cu", [
    ("\tfloat* out_color,\n\tfloat* depth,\n\tbool antialiasing,",
     "\tfloat* out_color,\n\tfloat* depth,\n\tint* max_contribution_area,\n\tbool antialiasing,", "Rasterizer::forward definition"),
    ("\t\tout_color,\n\t\tgeomState.depths,\n\t\tdepth), debug)",
     "\t\tout_color,\n\t\tgeomState.depths,\n\t\tdepth,\n\t\tmax_contribution_area), debug)", "FORWARD::render invocation"),
])
replace("rasterize_points.cu", [
    ("  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));",
     "  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));\n  torch::Tensor max_contribution_area = torch::zeros({P}, int_opts);", "count tensor allocation"),
    ("\t\tout_color.contiguous().data<float>(),\n\t\tout_invdepthptr,\n\t\tantialiasing,",
     "\t\tout_color.contiguous().data<float>(),\n\t\tout_invdepthptr,\n\t\tmax_contribution_area.contiguous().data_ptr<int>(),\n\t\tantialiasing,", "pass count pointer"),
    ("  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth);",
     "  return std::make_tuple(rendered, out_color, radii, geomBuffer, binningBuffer, imgBuffer, out_invdepth, max_contribution_area);", "append count to return tuple"),
])
replace("rasterize_points.h", [
    ("std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>\nRasterizeGaussiansCUDA(",
     "std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>\nRasterizeGaussiansCUDA(", "C++ forward tuple declaration"),
])
replace("diff_gaussian_rasterization/__init__.py", [
    ("num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths = _C.rasterize_gaussians(*args)",
     "num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer, invdepths, max_contribution_area = _C.rasterize_gaussians(*args)", "C++ forward unpack"),
    ("        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)\n        return color, radii, invdepths",
     "        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, opacities, geomBuffer, binningBuffer, imgBuffer)\n        ctx.mark_non_differentiable(max_contribution_area)\n        return color, radii, invdepths, max_contribution_area", "autograd outputs"),
    ("    def backward(ctx, grad_out_color, _, grad_out_depth):",
     "    def backward(ctx, grad_out_color, _, grad_out_depth, __):", "autograd backward signature"),
])
print("\nPatch applied to the pinned dr_aa source.")
print("Next: rebuild the extension, then run the synthetic validation cell.")
