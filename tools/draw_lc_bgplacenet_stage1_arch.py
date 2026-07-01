"""
绘制 LC-BGPlaceNet Stage 1 (W³ + CamPE 版) 顶会风格模型架构图。

依据 src/models/lc_bgplacenet/stage1.py 的前向数据流绘制:
    体素点云 -> SpConv backbone -> f_3d
    体素点云 + 相机位姿(E_c2w) -> CamPE -> pos_embed_cam (Source/Support 共用)
    指令文本 -> RoBERTa(frozen) -> token 特征
        -> Voxel-Language Fusion (dense) -> f_vl
        -> W³ Language Routing -> what / where / whole 三路
    what  -> Source Grounding Head 的 query -> source_box
    where -> FiLM 调制 Support Head       -> support_logits
    whole -> 透传 Stage 2

用法:
    conda run -n spatial python tools/draw_lc_bgplacenet_stage1_arch.py \
        --output docs/figures/lc_bgplacenet_stage1_arch.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# 顶会常用的柔和配色:每类模块一组 (填充, 描边)
PALETTE = {
    "input": ("#EAF0F6", "#7C93A8"),
    "backbone": ("#9FC0E8", "#2F5C93"),
    "text": ("#BFE3C0", "#3E8E4E"),
    "w3": ("#B8DCE0", "#2E8B94"),
    "campe": ("#D7E1F5", "#3A5BA0"),
    "fusion": ("#F7D9A6", "#C9852A"),
    "src": ("#D9C2E9", "#7A4FA0"),
    "sup": ("#F6C7B8", "#C85F3E"),
    "loss": ("#F2B8B5", "#B23A34"),
    "aux": ("#E4E9EF", "#8A97A6"),
}

GREEN = "#2E8B57"  # W³ 三路语义连线统一用绿色
BLUE = "#3A5BA0"   # CamPE 位置编码连线统一用蓝色
GREY = "#8A97A6"


def add_block(ax, xy, w, h, text, kind, fontsize=10, bold=True, subtext=None):
    """绘制一个圆角模块块,返回其中心与边界坐标字典。"""
    face, edge = PALETTE[kind]
    x, y = xy
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        linewidth=1.6, edgecolor=edge, facecolor=face, zorder=3,
    )
    ax.add_patch(box)
    cx, cy = x + w / 2, y + h / 2
    weight = "bold" if bold else "normal"
    label_y = cy + (0.16 if subtext else 0.0)
    ax.text(cx, label_y, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color="#1B2733", zorder=4)
    if subtext:
        ax.text(cx, cy - 0.20, subtext, ha="center", va="center",
                fontsize=fontsize - 2.5, color="#42525F", zorder=4)
    return {
        "cx": cx, "cy": cy,
        "left": (x, cy), "right": (x + w, cy),
        "top": (cx, y + h), "bottom": (cx, y),
    }


def add_arrow(ax, p0, p1, color="#2B3A45", style="-|>", lw=1.8, rad=0.0, ls="-"):
    """绘制带箭头的数据流连线。"""
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=14, linewidth=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=2,
    ))


def add_group(ax, x, y, w, h, title, color="#9AA7B4"):
    """绘制分组虚线框及标题。"""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=1.3, edgecolor=color, facecolor="none",
        linestyle=(0, (6, 4)), zorder=1,
    ))
    ax.text(x + 0.12, y + h - 0.02, title, ha="left", va="top",
            fontsize=9, style="italic", color=color, zorder=1)


def add_tensor(ax, p, text, color="#5A6B78", dy=0.30):
    """在连线上标注张量维度 / 语义。"""
    ax.text(p[0], p[1] + dy, text, ha="center", va="center",
            fontsize=7.5, color=color, style="italic", zorder=5)


def build_figure(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(16.6, 9.4))
    ax.set_xlim(0, 16.6)
    ax.set_ylim(-0.4, 9.4)
    ax.axis("off")

    # ---- 标题 ----
    ax.text(8.3, 9.0, "LC-BGPlaceNet — Stage 1 with W³ Text Routing + CamPE",
            ha="center", va="center", fontsize=15, fontweight="bold", color="#132430")
    ax.text(8.3, 8.55,
            "What / Where / Whole language decoupling + Camera-Relative Positional Encoding "
            "for direction-aware grounding",
            ha="center", va="center", fontsize=9.5, color="#4A5A67", style="italic")

    # ---- 输入 ----
    voxel_in = add_block(ax, (0.3, 5.55), 2.5, 1.0, "Active Voxel\nPoint Cloud",
                         "input", subtext="N x 6  (XYZ + RGB)")
    camera_in = add_block(ax, (0.3, 3.75), 2.5, 0.85, "Camera Pose\n(per-frame E_c2w)",
                          "input", fontsize=9)
    text_in = add_block(ax, (0.3, 1.75), 2.5, 1.0, "Language\nInstruction",
                        "input", subtext='"move the ... onto ..."')

    # ---- 编码器 ----
    add_group(ax, 3.35, 4.7, 2.95, 2.5, "3D Encoder")
    backbone = add_block(ax, (3.55, 5.35), 2.55, 1.4, "SpConv Backbone",
                         "backbone", subtext="3 x SubMConv3d\nBN + ReLU")
    add_group(ax, 3.35, 1.05, 2.95, 2.5, "Text Encoder (frozen)")
    roberta = add_block(ax, (3.55, 1.6), 2.55, 1.4, "RoBERTa-base",
                        "text", subtext="frozen + Linear proj")

    # ---- 融合 + CamPE + W³ 路由(同一列纵向堆叠,CamPE 居中方便两头共用) ----
    fusion = add_block(ax, (7.0, 5.2), 2.7, 1.4, "Voxel-Language\nFusion",
                       "fusion", subtext="Cross-Attention\n8 heads + FFN")
    campe = add_block(ax, (7.0, 3.65), 2.7, 1.0, "CamPE",
                      "campe", fontsize=10, subtext="R_w2c @ (p - cam)")
    w3 = add_block(ax, (7.0, 1.55), 2.7, 1.55, "W³ Language Routing",
                   "w3", subtext="3 learnable role queries\nattention pooling")
    stage2 = add_block(ax, (7.05, 0.18), 2.6, 0.62, "→ Stage 2 module",
                       "aux", fontsize=9)

    # ---- 预测头 ----
    src_head = add_block(ax, (10.9, 5.15), 3.05, 1.5, "Source Grounding\nHead",
                         "src", subtext="TransformerDecoder x4\n+ learnable query")
    sup_head = add_block(ax, (10.9, 1.5), 3.05, 1.5, "Support Head",
                         "sup", subtext="per-voxel MLP\n+ FiLM(where)")

    # ---- 输出 / 损失 ----
    src_out = add_block(ax, (12.4, 7.15), 2.3, 0.72, "source_box",
                        "aux", fontsize=9, subtext="(cx,cy,cz,l,w,h)")
    sup_out = add_block(ax, (12.4, 0.2), 2.3, 0.7, "support_logits",
                        "aux", fontsize=9, subtext="N x 1")
    src_loss = add_block(ax, (14.9, 5.4), 1.5, 1.0, "L_src",
                         "loss", fontsize=9, subtext="L1 center\n+ L1 size\n+ IoU")
    sup_loss = add_block(ax, (14.9, 1.75), 1.5, 1.0, "L_sup",
                         "loss", fontsize=9, subtext="BCE")

    # ---- 主数据流 ----
    add_arrow(ax, voxel_in["right"], backbone["left"])
    add_arrow(ax, text_in["right"], roberta["left"])

    # 世界坐标 + 相机位姿 -> CamPE(绕过两个 Encoder 分组框,走中间空档)
    add_arrow(ax, (2.8, 5.7), campe["left"], color=BLUE, rad=-0.3)
    add_tensor(ax, (4.9, 5.55), "world_coords", color=BLUE, dy=0.0)
    add_arrow(ax, camera_in["right"], campe["left"], color=BLUE, rad=-0.08)

    # backbone -> fusion (f_3d)
    add_arrow(ax, backbone["right"], (7.0, 5.95), rad=-0.1)
    add_tensor(ax, (6.15, 6.35), "f_3d : N x 256")

    # roberta -> W³  与  roberta -> fusion(密集 token)
    add_arrow(ax, roberta["right"], w3["left"])
    add_tensor(ax, (6.55, 2.35), "text tokens")
    add_arrow(ax, roberta["right"], (7.0, 5.35), rad=0.28)
    add_tensor(ax, (6.55, 4.85), "token feats (dense)")

    # fusion -> heads (f_vl)
    add_arrow(ax, fusion["right"], src_head["left"])
    add_tensor(ax, (10.3, 5.95), "f_vl : N x 256")
    add_arrow(ax, (9.7, 5.5), sup_head["left"], rad=-0.22)

    # W³ 三路语义连线(绿色虚线)
    add_arrow(ax, (9.7, 2.75), src_head["bottom"], color=GREEN, ls="--", lw=1.5, rad=-0.18)
    add_tensor(ax, (11.15, 3.85), "what → query", color=GREEN)
    add_arrow(ax, (9.7, 2.05), (10.9, 2.05), color=GREEN, ls="--", lw=1.5)
    add_tensor(ax, (10.3, 2.05), "where (FiLM)", color=GREEN, dy=-0.28)
    add_arrow(ax, w3["bottom"], stage2["top"], color=GREEN, ls="--", lw=1.5)
    add_tensor(ax, (8.95, 1.15), "whole", color=GREEN, dy=0.0)

    # CamPE -> Source Grounding Head 和 Support Head(两头共用同一份 pos_embed_cam)
    add_arrow(ax, campe["right"], src_head["bottom"], color=BLUE, rad=0.28)
    add_tensor(ax, (9.55, 5.35), "pos_embed_cam", color=BLUE)
    add_arrow(ax, campe["right"], sup_head["left"], color=BLUE, rad=-0.15)

    # heads -> outputs -> loss
    add_arrow(ax, (12.425, 6.65), src_out["bottom"])
    add_arrow(ax, src_out["right"], (15.6, 6.4), rad=-0.12)
    add_arrow(ax, sup_head["bottom"], sup_out["top"])
    add_arrow(ax, sup_out["right"], (15.6, 1.75), rad=0.12)

    ax.text(15.65, 4.0, "Total = λ_src·L_src + λ_sup·L_sup",
            ha="center", va="center", fontsize=8.5, color="#7A2621", fontweight="bold")

    # ---- 图例 ----
    legend_items = [
        ("3D branch", "backbone"),
        ("Text branch", "text"),
        ("W³ routing", "w3"),
        ("CamPE", "campe"),
        ("Fusion", "fusion"),
        ("Source head", "src"),
        ("Support head", "sup"),
        ("Loss", "loss"),
    ]
    lx = 0.5
    for label, kind in legend_items:
        face, edge = PALETTE[kind]
        ax.add_patch(FancyBboxPatch((lx, -0.28), 0.3, 0.3,
                     boxstyle="round,pad=0.01,rounding_size=0.05",
                     facecolor=face, edgecolor=edge, linewidth=1.2))
        ax.text(lx + 0.4, -0.13, label, ha="left", va="center", fontsize=8.5, color="#33424E")
        lx += 1.98

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] 架构图已保存: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 LC-BGPlaceNet Stage 1 (W³) 架构图")
    parser.add_argument("--output", type=Path,
                        default=Path("docs/figures/lc_bgplacenet_stage1_arch.png"),
                        help="输出 PNG 路径")
    args = parser.parse_args()
    build_figure(args.output)


if __name__ == "__main__":
    main()
