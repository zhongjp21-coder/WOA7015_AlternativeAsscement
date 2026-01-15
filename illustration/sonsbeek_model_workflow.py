import matplotlib.pyplot as plt

def draw_model_architecture(save_path="medvqa_model_architecture.png"):
    # 字体保持默认，不换行
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis("off")

    # 模块样式
    box_style = dict(boxstyle="round,pad=1.0", ec="#4A90E2", lw=1.5)
    text_style = dict(ha="center", va="center", fontsize=11)
    arrow_color = "#555555"

    # 模块纵向位置
    y_positions = [
        0.88,  # Medical Image
        0.80,  # CLIP
        0.72,  # Image Feature
        0.64,  # Mapping
        0.56,  # Visual Prefix
        0.46,  # Concatenation
        0.36,  # GPT-2
        0.28   # Generated Answer
    ]

    # 模块文字和颜色
    boxes = [
        ("Medical Image", "#EAF4FF"),
        ("CLIP Image Encoder (Frozen)", "#EAF4FF"),
        ("Image Feature (512-d)", "#F0FAF7"),
        ("Mapping Network (Trainable) Linear → ReLU → Linear", "#E8F8F2"),
        ("Visual Prefix Tokens (Length = 10)", "#F0FAF7"),
        ("Concatenation", "#FFF7E6"),
        ("GPT-2 Language Model (Frozen)", "#EAF4FF"),
        ("Generated Answer", "#E8F0FE"),
    ]

    # 画模块
    for (text, color), y in zip(boxes, y_positions):
        ax.text(
            0.5, y, text,
            bbox=dict(**box_style, fc=color),
            **text_style
        )

    # 主箭头（竖向）
    for i in range(len(y_positions) - 1):
        if y_positions[i + 1] == 0.46:  # Concatenation 上方箭头单独处理
            continue
        ax.annotate(
            "",
            xy=(0.5, y_positions[i + 1] + 0.02),
            xytext=(0.5, y_positions[i] - 0.02),
            arrowprops=dict(arrowstyle="->", lw=1.5, color=arrow_color)
        )

    # Visual Prefix → Concatenation 竖向箭头
    ax.annotate(
        "",
        xy=(0.5, 0.46 + 0.02),
        xytext=(0.5, 0.56 - 0.02),
        arrowprops=dict(arrowstyle="->", lw=1.5, color=arrow_color)
    )

    # ===== Question Tokens（紫色模块） =====
    q_x = 0.15
    q_y = 0.46  # 对齐 Concatenation 模块中轴
    ax.text(
        q_x, q_y,
        "Question Tokens (GPT-2 Embeddings)",
        bbox=dict(boxstyle="round,pad=1.0", fc="#F3E8FF", ec="#9B7EDC", lw=1.5),
        ha="center", va="center", fontsize=10
    )

    # Question → Concatenation 水平箭头（微调起点离模块稍远一点，避免重合）
    arrow_start_x = q_x + 0.15  # 适当偏右一点，不碰模块
    ax.annotate(
        "",
        xy=(0.42, q_y),            # Concatenation 左侧中轴
        xytext=(arrow_start_x, q_y), # Question Tokens 箭头起点
        arrowprops=dict(arrowstyle="->", lw=1.5, color=arrow_color)
    )

    # 标题
    ax.text(
        0.5, 0.95,
        "Prefix-based MedVQA Model Architecture",
        ha="center", va="center",
        fontsize=14, fontweight="bold"
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Architecture figure saved to: {save_path}")


if __name__ == "__main__":
    draw_model_architecture("medvqa_model_architecture.png")
