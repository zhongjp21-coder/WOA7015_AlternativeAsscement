import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# ============================================
# 计算每一行的 BLEU-1
# ============================================
def get_bleu1(row):
    """
    计算 BLEU-1 分数 (1-gram)。
    row: DataFrame 一行，必须包含 'True Answer' 和 'Pred Answer'
    """
    # 处理文本：小写、去首尾空格、分词
    ref = str(row.get('True Answer', '')).lower().strip().split()
    cand = str(row.get('Pred Answer', '')).lower().strip().split()

    # 避免空预测
    if len(cand) == 0 or len(ref) == 0:
        return 0.0

    # BLEU-1: weights=(1.0,0,0,0)
    return sentence_bleu(
        [ref],
        cand,
        weights=(1.0, 0, 0, 0),
        smoothing_function=SmoothingFunction().method1
    )


# ============================================
# 主函数：生成新 Excel 文件，添加 BLEU-1
# ============================================
def update_results_to_bleu1(excel_path):
    print(f"正在读取: {excel_path}")
    df = pd.read_excel(excel_path)

    print("正在重新计算 BLEU-1...")
    df['BLEU-1'] = df.apply(get_bleu1, axis=1)

    # 计算 OPEN 问题平均分，如果存在 answer_type 列
    if 'answer_type' in df.columns:
        open_mask = df['answer_type'].astype(str).str.strip().str.upper() == 'OPEN'
        avg_bleu1 = df[open_mask]['BLEU-1'].mean()
        print(f"\n[Open-Ended Only] Average BLEU-1: {avg_bleu1:.4f}")
    else:
        avg_bleu1 = df['BLEU-1'].mean()
        print(f"\n[Overall] Average BLEU-1: {avg_bleu1:.4f}")

    # 保存新 Excel 文件
    new_path = excel_path.replace(".xlsx", "_with_BLEU1.xlsx")
    df.to_excel(new_path, index=False)
    print(f"新结果已保存至: {new_path}")
    return new_path


# ============================================
# 调用示例
# ============================================
if __name__ == "__main__":
    RESULTS_FILE = "./llm_results/final_report_results.xlsx"
    update_results_to_bleu1(RESULTS_FILE)
