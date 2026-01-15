import pandas as pd
import os

def analyze_vqa_results(result_excel_path, original_csv_path):
    print("正在读取文件...")
    # 1. 读取模型生成的预测结果
    if not os.path.exists(result_excel_path):
        print(f"错误: 找不到结果文件 {result_excel_path}")
        return
    df_preds = pd.read_excel(result_excel_path)
    
    # 2. 读取原始数据集 (为了获取 Open/Closed 标签)
    if not os.path.exists(original_csv_path):
        print(f"错误: 找不到原始数据文件 {original_csv_path}")
        return
    df_original = pd.read_csv(original_csv_path)

    # --- 数据清洗与合并 ---
    print("正在合并数据以获取 Question Type...")
    
    # 1. 处理图片列名
    df_preds['image_filename'] = df_preds['Image'].apply(lambda x: os.path.basename(str(x)))
    
    # 智能查找原始CSV的图片列 (防止 KeyError)
    possible_cols = ['image_name', 'image', 'image_id', 'file_name', 'Image']
    img_col = None
    for col in possible_cols:
        if col in df_original.columns:
            img_col = col
            break
            
    if img_col is None:
        print("错误：在CSV中没找到图片列，尝试默认使用第一列...")
        img_col = df_original.columns[0]

    df_original['image_filename'] = df_original[img_col].apply(lambda x: os.path.basename(str(x)))
    
    # 2. 统一 Question 格式
    df_preds['join_key'] = df_preds['Question'].astype(str).str.strip().str.lower()
    
    # 智能查找原始CSV的问题列
    q_col = 'question' if 'question' in df_original.columns else 'Question'
    df_original['join_key'] = df_original[q_col].astype(str).str.strip().str.lower()
    
    # 3. 合并：把原始数据里的 'answer_type' (CLOSED/OPEN) 拼接到预测结果里
    # 查找可能的 type 列名
    type_col = 'answer_type'
    if 'answer_type' not in df_original.columns:
        if 'question_type' in df_original.columns:
            type_col = 'question_type'
        else:
            type_col = None

    if type_col:
        df_merged = pd.merge(
            df_preds, 
            df_original[['image_filename', 'join_key', type_col]], 
            on=['image_filename', 'join_key'], 
            how='left'
        )
        # 重命名为标准名称
        df_merged.rename(columns={type_col: 'answer_type'}, inplace=True)
    else:
        print("警告: 原始CSV中未找到 'answer_type' 列，使用规则生成...")
        df_merged = df_preds.copy()
        df_merged['answer_type'] = df_merged['True Answer'].apply(
            lambda x: 'CLOSED' if str(x).lower().strip() in ['yes', 'no'] else 'OPEN'
        )

    # 4. 【关键修复步骤】清洗 answer_type 列
    # 填充缺失值 -> 转字符串 -> 去除首尾空格 -> 转大写
    df_merged['answer_type'] = df_merged['answer_type'].fillna('OPEN')
    df_merged['answer_type'] = df_merged['answer_type'].astype(str).str.strip().str.upper()

    # 有时候数据里会有 'OTHER' 或其他奇怪的类型，强制归类为 OPEN (如果不是CLOSED的话)
    # 这一步是为了确保只出现 CLOSED 和 OPEN 两个类别
    df_merged['answer_type'] = df_merged['answer_type'].apply(lambda x: 'CLOSED' if x == 'CLOSED' else 'OPEN')
    
    # --- 核心计算：分组统计 ---
    print("\n正在计算各类指标均值...")
    
    # 按 answer_type 分组计算均值
    summary = df_merged.groupby('answer_type')[['Exact Match', 'BLEU']].mean()
    
    # 转换为百分比格式方便阅读
    summary_display = summary.copy()
    summary_display['Exact Match'] = (summary_display['Exact Match'] * 100).round(2).astype(str) + '%'
    summary_display['BLEU'] = summary_display['BLEU'].round(4)
    
    # 计算整体均值 (Overall)
    overall_acc = df_merged['Exact Match'].mean()
    overall_bleu = df_merged['BLEU'].mean()
    
    # --- 输出结果 ---
    print("\n" + "="*40)
    print("      FINAL REPORT DATA TABLE      ")
    print("="*40)
    print(summary_display)
    print("-" * 40)
    print(f"Overall Accuracy: {overall_acc*100:.2f}%")
    print(f"Overall BLEU:     {overall_bleu:.4f}")
    print("="*40)

    # --- 保存详细分析结果 (包含 Type) ---
    output_dir = os.path.dirname(result_excel_path) # 保存到结果文件的同级目录
    output_path = os.path.join(output_dir, "analyzed_results_with_type.xlsx")
    df_merged.to_excel(output_path, index=False)
    print(f"\n详细分类数据已保存至: {output_path}")

# 执行分析
if __name__ == "__main__":
    # 请确保这两个路径正确
    RESULTS_FILE = "./llm_results/final_report_results.xlsx" 
    ORIGINAL_CSV = "train_split.csv" # 确保这里指向正确的文件
    
    analyze_vqa_results(RESULTS_FILE, ORIGINAL_CSV)