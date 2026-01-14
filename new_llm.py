import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, CLIPProcessor, CLIPModel
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt

# ==========================================
# 1. 模型定义 (保持不变)
# ==========================================
class MedVQAModel(nn.Module):
    def __init__(self, prefix_length=10, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.prefix_length = prefix_length
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        
        for param in self.clip_model.parameters(): param.requires_grad = False
        for param in self.gpt2.parameters(): param.requires_grad = False
        
        clip_dim = self.clip_model.config.projection_dim
        gpt_dim = self.gpt2.config.n_embd
        self.mapping_network = nn.Sequential(
            nn.Linear(clip_dim, gpt_dim),
            nn.ReLU(),
            nn.Linear(gpt_dim, prefix_length * gpt_dim)
        )

    def forward(self, pixel_values, input_ids, labels=None):
        with torch.no_grad():
            vision_outputs = self.clip_model.get_image_features(pixel_values=pixel_values)
        
        visual_prefixes = self.mapping_network(vision_outputs).view(
            -1, self.prefix_length, self.gpt2.config.n_embd
        )
        text_embeds = self.gpt2.transformer.wte(input_ids)
        full_embeds = torch.cat((visual_prefixes, text_embeds), dim=1)
        
        if labels is not None:
            prefix_labels = torch.full((labels.size(0), self.prefix_length), -100).to(labels.device)
            full_labels = torch.cat((prefix_labels, labels), dim=1)
            return self.gpt2(inputs_embeds=full_embeds, labels=full_labels)
        
        return self.gpt2(inputs_embeds=full_embeds)

# ==========================================
# 2. 数据集类 (保持不变)
# ==========================================
class MedVQADataset(Dataset):
    def __init__(self, data_list, tokenizer, processor, prefix_length):
        self.data = data_list
        self.tokenizer = tokenizer
        self.processor = processor
        self.prefix_length = prefix_length

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        
        full_text = f"Question: {item['question']} Answer: {item['answer']} <|endoftext|>"
        tokens = self.tokenizer(full_text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        
        input_ids = tokens.input_ids.squeeze()
        return pixel_values, input_ids, input_ids 

# ==========================================
# 3. 新增：真实数据加载与评估功能
# ==========================================

def load_vqa_rad_data(csv_path, image_folder):
    """
    读取VQA-RAD CSV文件并转换为列表格式
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # !!! 请根据你的CSV实际列名修改这里 !!!
    # 假设CSV列名为: 'image_name', 'question', 'answer'
    # 如果你的列名是 'image_id' 或其他，请相应修改
    cleaned_data = []
    for idx, row in df.iterrows():
        img_name = str(row.get('image_name', row.get('image', ''))) # 尝试获取图片名
        question = str(row.get('question', ''))
        answer = str(row.get('answer', ''))
        
        if not img_name or not question or not answer:
            continue # 跳过缺失数据的行

        full_img_path = os.path.join(image_folder, img_name)
        
        # 检查图片是否存在，避免报错
        if os.path.exists(full_img_path):
            cleaned_data.append({
                'image_path': full_img_path,
                'question': question,
                'answer': answer
            })
    
    print(f"Successfully loaded {len(cleaned_data)} valid samples.")
    return cleaned_data

def evaluate_model(model, test_data, tokenizer, processor, device):
    """
    在测试集上运行推理并计算BLEU分数
    """
    print("\nStarting Evaluation on Test Set...")
    model.eval()
    results = []
    total_bleu = 0
    exact_matches = 0
    smoothing = SmoothingFunction().method1 # 用于短文本的平滑处理

    # 这里的循环不使用DataLoader，而是逐个处理以便于记录详细文本
    for i, item in enumerate(test_data):
        image_path = item['image_path']
        question = item['question']
        true_answer = item['answer'].lower().strip()
        
        # 1. 准备输入
        prompt = f"Question: {question} Answer:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

        # 2. 生成
        with torch.no_grad():
            vision_outputs = model.clip_model.get_image_features(pixel_values=pixel_values)
            prefix_embeds = model.mapping_network(vision_outputs).view(1, 10, -1)
            text_embeds = model.gpt2.transformer.wte(input_ids)
            inputs_embeds = torch.cat((prefix_embeds, text_embeds), dim=1)
            
            output_ids = model.gpt2.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=15, # 答案通常不长
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 3. 解码
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # 简单的后处理：因为模型可能生成后续问题，我们截取Answer后的部分
        # 注意：这里取决于你的generate输出是否包含prompt，通常inputs_embeds方式生成只包含新token
        # 但GPT2 decode通常是整个序列，如果只含新token则直接用
        pred_answer = generated_text.strip().lower()
        
        # 4. 计算指标
        # BLEU需要分词列表
        ref_tokens = true_answer.split()
        cand_tokens = pred_answer.split()
        bleu_score = sentence_bleu([ref_tokens], cand_tokens, smoothing_function=smoothing)
        
        is_exact_match = 1 if true_answer in pred_answer else 0 # 宽松匹配
        
        total_bleu += bleu_score
        exact_matches += is_exact_match
        
        # 保存详细结果用于分析
        results.append({
            "Image": os.path.basename(image_path),
            "Question": question,
            "True Answer": true_answer,
            "Pred Answer": pred_answer,
            "BLEU": bleu_score,
            "Exact Match": is_exact_match
        })
        
        if i % 20 == 0:
            print(f"Processed {i}/{len(test_data)} samples...")

    # 汇总统计
    avg_bleu = total_bleu / len(test_data)
    accuracy = exact_matches / len(test_data)
    
    print(f"\n=== Evaluation Results ===")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Accuracy (Soft Exact Match): {accuracy:.4f}")
    
    return pd.DataFrame(results)

# ==========================================
# 4. 主执行流
# ==========================================
def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # --- 配置路径 (请修改这里) ---
    csv_file_path = "VQA_RAD Dataset Public.csv" # 指向你的CSV文件
    image_folder_path = "VQA_RAD Image Folder"   # 指向你的图片文件夹
    
    # 1. 加载与分割数据
    # 根据Preliminary Report ，我们需要划分 80% Train, 20% Test
    full_data = load_vqa_rad_data(csv_file_path, image_folder_path)
    if len(full_data) == 0:
        print("Error: No data loaded. Check your paths and CSV column names.")
        return

    
    train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=42)
    print(f"Data Split: {len(train_data)} Train, {len(test_data)} Test")

    # 2. 初始化模型
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = MedVQAModel().to(device)
    
    # 3. 训练 (此处简化轮数供测试，实际报告建议跑 30 Epochs [cite: 285])
    train_dataset = MedVQADataset(train_data, tokenizer, processor, prefix_length=10)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # Batch size 32可能显存不够，改16
    optimizer = torch.optim.AdamW(model.mapping_network.parameters(), lr=1e-4) # 学习率 [cite: 352]
    
    print("\nStarting Training...")
    model.train()
    loss_history = []
    
    # 演示只跑 5 个 Epoch，正式跑请改为 30
    for epoch in range(5): 
        total_loss = 0
        for batch in train_loader:
            pixel_values, input_ids, labels = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            outputs = model(pixel_values, input_ids, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    # 保存Loss图
    plt.plot(loss_history)
    plt.title("Training Loss Trend")
    plt.savefig("final_training_loss.png")

    # 4. 评估 (生成Results表格数据)
    results_df = evaluate_model(model, test_data, tokenizer, processor, device)
    
    # 5. 保存结果到文件
    results_df.to_excel("final_report_results.xlsx", index=False)
    print("\nDetailed results saved to 'final_report_results.xlsx'")
    
    # 6. 打印几个例子用于 Qualitative Analysis
    print("\n--- Examples for Report Discussion ---")
    print(results_df.head(5)[['Question', 'True Answer', 'Pred Answer', 'BLEU']])

if __name__ == "__main__":
    run_experiment()