import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, CLIPProcessor, CLIPModel
from PIL import Image
import os
import matplotlib.pyplot as plt  # 新增：用于绘图
# ==========================================
# 1. 核心模型架构：视觉映射网络
# ==========================================
class MedVQAModel(nn.Module):
    def __init__(self, prefix_length=10, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.prefix_length = prefix_length
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # 冻结主干参数
        for param in self.clip_model.parameters(): param.requires_grad = False
        for param in self.gpt2.parameters(): param.requires_grad = False
        
        # 可训练层：将 512 维 CLIP 特征映射为 10个 768 维的词向量
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
        
        # [Batch, Prefix_Len, GPT_Dim]
        visual_prefixes = self.mapping_network(vision_outputs).view(
            -1, self.prefix_length, self.gpt2.config.n_embd
        )
        
        # [Batch, Seq_Len, GPT_Dim]
        text_embeds = self.gpt2.transformer.wte(input_ids)
        
        # 拼接后的输入维度: [Batch, Prefix_Len + Seq_Len, GPT_Dim]
        full_embeds = torch.cat((visual_prefixes, text_embeds), dim=1)
        
        if labels is not None:
            # 前缀位置不参与 Loss 计算 (设为 -100)
            prefix_labels = torch.full((labels.size(0), self.prefix_length), -100).to(labels.device)
            full_labels = torch.cat((prefix_labels, labels), dim=1)
            return self.gpt2(inputs_embeds=full_embeds, labels=full_labels)
        
        return self.gpt2(inputs_embeds=full_embeds)

# ==========================================
# 2. 数据处理：模拟与加载
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
        
        # 训练时拼接 Q 和 A
        full_text = f"Question: {item['question']} Answer: {item['answer']} <|endoftext|>"
        tokens = self.tokenizer(full_text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        
        input_ids = tokens.input_ids.squeeze()
        # 简化处理：训练时尝试让模型预测整个文本序列（除了前缀部分）
        return pixel_values, input_ids, input_ids 

# ==========================================
# 3. 绘图函数
# ==========================================
def plot_loss(loss_list):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_list, label='Training Loss')
    plt.title('MedVQA Training Loss Over Steps')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('train_loss_plot.png')
    print("\nLoss plot saved as 'train_loss_plot.png'")
    plt.show()
    
# ==========================================
# 4. 执行：训练 -> 保存 -> 推理
# ==========================================
def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device detect: {device}")

    # --- 环境准备 ---
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = MedVQAModel().to(device)

    # # 自动生成一张模拟图用于测试
    # if not os.path.exists("test_sample.jpg"):
    #     Image.new('RGB', (224, 224), color='gray').save("test_sample.jpg")

    # --- 训练阶段 ---
    dummy_data = [{'image_path': './VQA_RAD Image Folder/synpic47974.jpg', "question": "Are these masses encompassing the aorta?", 'answer': 'No'}] * 100
    dataset = MedVQADataset(dummy_data, tokenizer, processor, prefix_length=10)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.mapping_network.parameters(), lr=1e-4)
    
    loss_history = []  # 用于记录 Loss
    print("\nStarting Training...")
    model.train()
    for epoch in range(10): # 演示训练3轮
        for i, (p_val, i_ids, labs) in enumerate(dataloader):
            p_val, i_ids, labs = p_val.to(device), i_ids.to(device), labs.to(device)
            optimizer.zero_grad()
            outputs = model(p_val, i_ids, labels=labs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            
            if i % 2 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
    # 绘制并展示 Loss 曲线
    plot_loss(loss_history)
    # --- 权重保存 ---
    torch.save(model.mapping_network.state_dict(), "medvqa_projector.pt")
    print("\nWeight saved: medvqa_projector.pt")

    # --- 推理验证 ---
    print("\nStarting Inference...")
    model.eval()
    test_q = "Are these masses encompassing the aorta?"
    test_prompt = f"Question: {test_q} Answer:"
    
    # 编码问题
    prompt_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
    img = Image.open("VQA_RAD Image Folder/synpic676.jpg").convert("RGB")
    p_val = processor(images=img, return_tensors="pt").pixel_values.to(device)

    

    with torch.no_grad():
        # 获取图像前缀嵌入
        feats = model.clip_model.get_image_features(pixel_values=p_val)
        prefix_embeds = model.mapping_network(feats).view(1, 10, -1)
        
        # 获取文本嵌入
        text_embeds = model.gpt2.transformer.wte(prompt_ids)
        
        # 合并作为 generate 的初始输入
        inputs_embeds = torch.cat((prefix_embeds, text_embeds), dim=1)
        
        # 使用 generate 生成后续 Token
        outputs = model.gpt2.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=10,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        result_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Input Question: {test_q}")
        print(f"Model Generated Answer: {result_text}")

if __name__ == "__main__":
    run_experiment()