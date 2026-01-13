import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, CLIPProcessor, CLIPModel
from PIL import Image

# ==========================================
# 1. 核心模型架构：视觉映射网络 (The Projector)
# ==========================================
class MedVQAModel(nn.Module):
    def __init__(self, prefix_length=10, clip_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.prefix_length = prefix_length
        
        # 加载预训练 CLIP 和 GPT-2
        self.clip_model = CLIPModel.from_pretrained(clip_model_name)
        self.gpt2 = GPT2LMHeadModel.from_pretrained("gpt2")
        
        # 冻结所有主干参数，不参与梯度更新
        for param in self.clip_model.parameters(): param.requires_grad = False
        for param in self.gpt2.parameters(): param.requires_grad = False
        
        # 唯一需要学习的部分：MLP 映射网络
        # 将 CLIP 的视觉特征 (512维) 映射为 GPT-2 能理解的虚拟 Token Embedding (768维 * 长度k)
        clip_dim = self.clip_model.config.projection_dim
        gpt_dim = self.gpt2.config.n_embd
        self.mapping_network = nn.Sequential(
            nn.Linear(clip_dim, gpt_dim),
            nn.ReLU(),
            nn.Linear(gpt_dim, prefix_length * gpt_dim)
        )

    def forward(self, pixel_values, input_ids, labels=None):
      # 1. 提取视觉特征并映射为前缀
      with torch.no_grad():
          vision_outputs = self.clip_model.get_image_features(pixel_values=pixel_values)
      
      visual_prefixes = self.mapping_network(vision_outputs)
      visual_prefixes = visual_prefixes.view(-1, self.prefix_length, self.gpt2.config.n_embd)
      
      # 2. 获取文本 Embedding
      text_embeds = self.gpt2.transformer.wte(input_ids)
      
      # 3. 拼接输入: [Prefix + Text]
      full_embeds = torch.cat((visual_prefixes, text_embeds), dim=1)
      
      # 4. 关键：对齐 Labels
      if labels is not None:
          # 创建一个填充了 -100 的前缀标签
          # -100 是 PyTorch CrossEntropyLoss 的默认忽略索引
          batch_size = labels.shape[0]
          prefix_labels = torch.full((batch_size, self.prefix_length), -100).to(labels.device)
          
          # 将前缀标签与原始文本标签拼接
          # 新维度: [Batch, Prefix_Length + Text_Length]
          full_labels = torch.cat((prefix_labels, labels), dim=1)
      else:
          full_labels = None

      # 5. 传入模型
      outputs = self.gpt2(inputs_embeds=full_embeds, labels=full_labels)
      return outputs

# ==========================================
# 2. 数据处理：将医疗问答转为生成序列
# ==========================================
class MedVQADataset(Dataset):
    def __init__(self, data_list, tokenizer, processor, prefix_length):
        self.data = data_list # 假设格式: [{'image_path': '...', 'question': '...', 'answer': '...'}]
        self.tokenizer = tokenizer
        self.processor = processor
        self.prefix_length = prefix_length

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image_path']).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze()
        
        # 构造 GPT-2 输入序列: Question: [Q] Answer: [A]
        full_text = f"Question: {item['question']} Answer: {item['answer']} <|endoftext|>"
        tokens = self.tokenizer(full_text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        
        input_ids = tokens.input_ids.squeeze()
        
        # 构造 Labels (用于计算 Loss)
        # 前缀部分不计算 Loss，所以填充 -100
        labels = input_ids.clone()
        # 注意：这里的 label 处理需考虑 prefix_length 的偏移，简化版直接设为 input_ids
        # 在真实场景中，Question 部分也可以设为 -100，只让模型学习生成 Answer
        
        return pixel_values, input_ids, labels

# ==========================================
# 3. 训练循环 (Training Loop)
# ==========================================
def run_experiment():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 初始化组件
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = MedVQAModel().to(device)
    
    # 模拟数据 (实际实验时替换为 VQA-RAD 加载逻辑)
    dummy_data = [{'image_path': './VQA_RAD Image Folder/synpic47974.jpg', "question": "Are these masses encompassing the aorta?", 'answer': 'No'}] * 100
    dataset = MedVQADataset(dummy_data, tokenizer, processor, prefix_length=10)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 优化器：只更新映射层参数
    optimizer = torch.optim.AdamW(model.mapping_network.parameters(), lr=1e-4)
    
    model.train()
    for epoch in range(5):
        for batch in dataloader:
            pixel_values, input_ids, labels = [b.to(device) for b in batch]
            
            outputs = model(pixel_values, input_ids, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_experiment()