import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import pandas as pd
import json
import collections
import ssl
from torchvision import transforms, models
import matplotlib.pyplot as plt
from torchvision.models import resnet50, ResNet50_Weights

# 1. 环境配置
if not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
    ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- 第一步：数据转换与划分 ---
# 将原始 JSON 转换为 CSV 并划分训练/验证集
def convert_and_split_data(json_path, output_csv):
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    parsed_list = []
    for item in raw_data:
        parsed_list.append({
            'image_id': item.get('image_name', ''),
            'question': item.get('question', ''),
            'answer': str(item.get('answer', '')),
            'answer_type': item.get('answer_type', '')
        })
    df = pd.DataFrame(parsed_list).dropna(subset=['image_id', 'question', 'answer'])
    df.to_csv(output_csv, index=False)

    # 划分 80% 训练和 20% 验证
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)
    train_df.to_csv('train_split.csv', index=False)
    val_df.to_csv('val_split.csv', index=False)
    print(f'training data size：{train_df.size}, testing data size:{val_df.size}')
    return df


# 执行转换
all_data_df = convert_and_split_data('VQA_RAD Dataset Public.json', 'vqa_rad_train.csv')


# --- 第二步：构建完整词汇表 (这是关键修正点) ---
def build_vocab(csv_file, min_freq=1):
    df = pd.read_csv(csv_file)
    counter = collections.Counter()
    for q in df['question']:
        counter.update(str(q).lower().split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    with open('vocab.json', 'w') as f:
        json.dump(vocab, f)
    return vocab


# 在初始化 Dataset 之前先生成真实词汇表
vocab = build_vocab('vqa_rad_train.csv')
print(f"实测词汇表大小: {len(vocab)}")


# --- 第三步：定义 Dataset 类 ---
class MedVQADataset(Dataset):
    def __init__(self, csv_file, img_dir, tokenizer, ans_to_id, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.ans_to_id = ans_to_id  # 使用外部传入的统一映射

    def __len__(self):
        return len(self.data)

    def text_to_sequence(self, text, max_length=20):
        tokens = str(text).lower().split()
        # 使用生成的完整词汇表进行索引映射
        sequence = [self.tokenizer.get(token, 1) for token in tokens]
        if len(sequence) < max_length:
            sequence += [0] * (max_length - len(sequence))
        return torch.tensor(sequence[:max_length])

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        try:
            image = Image.open(os.path.join(self.img_dir, row['image_id'])).convert('RGB')
            if self.transform: image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)  # 容错处理

        question = self.text_to_sequence(row['question'])
        label = torch.tensor(self.ans_to_id.get(row['answer'], 0), dtype=torch.long)
        q_type = row['answer_type']  # 假设 CSV 中列名为 'answer_type'

        return image, question, label, q_type


# --- 第四步：初始化 DataLoaders ---
# 建立统一的答案映射
global_ans_to_id = {ans: i for i, ans in enumerate(all_data_df['answer'].unique())}
num_classes = len(global_ans_to_id)

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = MedVQADataset('train_split.csv', 'VQA_RAD Image Folder/', vocab, global_ans_to_id,
                              transform=data_transforms)
val_dataset = MedVQADataset('val_split.csv', 'VQA_RAD Image Folder/', vocab, global_ans_to_id,
                            transform=data_transforms)
print("--- 正在准备数据加载器 ---")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
print(f"--- 数据加载器已就绪，共有 {len(train_loader)} 个 training batches && ---{len(val_dataset)}个 testing  batches")


# --- 第五步：模型初始化与训练 ---
class MedVQA_ResNet_MLP(nn.Module):
    def __init__(self, num_classes, vocab_size):
        super(MedVQA_ResNet_MLP, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.vis_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.embedding = nn.Embedding(vocab_size, 512)
        self.text_fc = nn.Linear(512, 512)
        self.mlp = nn.Sequential(
            nn.Linear(self.vis_dim + 512, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, images, questions):
        img_feats = self.resnet(images)
        txt_feats = self.embedding(questions).mean(dim=1)
        txt_feats = torch.relu(self.text_fc(txt_feats))
        combined = torch.cat((img_feats, txt_feats), dim=1)
        return self.mlp(combined)


class MedVQA_ResNet_LSTM(nn.Module):
    def __init__(self, num_classes, vocab_size, embed_dim=512, hidden_dim=512):
        super(MedVQA_ResNet_LSTM, self).__init__()

        # 1. Image Encoder: ResNet50
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.vis_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # 移除分类层，输出 2048 维特征

        # 2. Text Encoder: LSTM
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False  # 如果需要更强能力，可设为True，但特征维度需*2
        )

        # 3. Fusion & Classifier
        self.mlp = nn.Sequential(
            nn.Linear(self.vis_dim + hidden_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )

    def forward(self, images, questions):
        # 提取图像特征 [batch_size, 2048]
        img_feats = self.resnet(images)

        # 提取文本特征
        # questions shape: [batch_size, seq_len]
        embedded = self.embedding(questions)  # [batch_size, seq_len, embed_dim]

        # LSTM 输出说明:
        # output: 包含序列中每个时间步的特征
        # (h_n, c_n): h_n 是最后一个隐藏状态，代表整个句子的压缩语义
        _, (h_n, _) = self.lstm(embedded)

        # 取最后一层 LSTM 的输出 [batch_size, hidden_dim]
        txt_feats = h_n[-1]

        # 特征融合
        combined = torch.cat((img_feats, txt_feats), dim=1)

        # 输出分类概率
        return self.mlp(combined)


print("--- 权重下载完成，开始初始化模型 ---")
# model = MedVQA_ResNet_MLP(num_classes=num_classes, vocab_size=len(vocab)).to(device)
model = MedVQA_ResNet_LSTM(num_classes=num_classes, vocab_size=len(vocab)).to(device)
print("--- 模型初始化成功 ---")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()


# --- 升级后的训练与验证函数 ---
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=30):
    # 增加 train_acc 记录
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(epochs):
        # --- 训练阶段 ---
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for imgs, ques, lbls, _ in train_loader:  # 注意：这里解包增加了 q_type 的占位符 _
            imgs, ques, lbls = imgs.to(device), ques.to(device), lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs, ques)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += lbls.size(0)
            correct_train += (predicted == lbls).sum().item()

        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # --- 验证阶段（包含分类统计） ---
        model.eval()
        correct_val = 0
        total_val = 0

        # 初始化分类统计字典
        type_stats = {'CLOSED': {'correct': 0, 'total': 0},
                      'OPEN': {'correct': 0, 'total': 0}}

        with torch.no_grad():
            for imgs, ques, lbls, q_types in val_loader:
                imgs, ques, lbls = imgs.to(device), ques.to(device), lbls.to(device)
                outputs = model(imgs, ques)
                _, predicted = torch.max(outputs.data, 1)

                total_val += lbls.size(0)
                correct_val += (predicted == lbls).sum().item()

                # 逐个样本进行类型归类
                for i in range(len(lbls)):
                    q_type = q_types[i]
                    is_correct = (predicted[i] == lbls[i]).item()
                    if q_type in type_stats:
                        type_stats[q_type]['total'] += 1
                        if is_correct:
                            type_stats[q_type]['correct'] += 1

        val_acc = 100 * correct_val / total_val
        history['train_loss'].append(epoch_loss)
        history['val_acc'].append(val_acc)
        # 将 train_acc 存入 history
        history['train_acc'].append(train_acc)
        last_type_stats = type_stats  # 记录最后一轮统计

        # 打印实时进度
        print(f"Epoch [{epoch + 1}/{epochs}]")
        print(f"  Train Loss: {epoch_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")

        # 仅在最后一个 Epoch 打印详细的类型报告
        if epoch == epochs - 1:
            print("\n>>> Final Detailed Accuracy Analysis (Last Epoch):")
            for qt, data in type_stats.items():
                if data['total'] > 0:
                    type_acc = 100 * data['correct'] / data['total']
                    print(f"    - Type: {qt:8} | Total: {data['total']:4} | Acc: {type_acc:.2f}%")
        print("-" * 30)

    return history, last_type_stats


# --- 重新调用训练 ---
# 注意：确保你的 MedVQADataset 的 __getitem__ 返回了 4 个值 (img, ques, label, q_type)
train_history, final_type_stats = train_and_validate(model, train_loader, val_loader, criterion, optimizer, epochs=30)

# --- 修改后的绘图部分 ---
# plt.figure(figsize=(12, 5))

# 左图：Loss 曲线
# plt.subplot(1, 2, 1)
# plt.plot(train_history['train_loss'], label='Train Loss', color='blue')
# plt.title('Loss Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# # 右图：Accuracy 曲线 (包含 Train 和 Val)
# plt.subplot(1, 2, 2)
# plt.plot(train_history['train_acc'], label='Train Accuracy', color='green', linestyle='--')
# plt.plot(train_history['val_acc'], label='Val Accuracy', color='red')
# plt.title('Accuracy Curve')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy (%)')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()
# --- 1. 创建保存结果的文件夹 ---
output_dir = 'baseline_vqa_results_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# --- 2. 词汇表频率可视化 (基于 Step 2 的 Counter) ---
def plot_vocab_frequency(csv_file, save_path):
    df = pd.read_csv(csv_file)
    counter = collections.Counter()
    for q in df['question']:
        counter.update(str(q).lower().split())

    # 获取前 20 个最常见的单词
    most_common = counter.most_common(20)
    words = [item[0] for item in most_common]
    counts = [item[1] for item in most_common]

    plt.figure(figsize=(12, 6))
    plt.bar(words, counts, color='skyblue')
    plt.title('Top 20 Word Frequencies in Questions')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'vocab_frequency.png'), dpi=300)
    plt.close()
    print(f"--- 词汇频率图已保存至: {save_path}/vocab_frequency.png ---")


# --- 3. 分类准确率可视化 (基于最后的 type_stats) ---
def plot_type_accuracy(type_stats, save_path):
    types = []
    accs = []
    totals = []

    for qt, data in type_stats.items():
        if data['total'] > 0:
            types.append(qt)
            accs.append(100 * data['correct'] / data['total'])
            totals.append(data['total'])

    plt.figure(figsize=(8, 6))
    bars = plt.bar(types, accs, color=['salmon', 'lightgreen'])
    plt.ylim(0, 100)
    plt.title('Accuracy by Question Type (Last Epoch)')
    plt.ylabel('Accuracy (%)')

    # 在柱状图上方标注具体百分比和样本数
    for i, bar in enumerate(bars):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1,
                 f'{yval:.2f}%\n(n={totals[i]})', ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'type_accuracy_comparison.png'), dpi=300)
    plt.close()
    print(f"--- 分类准确率对比图已保存至: {save_path}/type_accuracy_comparison.png ---")


# --- 4. 修改训练函数以返回 type_stats 并执行绘图 ---

# 执行词汇表统计图
plot_vocab_frequency('vqa_rad_train.csv', output_dir)

# 绘制 Loss 和 Accuracy 趋势图并保存
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(train_history['train_loss'], label='Train Loss', lw=2)
plt.title('Training Loss Trend')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_history['train_acc'], label='Train Accuracy', color='green', linestyle='--')
plt.plot(train_history['val_acc'], label='Val Accuracy', color='red', lw=2)
plt.title('Accuracy Trend (Train vs Val)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=300)
plt.show()

# 执行分类准确率绘图
plot_type_accuracy(final_type_stats, output_dir)
