# Slide scripts for presentation
[15s]hi, everyone. Today our presentation topic is A Comparative Study of CNN Baseline Classifier and LLM-Based Model for Medical Visual Question Answering.
## introduction: problems
[32s]It is  easy for patients and clinicians to access medical images now. However, how to effectively interpret these image information to support diagnosis and decision-making has become a key demand.
Visual question answering (VQA) has demonstrated potential in general fields, but when applied to the field of radiology, it faces unique challenges such as clinical reliability and safety.
## ro&&rq
[50s ]Against this backdrop, we have raised two core research questions: First, Which model performs better in the baseline model and the VLM model?
What is the expression quality  and the risk of hallucinationis  of the VLM model in open-ended questions.
To address these questions, we have set clear research goals: to evaluate the results of closed-end questions through overall accuracy, macro F1;
Meanwhile,  to evaluate the expression quality of open-ended questions by language similarity indicators such as BLEU and ROUGE-L.
## methods: baseline model
[45s]Next, let's introduce the research methods in detail.
First  is the construction of the baseline model, It's divided into four core modules.
In the data preprocessing stage, After cleaning data, We divided the data  into the training set, validation set and test set at a percentage of 70%, 15% and 15% respectively.
Afterwards, we constructed a vocabulary list based on the question texts in the training set. Meanwhile, we built an answer mapping table based on the answers in the training set, converting the text answers into classification labels.

[43s]During the data Transformation  and loading process, images are resized and standardized, while text is processed with a maximum sequence length of 25 as the standard, and then mapped to the corresponding ID. Eventually, all are converted into  tensors.
Subsequently, a batch iterative data stream is constructed through DataLoader. Among them, the training set is shuffled  in order to prevent overfitting, while the validation set and test set remain in order for easy result tracking.

[37s] The baseline model adopts the MedVQA_ResNet_LSTM dual-stream fusion architecture. ResNet50 is used to extract the advanced visual features of the image, and Embedding+LSTM is used to extract the semantic features of the text. Then, after concatenating the two types of features, MLP is used for nonlinear transformation, and finally we have the classification result.

[45s]The training process is based on the PyTorch framework, adopts a training-validation iterative strategy, and also introduces an early stop mechanism.
stage1 uses the training set to train the model, while stage2 uses the validation set to verify the model's accuracy and F1. At the same time, if there is no improvement in the validation loss for N consecutive cycles, the training is stopped to avoid overfitting.
The batch size of the model is 16, and LR is dynamically adjusted using ReduceLROnPlateau.

## methods: generative model
## 
## results: baseline model
[90s]The following are the core results of this research.
First, let's look at the baseline model. Its training accuracy rate reaches 78%, and the training loss is as low as 0.66, indicating that the model has basic fitting ability on the training data and the configuration is reasonable.
However, the accuracy rate of the validation set is only 44%, and test set is as low as 33%. The gap between the training and test accuracy rates is as high as 44%, indicating a serious overfitting phenomenon.
The F1 score, precision rate, recall rate are also generally low, at 13.64%, 14.54% and 13.27% respectively, reflecting the significant impact of category imbalance on prediction quality.
From the perspective of problem types, the model has the highest accuracy rate on closed-end problems, reaching 48%, but the comprehensive evaluation index is still at a low level, indicating that it only performs slightly better on simple judgment problems, and its overall generalization ability is insufficient.

## results: generative model

## comparison


## conclusion









⚡ Training Characteristics
Parameter Efficiency: Only the mapping network is trained (~1% of parameters)
Loss Calculation: Only text tokens contribute to loss; visual prefixes are masked
Objective: Causal language modeling, learning cross-modal alignment
Advantages: Reduces computational complexity, prevents overfitting on small datasets


💡 Summary of Evaluation Principles
Task Adaptability: Different evaluation metrics for different question types

Medical Specificity: Considers clinical validity and semantic accuracy

Benchmark Consistency: Follows Med-VQA domain standard practices

Comprehensiveness: Multi-dimensional assessment of model capabilities

# Generative Model Experimental Results Analysis
Page 1: Training Performance and Overall Results
📉 Training Loss Curve Characteristics
Rapid Initial Decline: Sharp drop from Epoch 0 to Epoch 1

Cause Analysis:

Both CLIP encoder and GPT-2 model remain frozen

Only requires learning simple visual-text alignment mapping

Model quickly converges to a reasonable solution

Stable Phase: Entering plateau around Epoch 5

Loss stabilizes around 0.3 with minimal fluctuation

Learning rate (1e-4) and AdamW optimizer appropriately set

📊 Overall Performance Evaluation Results
Question Type	Accuracy	BLEU-1	BERTScore-F1
Closed-Ended Questions	63.03%	–	–
Open-Ended Questions	–	0.084	0.911
Overall	36.00%	–	–
🔍 Key Findings
Stable Closed Question Performance: 63% accuracy indicates reliable abnormality detection

Open Question Evaluation Divergence:

Extremely Low Surface Metrics: Exact Match 5.7%, BLEU-1 only 0.084

High Semantic Score: BERTScore-F1 reaches 0.911

Core Contradiction: Severe separation between surface form and semantic content

Page 2: Difference Analysis and Model Insights
⚙️ Experimental Differences and Their Impact
1. Prefix Configuration Differences
Prefix Length: lx=10, lq/la=128 (fixed long prefixes)

Comparison: Previous work used shorter or dynamic prefixes

Impact: Long prefixes dilute visual information injection, particularly affecting open-ended answer generation

2. Language Model Limitations
Model Selection: GPT2-base (fewer parameters)

Comparison: Previous work used GPT2-XL or BioMedLM

Impact:

Limits precise word form generation ability

Insufficient lexical expression precision leads to low BLEU

Semantic content remains intact

3. Training Strategy Impact
Freezing Strategy: Language model completely frozen

Dataset Scale: Only VQA-RAD (small scale)

Optimizer: Lower learning rate, no warmup

Combined Impact: Constrains surface form accuracy while preserving semantic correctness

💡 Core Insights and Evaluation Reflections
🔄 Evaluation Metric Sensitivity Analysis
Metric Type	Sensitivity	Medical VQA Applicability
BLEU	Highly sensitive to word-level differences and short answer length	May underestimate model's clinical reasoning ability
BERTScore	Captures semantic similarity, considers synonyms	More comprehensively reflects clinical relevance
Accuracy	Directly effective for closed questions	Cannot assess open question quality
📈 Performance Feature Summary
High BERTScore-F1: Model captures clinically relevant semantics even with different word forms

Low Surface Metrics: Caused by prefix length, model capacity, freezing strategy, dataset scale

Closed Question Robustness: Binary decision tasks not significantly affected



## 性能差距与表达质量
第1页：RQ1 - 基线模型与生成式模型的性能差距
📊 性能对比结果
模型类型	测试准确率	相对提升	关键特征
基线模型 (ResNet-LSTM)	约33%	-	接近二元分类随机猜测
生成式VLM (CLIP+GPT-2)	63.03%	+30个百分点	诊断正确性显著提升
🔍 性能差距原因分析
1. 表征学习差异
基线模型问题:

严重过拟合现象

训练准确率高但测试性能大幅下降

从小数据集从头训练，泛化能力有限

生成式模型优势:

使用冻结的大规模预训练骨干网络

CLIP（视觉编码）+ GPT-2（语言生成）

视觉-语义对齐提供强归纳偏置

2. 低资源条件下的泛化能力
数据集限制: VQA-RAD规模较小

预训练优势: 即使不更新骨干网络参数，预训练知识仍可迁移

对齐学习: 映射网络学习有效的跨模态对齐，提升下游任务表现

💡 核心结论
生成式方法显著优越: 在封闭式医学问答任务上表现更佳

预训练是关键: 大规模预训练模型在小数据集上仍能保持良好泛化

架构设计影响: 冻结骨干网络+轻量映射网络的组合策略有效

第2页：RQ2 - 表达质量与幻觉风险分析
📈 评估指标的"悖论"现象
指标表现对比
评估维度	指标表现	解释
词法严格指标	表现差	精确匹配和BLEU分数极低
语义相似度指标	表现优	BERTScore高达0.911
🧠 错误案例分析
典型失败案例（表10）
真实答案	模型预测	指标结果	临床解释
the brain	brain	BLEU-4=0.065	临床含义完全相同
right side	right	精确匹配=0	缺失修饰语，空间概念正确
diffusion weighted MRI	MRI	精确匹配=0	正确识别模态，缺失具体子类型
⚠️ 幻觉风险分类
1. 软幻觉
特征: 改述或替换临床相关术语

示例: "brain" vs "the brain"

风险等级: 较低，不影响临床决策

原因: 语言模型的流畅生成特性

2. 硬幻觉
特征: 编造不存在疾病或发现

示例: 预测不存在病变

风险等级: 较高，可能导致误诊

本研究观察: 较少出现，得益于冻结GPT-2的连贯性

🔬 安全性分析
语言模型冻结的优势
输出连贯性: 减少无意义输出

术语一致性: 保持医学术语准确性

风险控制: 降低硬幻觉发生概率

临床适用性评估
语义正确性优先: 即使词法不完全匹配，临床含义正确即可接受

修饰语敏感性: 缺失非关键修饰语对诊断影响有限

子类型特异性: 识别主模态比精确子类型更重要