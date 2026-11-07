# CIFAR-10 图像分类：CNN、残差网络、注意力与迁移学习

当代人工智能作业三：图像分类模型的设计、训练、评估与可解释性分析。

仓库：https://github.com/shixuhui-ops/cifar10-image-classification

## 1. 项目内容

本项目完成以下实验：

- 基础 CNN 与残差网络的架构对比；

- SE 通道注意力消融；

- 数据增强消融；

- Label Smoothing 损失函数消融；

- ImageNet 预训练 ResNet-18 的部分微调；

- 学习率调度、权重衰减、Dropout、BatchNorm 和早停；

- 训练曲线、混淆矩阵、总体及每类分类指标；

- 高置信度错误样本分析与 Grad-CAM 可视化；

- CPU、GPU 不同软件环境下的复现记录。

这是对已有方法的实现、组合和实验分析，不声称提出全新的网络算法。

## 2. 数据与划分

使用 CIFAR-10：

- 图像大小：32×32，RGB；

- 类别数：10；

- 官方训练集：50,000 张；

- 官方测试集：10,000 张。

从官方训练集中随机划分：

- 训练集：45,000 张；

- 验证集：5,000 张；

- 测试集：保留官方 10,000 张。

划分随机种子为 42，采用随机划分而非分层划分。

训练增强包括：

- RandomCrop(32, padding=4)；

- RandomHorizontalFlip；

- ColorJitter，亮度、对比度和饱和度范围参数均为 0.2。

验证集和测试集不使用随机增强。

CIFAR-10 标准化参数：

    mean = (0.4914, 0.4822, 0.4465)

    std  = (0.2023, 0.1994, 0.2010)

迁移学习模型在内部还原标准化后的像素，将输入放大为 96×96，

再使用 ImageNet 均值和标准差进行标准化。

数据由 torchvision 自动下载。数据集不上传至本仓库。

## 3. 实验结果

下表中准确率以百分比表示，Macro F1 范围为 0～1。

所有实验的随机种子均为 42。

| 实验目录 | 最佳验证准确率 | 测试准确率 | Macro F1 |
|---|---:|---:|---:|
| baseline，20 轮初步实验 | 88.22% | 87.24% | 0.8711 |
| baseline_matched，30 轮正式对照 | 90.00% | 88.55% | 0.8846 |
| resnet_se | 92.64% | 92.14% | 0.9214 |
| resnet_se_noaug | 88.44% | 87.90% | 0.8788 |
| resnet_se_no_smoothing | 92.20% | 91.73% | 0.9173 |
| resnet_no_se | 93.46% | 92.32% | 0.9232 |
| resnet18_transfer | 93.24% | 92.82% | 0.9282 |
| gpu_resnet_seed42，独立环境复现 | 92.98% | 91.92% | 0.9190 |

前七组为 CPU 实验，最后一组为 GPU 环境复现实验。

### 模型选择

按正式 CPU 实验的最佳验证准确率选择主模型：

    resnet_no_se

    验证准确率：93.46%

    测试准确率：92.32%

    参数量：2,777,674

不能因为迁移学习模型测试准确率更高，就跳过验证集选择原则。

### 主要观察

- 相同最大轮数及主要训练设置下，残差网络相对基础 CNN，

  测试准确率提高 3.77 个百分点。

- SE 模型关闭数据增强后，测试准确率下降 4.24 个百分点。

- SE 模型使用标签平滑，相对普通交叉熵提高 0.41 个百分点。

- SE 模型相对无 SE 模型低 0.18 个百分点，本次实验未观察到收益。

- 预训练模型使用额外 ImageNet 数据、更大输入和不同训练预算，

  其结果不是严格控制变量的架构对比。

这些观察来自单随机种子实验，不构成统计显著性结论。

## 4. 软件与硬件环境

| 环境 | Python | PyTorch | torchvision / CUDA |
|---|---|---|---|
| CPU 原始实验 | 3.11.18 | 2.15.0+cpu | 详见环境记录 |
| GPU 复现实验 | 3.11.18 | 2.7.1+cu128 | torchvision 0.22.1+cu128，CUDA 12.8 |

GPU：NVIDIA GeForce RTX 4060 Laptop GPU，约 8 GB 显存。

完整版本记录：

    reports/environment_cpu.json

    reports/environment_gpu.json

    requirements-cpu.txt

    requirements-gpu.txt

CPU 与 GPU 实验同时存在计算设备和 PyTorch 版本差异，

不能把准确率差异直接归因于 GPU。

固定随机种子可以减少随机性，但不保证跨版本、跨设备逐位一致。

## 5. 安装

以下命令适用于 Windows PowerShell 和 Conda。

先克隆并进入项目：

    git clone https://github.com/shixuhui-ops/cifar10-image-classification.git

    cd cifar10-image-classification

### GPU 环境

需要支持 CUDA 的 NVIDIA 显卡和兼容驱动。

    conda create -n cifar10-gpu python=3.11 -y

    conda activate cifar10-gpu

    python -m pip install -r requirements-gpu.txt --extra-index-url https://download.pytorch.org/whl/cu128

    python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

PyTorch CUDA wheel 包含所需运行时，本项目不编译自定义 CUDA 算子，

通常不需要另外安装 CUDA Toolkit。

### CPU 原始环境

    conda create -n cifar10-ai python=3.11 -y

    conda activate cifar10-ai

    python -m pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu

如无法取得原始版本，应明确记录替代版本，将结果视为新的复现实验，

不要宣称与原始环境完全一致。

## 6. 训练

请在项目根目录运行脚本。

以下正式实验命令应在新克隆目录运行，

不要覆盖已经保存的原始实验记录。

首次训练会自动下载 CIFAR-10。

### CPU 原始实验

    conda activate cifar10-ai

20 轮基础 CNN 初步实验：

    python .\src\train.py --model baseline --epochs 20 --batch-size 128 --lr 0.001 --patience 6 --seed 42 --output-dir outputs\baseline

30 轮基础 CNN 正式对照：

    python .\src\train.py --model baseline --epochs 30 --batch-size 128 --lr 0.001 --patience 8 --seed 42 --output-dir outputs\baseline_matched

SE 残差网络：

    python .\src\train.py --model resnet_se --epochs 30 --batch-size 128 --lr 0.001 --patience 8 --seed 42 --output-dir outputs\resnet_se

关闭数据增强：

    python .\src\train.py --model resnet_se --epochs 30 --batch-size 128 --lr 0.001 --patience 8 --seed 42 --no-augmentation --output-dir outputs\resnet_se_noaug

关闭标签平滑：

    python .\src\train.py --model resnet_se --epochs 30 --batch-size 128 --lr 0.001 --patience 8 --seed 42 --label-smoothing 0 --output-dir outputs\resnet_se_no_smoothing

无 SE 残差网络：

    python .\src\train.py --model resnet --epochs 30 --batch-size 128 --lr 0.001 --patience 8 --seed 42 --output-dir outputs\resnet_no_se

预训练 ResNet-18 部分微调：

    python .\src\train_transfer.py --epochs 15 --batch-size 64 --patience 5 --seed 42 --output-dir outputs\resnet18_transfer

### GPU 独立复现

    conda activate cifar10-gpu

    python .\src\train.py --model resnet --epochs 30 --batch-size 128 --lr 0.001 --patience 8 --seed 42 --output-dir outputs\gpu_resnet_seed42

脚本自动选择可用的 CUDA 设备，否则使用 CPU。

### 训练设置

从零训练的正式对照实验使用：

- AdamW；

- 初始学习率 0.001；

- weight decay = 0.0005；

- batch size = 128；

- 最多 30 轮；

- CosineAnnealingLR；

- 验证准确率连续 8 轮不提升时早停；

- 默认标签平滑系数 0.1。

迁移学习使用：

- ImageNet 预训练 ResNet-18；

- 冻结前面的特征提取层及其 BatchNorm 统计；

- 仅训练 layer4 和分类层；

- layer4 学习率 0.0001；

- 分类层学习率 0.001；

- batch size = 64；

- 最多 15 轮，早停耐心值为 5；

- 总参数量 11,181,642，可训练参数量 8,398,858。

每次训练保存：

    config.json

    history.json

    best_model.pt

    last_model.pt

    test_metrics.json

最优权重按验证准确率保存，最终加载最优权重进行测试。

训练入口 train.py 不直接提供恢复参数，可使用独立脚本
src/resume_train.py 从 last_model.pt 继续训练，使用方法见第 15.3 节。

train.py 新生成的检查点会保存 Python、NumPy、PyTorch CPU 和
可用 CUDA 设备的随机数状态，resume_train.py 会在继续训练前恢复这些状态。

已在当前 GPU 环境下通过两轮、小批次的一致性测试：
连续训练与中断后恢复训练的模型参数、优化器、调度器、
随机状态及训练历史（不含耗时）完全一致。
该结果不保证跨设备、跨软件版本或其他训练设置下逐位一致。

旧检查点若不包含随机状态，仍可按原有方式恢复，
但不能保证与不中断训练得到完全一致的结果。

## 7. 评估与可视化

### CPU 实验

先在对应环境完成训练，再执行：

    python .\src\evaluate.py --runs outputs\baseline outputs\baseline_matched outputs\resnet_se outputs\resnet_se_noaug outputs\resnet_se_no_smoothing outputs\resnet_no_se outputs\resnet18_transfer

### GPU 实验

    python .\src\evaluate.py --runs outputs\gpu_resnet_seed42 --comparison-path outputs\gpu_model_comparison.csv

各实验目录生成：

    figures/training_curves.png

    figures/confusion_matrix.png

    figures/error_samples.png

    classification_report.json

    per_class_metrics.csv

    confusion_matrix.csv

    predictions.npz

    evaluation_summary.json

每类准确率定义为：

    该类正确预测数量 / 该类真实样本数量

对应 CSV 中的 recall_per_class_accuracy，而非 precision。

原始计数混淆矩阵和按真实类别逐行归一化的混淆矩阵均被保存为图像。

测试集每类有 1,000 张图像。

### Grad-CAM

脚本固定分析 CPU 主模型目录 outputs/resnet_no_se：

    python .\src\gradcam.py

输出：

    outputs/resnet_no_se/figures/gradcam.png

    outputs/resnet_no_se/gradcam_samples.csv

每行依次展示：

1. 原始图像；

2. 预测类别 Grad-CAM；

3. 真实类别 Grad-CAM。

正确样本中，预测类别与真实类别相同，两张热力图因此相同。

错误样本选取置信度最高的四个，并非随机样本，

不能代表所有误分类。

每张热力图单独归一化，不能通过颜色强弱直接比较不同类别的贡献大小。

Grad-CAM 是局部解释工具，不构成因果证明。

## 8. 错误分析摘要

主模型共正确分类 9,232 张，错误分类 768 张。

- 汽车类别准确率最高：96.5%；

- 猫类别准确率最低：84.1%；

- 猫被误判为狗：70 张；

- 狗被误判为猫：69 张；

- 两个方向合计 139 张，占全部错误约 18.1%。

Grad-CAM 显示，正确样本的高响应区域大体覆盖目标主体。

部分错误样本也关注了主体，例如狗头被误判为猫，

说明误差不一定全部来自背景干扰。

低分辨率、姿态变化、局部外观相似和背景关联是值得进一步验证的解释，

但仅凭当前热力图不能确认这些因素的因果作用。

## 9. 汇总与一致性检查

完成全部八组训练及评估后：

    python .\src\summarize_results.py

脚本核对测试准确率与混淆矩阵是否一致，并生成：

    reports/experiment_summary.csv

    reports/results_summary.json

仓库保留原始指标和图表，因此仅复核这些汇总文件不需要模型权重。

环境导出：

    python .\src\export_environment.py --name gpu

CPU 环境下则使用 --name cpu。

## 10. 项目结构

    src/

      data.py                 数据下载、变换与划分

      models.py               CNN、残差、SE 与迁移学习模型

      train.py                从零训练入口

      train_transfer.py       预训练模型微调入口

      evaluate.py             分类指标、曲线及错误样本

      gradcam.py              Grad-CAM 可解释性分析

      export_environment.py   环境版本记录

      summarize_results.py    实验汇总与一致性检查

    outputs/                  正式实验指标和图表

    reports/                  报告及实验、环境汇总

    requirements-cpu.txt      CPU 环境依赖版本

    requirements-gpu.txt      GPU 环境依赖版本

数据集、安装包、模型权重和快速检查目录由 .gitignore 排除。

仓库不提供预训练后的项目权重文件；

需要通过对应训练命令生成 best_model.pt 后才能重新评估或生成 Grad-CAM。

## 11. 局限与改进方向

- CPU 原始实验主要使用单随机种子；GPU 补充实验对部分架构、
  注意力、标签平滑及增强策略进行了三随机种子重复实验，
  结果需分别解释。

- 已报告部分对比的配对 t 检验与 95% 置信区间，但每组仅有
  三个随机种子，统计证据仍具有探索性。

- 参数量和 FLOPs 接近的架构对照减少了计算资源差异的影响，
  但网络深度、池化和 Dropout 等仍不同，不能将全部性能差异
  单独归因于残差连接。

- 迁移学习与从零训练的输入尺寸、外部数据和训练预算不同。

- CPU 和 GPU 软件版本不同，不适合作为严格的硬件效果对照。

- 不同标签平滑系数对应不同损失目标，不能直接用测试 loss
  的数值高低判断模型优劣。

- 高置信度错误样本和 Grad-CAM 个例具有选择性；
  热力图不能作为因果证明。

后续可增加随机种子数量，在相同网络结构中仅移除跳跃连接，
进一步验证残差连接的作用，并研究置信度校准和迁移学习的公平对照。

## 12. 参考资料

- CIFAR-10：https://www.cs.toronto.edu/\~kriz/cifar.html

- PyTorch：https://pytorch.org/

- Torchvision ResNet-18：https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html

- He et al. Deep Residual Learning for Image Recognition. CVPR, 2016.

- Hu et al. Squeeze-and-Excitation Networks. CVPR, 2018.

- Selvaraju et al. Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV, 2017.

- Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR, 2016.

## 13. 生成五页 PDF 实验报告

报告生成脚本适用于 Windows，使用系统宋体字体：

    C:/Windows/Fonts/simsun.ttc

先安装本项目 CPU 或 GPU 环境，再安装报告依赖：

    python -m pip install -r requirements-report.txt

### 填写个人报告信息

手动创建 reports/report_info.json，填写以下内容。

姓名和学号不能为空，班级可留空：

    {

      "title": "CIFAR-10 图像分类模型设计与实验分析",

      "author": "",

      "student_id": "",

      "class_name": "",

      "github_url": "https://github.com/shixuhui-ops/cifar10-image-classification"

    }

此配置包含个人信息，已被 .gitignore 排除，不应上传至公开仓库。

### 使用仓库现有证据生成报告

仓库保留实验指标、历史记录及报告专用图片，因此生成 PDF

不需要重新训练，也不需要下载模型权重：

    python .\src\summarize_results.py

    python .\src\build_report.py

生成文件：

    reports/experiment_report.pdf

脚本自动检查：

- PDF 必须为五页；

- 首页包含可点击的 GitHub 仓库链接；

- 正文、表格和图片不越过预设排版边界。

自动检查不能代替人工检查，提交前仍须查看所有页面。

报告 PDF 含个人信息，仅用于课程提交，默认不上传公开仓库。

### 重新生成报告图片

若已经完成原始训练，且本地具有 CIFAR-10 数据及主模型权重，

可以重新生成报告专用图：

    python .\src\prepare_report_figures.py

输出保存在 reports/figures/。

缺少模型权重时，不需要执行此命令，直接使用仓库现有图片即可。

## 14. 新增稳健性与控制变量实验

### 14.1 三随机种子实验

GPU 环境下使用 seed=42、43、44，训练设置保持一致。

| 模型 | 测试准确率均值 | 样本标准差 |
|---|---:|---:|
| 无 SE 残差网络 | 92.07% | 0.16% |
| SE 残差网络 | 91.50% | 0.29% |
| SE 残差网络，无标签平滑 | 91.36% | 0.17% |

SE 模型在三个种子上均低于无 SE 残差网络，平均低 0.57 个百分点。当前设置下未观察到 SE 的稳定收益。

标签平滑相对于无标签平滑的平均提升为 0.13 个百分点，但提升幅度较小，不能据此宣称具有普遍有效性。

完整逐次实验记录：

- `reports/multiseed_runs.csv`
- `outputs/gpu_resnet_seed42`
- `outputs/gpu_resnet_seed43`
- `outputs/gpu_resnet_seed44`
- `outputs/gpu_resnet_se_seed42`
- `outputs/gpu_resnet_se_seed43`
- `outputs/gpu_resnet_se_seed44`
- `outputs/gpu_resnet_se_no_smoothing_seed42`
- `outputs/gpu_resnet_se_no_smoothing_seed43`
- `outputs/gpu_resnet_se_no_smoothing_seed44`

### 14.2 数据增强拆分实验

固定无 SE 残差网络和 seed=42，分别比较单独增强策略。

| 增强策略 | 测试准确率 | 相对无增强 |
|---|---:|---:|
| 无增强 | 87.51% | 0.00 个百分点 |
| 仅 ColorJitter | 87.51% | 0.00 个百分点 |
| 仅 RandomCrop | 90.55% | +3.04 个百分点 |
| 仅 RandomHorizontalFlip | 90.76% | +3.25 个百分点 |
| 三者组合 | 91.92% | +4.41 个百分点 |

该实验为单随机种子探索性结果。完整记录见：

- `reports/augmentation_policy_summary.csv`
- `outputs/gpu_resnet_crop_seed42`
- `outputs/gpu_resnet_flip_seed42`
- `outputs/gpu_resnet_color_seed42`
- `outputs/gpu_resnet_none_seed42`

### 14.3 参数量匹配的架构对照

新增 `baseline_wide` CNN，使其参数量接近无 SE 残差网络：

| 模型 | 参数量 | 测试准确率均值 | 样本标准差 |
|---|---:|---:|---:|
| 无 SE 残差网络 | 2,777,674 | 92.07% | 0.16% |
| 参数量匹配 CNN | 2,799,362 | 90.73% | 0.19% |

参数量相差约 0.8%，但残差网络平均高 1.33 个百分点，说明性能差异不能仅由参数量解释。

完整记录见：

- `reports/architecture_multiseed_runs.csv`
- `outputs/gpu_baseline_wide_seed42`
- `outputs/gpu_baseline_wide_seed43`
- `outputs/gpu_baseline_wide_seed44`

上述新增实验均使用独立 GPU 环境，不能与 CPU 实验直接作为纯硬件对照。
多随机种子结果报告均值和样本标准差；部分对比进一步进行了配对 t 检验，
并报告 95% 置信区间，详见第 15.1 节。每组仅有三个随机种子，
相关统计结论应视为探索性证据。

## 15. 稳健性统计、计算量控制与断点恢复

### 15.1 置信区间与配对检验

三种子结果使用样本标准差和 95% t 置信区间，并按相同 seed 进行配对 t 检验。

| 比较 | 平均差异 | 95% CI | p 值 |
|---|---:|---:|---:|
| 无 SE - SE | +0.57 个百分点 | [+0.20%, +0.94%] | 0.0217 |
| SE 标签平滑 - 无标签平滑 | +0.13 个百分点 | [-0.22%, +0.48%] | 0.2443 |
| RandomCrop - 无增强 | +2.94 个百分点 | [+2.55%, +3.34%] | 0.0010 |
| RandomHorizontalFlip - 无增强 | +2.86 个百分点 | [+2.03%, +3.70%] | 0.0046 |
| ColorJitter - 无增强 | -0.04 个百分点 | [-1.09%, +1.01%] | 0.8845 |

由于每组只有三个随机种子，统计检验仅作为探索性证据，不能替代大规模重复实验。

结果文件：

- `reports/statistical_analysis.json`
- `src/analyze_robustness.py`

### 15.2 FLOPs 控制

FLOPs 按 32×32 输入、Conv2d 和 Linear 的乘加运算统计，一个乘加计为 2 FLOPs。

| 模型 | 参数量 | FLOPs |
|---|---:|---:|
| 无 SE 残差网络 | 2,777,674 | 0.8424 GFLOPs |
| FLOPs 匹配 CNN | 2,882,498 | 0.8728 GFLOPs |

FLOPs 匹配 CNN 与残差网络的参数量差异约 3.8%，FLOPs 差异约 3.6%。

三种子结果：

| 模型 | 测试准确率均值 | 样本标准差 |
|---|---:|---:|
| 无 SE 残差网络 | 92.07% | 0.16% |
| FLOPs 匹配 CNN | 90.90% | 0.08% |

在参数量和计算量都接近时，残差网络仍平均高 1.17 个百分点。

结果文件：

- `reports/flops_summary.json`
- `reports/flops_matched_multiseed_runs.csv`
- `src/measure_flops.py`

### 15.3 断点恢复

src/resume_train.py 支持恢复 src/train.py 生成的检查点，包括：

- 模型参数、优化器状态和学习率调度器状态；
- 已完成轮数、最佳验证准确率和最佳轮次；
- 训练历史及早停状态；
- 新检查点中保存的 Python、NumPy、PyTorch CPU 和 CUDA 随机状态。

恢复前须保留原实验目录中的 last_model.pt 和 best_model.pt。
指定新的输出目录时，脚本会复制原来的最佳模型，
供后续模型选择及最终评估使用。

恢复训练示例：

```powershell
python .\src\resume_train.py `
    --resume outputs\resnet\last_model.pt `
    --epochs 30 `
    --output-dir outputs\resnet
```

上述路径为示例，运行时应替换为实际检查点所在路径。
--epochs 表示目标总轮数，而非新增轮数。
恢复原训练计划时，应保持原定总轮数，以延续原学习率调度。

一致性检查：

```powershell
python .\src\verify_resume.py
```

已在当前 GPU 环境下通过两轮、小批次测试，连续训练与断点续训的
模型参数、优化器、调度器、随机状态和训练历史（不含耗时）完全一致。
该结果不保证跨设备、跨软件版本或其他训练设置下逐位一致。

旧检查点若缺少随机状态，仍可按原有方式恢复，
但不能保证与不中断训练得到完全一致的结果。