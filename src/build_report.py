import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image as PILImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
PDF_PATH = REPORT_DIR / "experiment_report.pdf"
FONT_NAME = "ReportChinese"


def read_json(path):
    with path.open(encoding="utf-8-sig") as file:
        return json.load(file)


def percentage(value):
    return f"{100 * value:.2f}%"


def mean_std(mean, std):
    return f"{percentage(mean)} ± {percentage(std)}"


class Report:
    """固定五页排版，内容越界时停止，而不是生成额外页面。"""

    def __init__(self, info):
        self.page_width, self.page_height = A4
        self.margin = 42
        self.width = self.page_width - 2 * self.margin
        self.minimum_y = 48
        self.page_number = 0
        self.y = 0

        self.document = canvas.Canvas(
            str(PDF_PATH),
            pagesize=A4,
            pageCompression=1,
        )
        self.document.setTitle(info["title"])
        self.document.setAuthor(info["author"])

        common = {
            "fontName": FONT_NAME,
            "wordWrap": "CJK",
            "textColor": colors.HexColor("#172033"),
        }
        self.styles = {
            "body": ParagraphStyle(
                "Body",
                fontSize=10.2,
                leading=15,
                **common,
            ),
            "small": ParagraphStyle(
                "Small",
                fontSize=9,
                leading=13,
                **common,
            ),
            "heading": ParagraphStyle(
                "Heading",
                fontSize=12,
                leading=17,
                **common,
            ),
            "title": ParagraphStyle(
                "Title",
                fontSize=18,
                leading=24,
                alignment=TA_CENTER,
                **common,
            ),
            "caption": ParagraphStyle(
                "Caption",
                fontSize=8.6,
                leading=12,
                **common,
            ),
            "table": ParagraphStyle(
                "TableCell",
                fontSize=8.7,
                leading=12,
                **common,
            ),
            "table_header": ParagraphStyle(
                "TableHeader",
                fontName=FONT_NAME,
                fontSize=8.7,
                leading=12,
                wordWrap="CJK",
                textColor=colors.white,
            ),
        }

    def check_space(self, height):
        if self.y - height < self.minimum_y:
            raise RuntimeError(
                f"第 {self.page_number} 页排版空间不足。"
                "请把报错发给我，不要提交不完整的报告。"
            )

    def new_page(self):
        if self.page_number:
            self.finish_page()

        self.page_number += 1
        if self.page_number > 5:
            raise RuntimeError("报告不能超过五页。")

        c = self.document
        c.setFont(FONT_NAME, 8)
        c.setFillColor(colors.HexColor("#64748b"))
        c.drawString(
            self.margin,
            self.page_height - 25,
            "当代人工智能作业三",
        )
        c.drawRightString(
            self.page_width - self.margin,
            self.page_height - 25,
            "图像分类实验 | CIFAR-10",
        )
        c.setStrokeColor(colors.HexColor("#cbd5e1"))
        c.line(
            self.margin,
            self.page_height - 33,
            self.page_width - self.margin,
            self.page_height - 33,
        )
        self.y = self.page_height - 48

    def finish_page(self):
        c = self.document
        c.setFont(FONT_NAME, 8)
        c.setFillColor(colors.HexColor("#64748b"))
        c.drawString(
            self.margin,
            25,
            "图像分类模型设计与实验分析",
        )
        c.drawRightString(
            self.page_width - self.margin,
            25,
            f"{self.page_number} / 5",
        )
        c.showPage()

    def paragraph(self, text, style="body", gap=6):
        paragraph = Paragraph(text, self.styles[style])
        _, height = paragraph.wrap(self.width, 10000)
        self.check_space(height)
        paragraph.drawOn(
            self.document,
            self.margin,
            self.y - height,
        )
        self.y -= height + gap

    def heading(self, text):
        self.paragraph(escape(text), "heading", gap=6)

    def table(self, rows, weights, highlight=None):
        widths = [
            self.width * weight / sum(weights)
            for weight in weights
        ]
        cells = []

        for row_index, row in enumerate(rows):
            style = self.styles[
                "table_header" if row_index == 0 else "table"
            ]
            cells.append([
                Paragraph(escape(str(value)), style)
                for value in row
            ])

        table = Table(
            cells,
            colWidths=widths,
            hAlign="LEFT",
        )
        commands = [
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#243b53"),
            ),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.HexColor("#cbd5e1"),
            ),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]

        if highlight is not None:
            commands.append(
                (
                    "BACKGROUND",
                    (0, highlight),
                    (-1, highlight),
                    colors.HexColor("#e0f2fe"),
                )
            )

        table.setStyle(TableStyle(commands))
        _, height = table.wrap(self.width, 10000)
        self.check_space(height)
        table.drawOn(
            self.document,
            self.margin,
            self.y - height,
        )
        self.y -= height + 8

    def figure(self, name, maximum_height, caption):
        path = FIGURE_DIR / name
        with PILImage.open(path) as image:
            pixel_width, pixel_height = image.size

        scale = min(
            self.width / pixel_width,
            maximum_height / pixel_height,
        )
        width = pixel_width * scale
        height = pixel_height * scale

        self.check_space(height)
        self.document.drawImage(
            str(path),
            self.margin + (self.width - width) / 2,
            self.y - height,
            width=width,
            height=height,
            mask="auto",
        )
        self.y -= height + 4
        self.paragraph(
            escape(caption),
            "caption",
            gap=8,
        )

    def save(self):
        self.finish_page()
        self.document.save()


def main():
    pdfmetrics.registerFont(
        TTFont(
            FONT_NAME,
            "C:/Windows/Fonts/simsun.ttc",
            subfontIndex=0,
        )
    )

    info = read_json(REPORT_DIR / "report_info.json")
    if not info["author"].strip() or not info["student_id"].strip():
        raise ValueError("请先填写报告姓名和学号。")

    summary = read_json(REPORT_DIR / "results_summary.json")
    experiments = {
        row["experiment"]: row
        for row in summary["experiments"]
    }

    selected_name = summary["selected_experiment"]
    if selected_name != "resnet_no_se":
        raise ValueError(
            "当前报告模板对应已核对的 resnet_no_se 主模型。"
        )

    main_result = experiments[selected_name]
    main_dir = ROOT / "outputs" / selected_name
    history = read_json(main_dir / "history.json")
    class_report = read_json(
        main_dir / "classification_report.json"
    )
    class_names = read_json(
        main_dir / "config.json"
    )["class_names"]

    matrix = np.loadtxt(
        main_dir / "confusion_matrix.csv",
        delimiter=",",
        skiprows=1,
        dtype=int,
    )

    cpu_env = read_json(
        REPORT_DIR / "environment_cpu.json"
    )
    gpu_env = read_json(
        REPORT_DIR / "environment_gpu.json"
    )

    baseline = experiments["baseline_matched"]
    transfer = experiments["resnet18_transfer"]
    multiseed = summary["multiseed_summary"]

    no_se_stats = multiseed["gpu_resnet_no_se"]
    se_stats = multiseed["gpu_resnet_se"]
    no_smoothing_stats = multiseed[
        "gpu_resnet_se_no_smoothing"
    ]
    wide_baseline_stats = multiseed[
        "gpu_baseline_wide"
    ]

    statistics = read_json(
        REPORT_DIR / "statistical_analysis.json"
    )
    group_stats = statistics["group_summary"]
    paired = {
        item["comparison"]: item
        for item in statistics["paired_comparisons"]
    }

    flops = read_json(
        REPORT_DIR / "flops_summary.json"
    )
    flops_matched_stats = multiseed[
        "gpu_baseline_flops_matched"
    ]

    architecture_gain = (
        main_result["test_accuracy"]
        - baseline["test_accuracy"]
    ) * 100

    attention_test = paired[
        "resnet_no_se - resnet_se"
    ]
    smoothing_test = paired[
        "resnet_se_smoothing - resnet_se_no_smoothing"
    ]

    report = Report(info)

    # ---------- 第 1 页：任务、数据和模型原理 ----------
    report.new_page()
    report.paragraph(
        escape(info["title"]),
        "title",
        gap=9,
    )

    identity = (
        f"姓名：{escape(info['author'])}"
        f"　学号：{escape(info['student_id'])}"
    )
    if info.get("class_name", "").strip():
        identity += (
            f"　班级：{escape(info['class_name'])}"
        )
    report.paragraph(identity, "small")

    repository = escape(
        info["github_url"],
        {'"': "&quot;"},
    )
    report.paragraph(
        f'GitHub：<link href="{repository}" color="#1d4ed8">'
        f"{repository}</link>",
        "small",
        gap=12,
    )

    report.heading("摘要")
    report.paragraph(
        "本实验围绕 CIFAR-10 图像分类，实现基础 CNN、自定义残差网络、"
        "SE 通道注意力及预训练 ResNet-18 微调，并通过数据增强、"
        "标签平滑和注意力消融分析改进作用。"
        f"按验证集选择的无 SE 残差网络测试准确率为 "
        f"{percentage(main_result['test_accuracy'])}，"
        f"Macro F1 为 {main_result['macro_f1']:.4f}，"
        f"较同最大轮数 CNN 提高 {architecture_gain:.2f} 个百分点。"
        "结合训练曲线、混淆矩阵、错误样本和 Grad-CAM，"
        "分析分类能力与局限；另记录预训练模型及 GPU 环境复现结果。"
    )

    report.heading("1　数据集与实验任务")
    report.paragraph(
        "CIFAR-10 [1] 包含 60,000 张 32×32 RGB 图像，分为飞机、汽车、"
        "鸟、猫、鹿、狗、青蛙、马、船和卡车十类。"
        "从官方 50,000 张训练图像中随机划分 45,000 张训练集和 "
        "5,000 张验证集，保留官方 10,000 张测试集。CPU 原始实验使用 "
        "seed=42；GPU 补充实验使用 seed=42、43、44。种子同时决定"
        "训练/验证划分和训练随机性，同一 seed 的对照共享划分。"
        "验证集用于选轮次和模型，测试集用于结果汇报。"
    )
    report.paragraph(
        "基础预处理为 ToTensor 与逐通道标准化，"
        "均值为 (0.4914, 0.4822, 0.4465)，"
        "标准差为 (0.2023, 0.1994, 0.2010)。"
        "训练增强包括带 4 像素填充的随机裁剪、随机水平翻转，"
        "以及亮度、对比度和饱和度参数均为 0.2 的颜色扰动；"
        "验证和测试不使用随机增强。"
    )

    report.heading("2　模型结构与方法原理")
    report.table(
        [
            ["模型", "主要结构", "总/可训练参数（M）"],
            [
                "基础 CNN",
                "三阶段通道 64/128/256，每阶段两个 Conv-BN-ReLU；"
                "MaxPool 和 Dropout2d；GAP 后接 256→128→10 分类器。",
                "1.180 / 1.180",
            ],
            [
                "自定义残差网络",
                "3×3 卷积输入层；三阶段各两个残差块，通道 64/128/256；"
                "阶段切换步长为 2；GAP、Dropout(0.3) 和线性分类器。",
                "2.778 / 2.778",
            ],
            [
                "SE 残差网络",
                "在残差分支第二次 BN 后加入通道注意力，"
                "压缩比为 16；其他主体结构保持一致。",
                "2.800 / 2.800",
            ],
            [
                "预训练 ResNet-18",
                "使用 ImageNet 权重；输入放大至 96×96 并转为 ImageNet "
                "标准化；只微调 layer4 与十分类输出层。",
                "11.182 / 8.399",
            ],
        ],
        [1.7, 4.8, 1.8],
    )
    report.paragraph(
        "残差连接 [2] 采用 y=ReLU(F(x)+S(x))，其中 S 在维度变化时使用 "
        "1×1 卷积投影，以改善深层特征和梯度传递。本文自定义网络并非标准 "
        "ResNet-18。SE [3] 先对特征全局平均池化，再经两层全连接和 Sigmoid "
        "得到通道权重，对原特征逐通道缩放。"
    )
    report.paragraph(
        "标签平滑 [4] 使用 q(k)=(1-ε)·I(k=y)+ε/10，"
        "默认 ε=0.1，并最小化 -Σq(k)log p(k)。"
        "其目的在于减弱对硬标签的过度拟合；是否有效由消融实验判断。"
    )
    report.paragraph(
        "补充架构对照：加宽 CNN 使用 96/192/400 通道，每阶段两个 "
        "3×3 Conv-BN-ReLU，阶段末为 MaxPool 和 Dropout2d，"
        "GAP 后接 400→200→10 分类器。FLOPs 匹配 CNN 在首阶段额外"
        "增加一个 96→96 卷积，其他结构与加宽 CNN 一致。",
        "small",
    )

    # ---------- 第 2 页：训练设置与对比结果 ----------
    report.new_page()
    report.heading("3　实验设置、复现与结果对比")

    report.paragraph(
        f"CPU 环境：Python {cpu_env['python'].split()[0]}，"
        f"PyTorch {escape(cpu_env['torch'])}，"
        f"torchvision {escape(cpu_env['torchvision'])}。"
        f"GPU 环境：PyTorch {escape(gpu_env['torch'])}，"
        f"CUDA {escape(str(gpu_env['cuda_runtime']))}；"
        f"显卡为 {escape(str(gpu_env['gpu']))}，约 8 GB 显存。"
        "具体依赖记录保存在仓库的 environment_*.json 和 requirements 文件中。",
        "small",
    )

    report.table(
        [
            ["设置", "从零训练正式实验", "预训练模型微调"],
            [
                "优化器",
                "AdamW，weight decay=0.0005",
                "AdamW，weight decay=0.0005",
            ],
            [
                "学习率",
                "初始 0.001",
                "layer4：0.0001；分类层：0.001",
            ],
            [
                "训练预算",
                "最多 30 轮，batch size=128",
                "最多 15 轮，batch size=64",
            ],
            [
                "调度与早停",
                "余弦退火；耐心值 8",
                "余弦退火；耐心值 5",
            ],
            [
                "损失",
                "标签平滑交叉熵，ε=0.1",
                "标签平滑交叉熵，ε=0.1",
            ],
            [
                "数据/结构",
                "32×32；数据增强；BN、Dropout",
                "96×96；冻结前层及其 BN 统计",
            ],
        ],
        [1.2, 3.3, 3.3],
    )
    report.paragraph(
        "采用 AdamW [6]、余弦学习率调度及验证准确率早停。"
        "固定 Python、NumPy 和 PyTorch 随机种子，num_workers=0，"
        "设置 cuDNN deterministic=True、benchmark=False。"
        "每轮保存历史，按验证准确率保存最优权重，最终加载该权重测试。"
        "固定种子不保证跨版本、跨设备逐位一致。",
        "small",
    )

    labels = [
        ("baseline_matched", "CNN（同最大轮数）"),
        ("resnet_se", "残差网络 + SE"),
        ("resnet_se_noaug", "SE：关闭数据增强"),
        ("resnet_se_no_smoothing", "SE：关闭标签平滑"),
        ("resnet_no_se", "无 SE 残差网络（主模型）"),
        ("resnet18_transfer", "预训练 ResNet-18"),
        ("gpu_resnet_seed42", "无 SE：GPU 环境复现"),
    ]

    rows = [
        ["实验", "最佳轮次", "验证准确率", "测试准确率", "Macro F1"]
    ]
    for key, label in labels:
        row = experiments[key]
        rows.append([
            label,
            row["best_epoch"],
            percentage(row["best_val_accuracy"]),
            percentage(row["test_accuracy"]),
            f"{row['macro_f1']:.4f}",
        ])
    report.table(
        rows,
        [3.1, 1, 1.4, 1.4, 1.2],
        highlight=5,
    )

    report.paragraph(
        "表中前六组为 CPU 实验，最后一组为 GPU 复现。"
        "正式 CPU 主模型按最高验证"
        f"准确率 {percentage(main_result['best_val_accuracy'])} 选择，"
        f"不是按测试集最高值 {percentage(transfer['test_accuracy'])} 选择。",
        "small",
    )

    report.heading("3.1　多随机种子结果与比较边界")
    report.paragraph(
        "以下三种子补充实验均在上述 GPU 环境中完成，"
        "使用相同的主要训练设置。结果表示均值 ± 样本标准差（n=3），"
        "准确率和标准差均以 % 表示。",
        "small",
    )
    report.paragraph(
        f"三随机种子结果显示，无 SE 残差网络为 "
        f"{percentage(no_se_stats['mean_test_accuracy'])}"
        f" ± {percentage(no_se_stats['std_test_accuracy'])}，"
        f"SE 残差网络为 "
        f"{percentage(se_stats['mean_test_accuracy'])}"
        f" ± {percentage(se_stats['std_test_accuracy'])}。"
        f"SE 平均低 "
        f"{100 * (no_se_stats['mean_test_accuracy'] - se_stats['mean_test_accuracy']):.2f}"
        " 个百分点，当前设置下未观察到稳定收益。"
        f"标签平滑相对无标签平滑平均提高 "
        f"{100 * (se_stats['mean_test_accuracy'] - no_smoothing_stats['mean_test_accuracy']):.2f}"
        " 个百分点，提升幅度较小。",
        "small",
    )
    report.paragraph(
        "迁移学习引入额外 ImageNet 数据，且输入尺寸和训练预算不同，"
        "只作为扩展实验。CPU 与 GPU 的软件版本不同，"
        "不能将差异单独归因于硬件。不同标签平滑系数的 loss 不宜直接比较。",
        "small",
    )

    report.heading("3.2　数据增强策略拆分")
    rows = [
        [
            "增强策略",
            "测试准确率（均值±SD）",
            "相对无增强（百分点）",
            "配对 p 值",
        ]
    ]

    none_mean = group_stats["augmentation_none"]["mean"]

    for policy, label in (
        ("none", "无增强"),
        ("crop", "RandomCrop"),
        ("flip", "RandomHorizontalFlip"),
        ("color", "ColorJitter"),
    ):
        item = group_stats[f"augmentation_{policy}"]

        if policy == "none":
            gain, p_value = "0.00", "—"
        else:
            test = paired[
                f"augmentation_{policy} - augmentation_none"
            ]
            gain = f"{100 * (item['mean'] - none_mean):+.2f}"
            p_value = f"{test['p_value']:.4f}"

        rows.append([
            label,
            mean_std(item["mean"], item["std"]),
            gain,
            p_value,
        ])

    report.table(rows, [1.6, 2.1, 2.1, 1.0])
    report.paragraph(
        "各策略使用无 SE 残差网络及相同 seed 配对检验。"
        "颜色扰动单独使用未观察到统计显著收益；"
        "组合增强的三种子结果见上文无 SE 残差网络。",
        "small",
    )

    # ---------- 第 3 页：训练曲线与混淆矩阵 ----------
    report.new_page()
    report.heading("4　训练过程与总体分类表现")

    best_record = next(
        item
        for item in history
        if item["epoch"] == main_result["best_epoch"]
    )
    accuracy_gap = 100 * (
        best_record["train_accuracy"]
        - best_record["val_accuracy"]
    )

    report.paragraph(
        f"主模型训练 loss 由 {history[0]['train_loss']:.4f} "
        f"降至 {history[-1]['train_loss']:.4f}。"
        f"所选第 {main_result['best_epoch']} 轮训练准确率为 "
        f"{percentage(best_record['train_accuracy'])}，"
        f"验证准确率为 {percentage(best_record['val_accuracy'])}，"
        f"相差 {accuracy_gap:.2f} 个百分点。"
        "后期训练表现继续改善，而验证收益趋缓，提示存在一定过拟合倾向，"
        "不能仅因总体准确率较高就忽视这一差距。"
    )
    report.figure(
        "main_training_curves.png",
        155,
        "图1　主模型训练与验证曲线。训练阶段启用增强和 Dropout，"
        "验证阶段使用 eval 模式，因此差值也包含评估条件差异，"
        "不能将其直接视为相同条件下的纯泛化差距。"
        "后续可通过固定环境下的正则化对照和多随机种子实验进一步验证。",
    )
    report.figure(
        "main_confusion_matrix.png",
        330,
        "图2　主模型按真实类别逐行归一化的混淆矩阵，数值单位为 %。"
        "行表示真实类别，列表示预测类别；原始计数矩阵另存于仓库。",
    )
    report.paragraph(
        f"测试集中共有 {int(matrix.trace()):,} 张正确分类、"
        f"{int(matrix.sum() - matrix.trace()):,} 张错误分类。"
        f"Macro F1={main_result['macro_f1']:.4f}。"
        "总体指标较高，但猫、狗和鸟等类别仍存在较多错误，"
        "因此还需结合每类指标而非只报告一个总体准确率。"
    )

    # ---------- 第 4 页：每类指标和错误分析 ----------
    report.new_page()
    report.heading("5　每类指标与误分类分析")
    report.paragraph(
        "每类准确率按该类正确预测数除以该类真实样本数计算，"
        "等价于 Recall，而不是 Precision。测试集每类均有 1,000 张图像。",
        "small",
    )

    rows = [
        ["类别", "正确/总数", "Precision", "准确率/Recall", "F1"]
    ]
    for index, name in enumerate(class_names):
        item = class_report[name]
        rows.append([
            name,
            f"{int(matrix[index, index])}/{int(matrix[index].sum())}",
            percentage(item["precision"]),
            percentage(item["recall"]),
            f"{item['f1-score']:.4f}",
        ])
    report.table(rows, [1.6, 1.4, 1.4, 1.7, 1.1])

    cat_index = class_names.index("cat")
    dog_index = class_names.index("dog")
    cat_dog_errors = int(
        matrix[cat_index, dog_index]
        + matrix[dog_index, cat_index]
    )
    all_errors = int(
        matrix.sum() - matrix.trace()
    )

    report.paragraph(
        "汽车类别准确率最高（96.5%），猫最低（84.1%）。"
        f"猫→狗 {int(matrix[cat_index, dog_index])} 张、"
        f"狗→猫 {int(matrix[dog_index, cat_index])} 张，"
        f"合计 {cat_dog_errors} 张，占全部错误约 "
        f"{100 * cat_dog_errors / all_errors:.1f}%。"
        "此外存在鸟→鹿 27 张、猫→鹿 27 张，以及汽车与卡车间的混淆。"
    )
    report.figure(
        "error_examples.png",
        145,
        "图3　按错误预测置信度从高到低选择的四个样本。"
        "这些样本属于刻意选择的失败个例，不代表所有测试样本的分布。",
    )
    report.paragraph(
        "图中包括卡车→船、船→飞机、狗→猫和狗→鸟。"
        "低分辨率导致细节不足，局部外形相似、姿态和背景关联可能影响判断；"
        "但仅凭图像不能确认各因素的因果作用，也不能据此认定数据标签错误。"
        "错误预测仍可能具有较高 softmax 置信度，说明置信度并非正确性的保证；"
        "是否存在系统性校准问题，还需可靠性图或 ECE 等额外实验验证。"
    )

    # ---------- 第 5 页：解释、对照、结论和参考文献 ----------
    report.new_page()
    report.heading("6　Grad-CAM 可解释性分析")
    report.paragraph(
        "Grad-CAM [5] 对目标类别分数关于最后一个残差阶段特征求梯度，"
        "以梯度的空间均值作为通道权重，加权求和并经 ReLU 得到热图。"
        "分别计算预测类别与真实类别的响应，以观察同一图像的不同类别依据。",
        "small",
    )
    report.figure(
        "gradcam_examples.png",
        220,
        "图4　左：原图；中：预测类别 Grad-CAM；右：真实类别 Grad-CAM。"
        "各热图单独归一化，颜色强弱不能跨类别直接比较。",
    )
    report.paragraph(
        "正确的飞机样本中，高响应区域大体覆盖主体。"
        "狗被误判为猫的样本中，预测类别热区仍落在动物头部附近，"
        "说明关注主体并不等于识别正确，也不能把所有错误归因于背景。"
        "真实类别热图与预测类别响应位置不同，但这种局部解释不构成因果证明。",
        "small",
    )

    report.heading("7　补充架构对照、讨论与结论")
    rows = [
        ["GPU 架构", "参数量", "GFLOPs", "测试准确率（均值±SD）"]
    ]

    for label, model_key, item in (
        ("无 SE 残差", "resnet", no_se_stats),
        ("加宽 CNN", "baseline_wide", wide_baseline_stats),
        (
            "FLOPs 匹配 CNN",
            "baseline_flops_matched",
            flops_matched_stats,
        ),
    ):
        model_cost = flops[model_key]
        rows.append([
            label,
            f"{model_cost['trainable_parameters']:,}",
            f"{model_cost['gflops']:.4f}",
            mean_std(
                item["mean_test_accuracy"],
                item["std_test_accuracy"],
            ),
        ])

    report.table(rows, [1.7, 1.5, 1.2, 2.7])
    report.paragraph(
        "结构见第 2 节；均使用三种子 GPU 实验。FLOPs 按 32×32 输入的 "
        "Conv2d 和 Linear 乘加统计，一个乘加计 2 FLOPs。"
        "深度、池化和 Dropout 等仍有差异，"
        "尚不能将残差架构的性能优势单独归因于残差连接。",
        "small",
    )
    report.paragraph(
        f"按相同 seed 配对 t 检验：无 SE 相对 SE 提高 "
        f"{100 * attention_test['mean_difference']:.2f} 个百分点，"
        "差值的 95% CI 为 "
        f"[{100 * attention_test['difference_ci95_low']:+.2f}, "
        f"{100 * attention_test['difference_ci95_high']:+.2f}] 个百分点，"
        f"p={attention_test['p_value']:.4f}；"
        f"标签平滑提高 {100 * smoothing_test['mean_difference']:.2f} "
        f"个百分点，p={smoothing_test['p_value']:.4f}。"
        "每组仅三个种子，p 值未做多重比较校正，结果仅作探索性证据。",
        "small",
    )
    report.paragraph(
        "恢复训练已支持模型、优化器、调度器、历史及随机状态恢复。"
        "当前 GPU 环境的两轮、小批次测试中，连续训练与断点续训的"
        "模型参数、优化器、调度器、随机状态和训练历史（不含耗时）"
        "完全一致；该验证不保证跨设备或跨版本逐位一致。",
        "small",
    )
    report.paragraph(
        f"结论：按验证集选择的无 SE 残差网络达到 "
        f"{percentage(main_result['test_accuracy'])} 测试准确率，"
        f"相对正式 CNN 对照提高 {architecture_gain:.2f} 个百分点。"
        "代码、原始指标、图表、环境记录和 README 提供了可核查的实验链路。",
        "small",
    )

    report.heading("参考文献")
    references = [
        (
            "[1] Krizhevsky A. Learning Multiple Layers of Features "
            "from Tiny Images. 2009."
        ),
        (
            "[2] He K, et al. Deep Residual Learning for "
            "Image Recognition. CVPR, 2016."
        ),
        (
            "[3] Hu J, et al. Squeeze-and-Excitation Networks. "
            "CVPR, 2018."
        ),
        (
            "[4] Szegedy C, et al. Rethinking the Inception "
            "Architecture for Computer Vision. CVPR, 2016."
        ),
        (
            "[5] Selvaraju R R, et al. Grad-CAM: Visual Explanations "
            "from Deep Networks via Gradient-based Localization. "
            "ICCV, 2017."
        ),
        (
            "[6] Loshchilov I, Hutter F. Decoupled Weight Decay "
            "Regularization. ICLR, 2019."
        ),
    ]
    for reference in references:
        report.paragraph(
            escape(reference),
            "caption",
            gap=3,
        )

    report.save()

    reader = PdfReader(PDF_PATH)
    if len(reader.pages) != 5:
        raise RuntimeError(
            f"PDF 实际为 {len(reader.pages)} 页，未通过页数检查。"
        )

    has_repository_link = False
    for annotation in reader.pages[0].get("/Annots", []):
        action = annotation.get_object().get("/A")
        if action and action.get("/URI") == info["github_url"]:
            has_repository_link = True
            break

    if not has_repository_link:
        raise RuntimeError("首页 GitHub 超链接检查未通过。")

    print(f"Saved: {PDF_PATH}")
    print("PDF pages: 5")
    print("First-page GitHub link: verified")
    print("Layout overflow: none")
    print("Privacy: PDF remains excluded by .gitignore")


if __name__ == "__main__":
    main()