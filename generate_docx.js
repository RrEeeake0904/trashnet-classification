/**
 * 生成《基于CNN的垃圾分类识别算法复现研究》.docx
 * 格式：小四号宋体正文 / 黑体标题 / 1.5倍行距 / 标准页边距
 */
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  ImageRun, PageBreak, ExternalHyperlink, LevelFormat
} = require('docx');

const PROJECT = 'C:\\Users\\xushipan\\Desktop\\人工智能课程论文\\project';
const OUT = path.join(PROJECT, 'outputs');

// 字体与尺寸定义 (docx-js 使用 half-points: 1pt = 2)
const SZ_XIAOER    = 36; // 小二 (18pt) - 论文标题
const SZ_XIAOSAN   = 30; // 小三 (15pt) - 一级标题 (一、二...)
const SZ_SI        = 28; // 四号 (14pt) - 二级标题 (1.1, 1.2...)
const SZ_XIAOSI    = 24; // 小四 (12pt) - 正文
const SZ_WU        = 21; // 五号 (10.5pt) - 表格
const SZ_XIAOWU    = 18; // 小五 (9pt) - 参考文献/来源说明

// 1.5倍行距: 1.5 * 240 = 360 (twips)
const LINE_SPACING_15 = { line: 360, lineRule: "auto" };
// 单倍行距
const LINE_SPACING_1 = { line: 240, lineRule: "auto" };

// 字体
const FONT_BODY = "SimSun";     // 宋体
const FONT_HEI = "SimHei";      // 黑体
const FONT_KAI = "KaiTi";       // 楷体

// 页边距 (cm → DXA: 1cm = 567 DXA)
const marginTop = Math.round(2.54 * 567);
const marginBottom = Math.round(2.54 * 567);
const marginLeft = Math.round(3.18 * 567);
const marginRight = Math.round(3.18 * 567);

// 内容宽度
const pageWidth = 11906; // A4
const contentWidth = pageWidth - marginLeft - marginRight;

// 辅助函数
function bodyText(text, options = {}) {
  return new Paragraph({
    spacing: { after: 0, ...LINE_SPACING_15 },
    indent: options.noIndent ? undefined : { firstLine: 480 }, // 两字符缩进
    alignment: options.align || AlignmentType.JUSTIFIED,
    ...options.paragraphOpts,
    children: [
      new TextRun({
        text,
        font: options.font || FONT_BODY,
        size: options.size || SZ_XIAOSI,
        bold: options.bold || false,
        italics: options.italics || false,
      }),
    ],
  });
}

function multiRunParagraph(runs, options = {}) {
  return new Paragraph({
    spacing: { after: 0, ...LINE_SPACING_15 },
    indent: options.noIndent ? undefined : { firstLine: 480 },
    alignment: options.align || AlignmentType.JUSTIFIED,
    ...options.paragraphOpts,
    children: runs.map(r =>
      new TextRun({
        text: r.text,
        font: r.font || FONT_BODY,
        size: r.size || SZ_XIAOSI,
        bold: r.bold || false,
        italics: r.italics || false,
      })
    ),
  });
}

function heading1(text) {
  return new Paragraph({
    spacing: { before: 300, after: 200, ...LINE_SPACING_15 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, font: FONT_HEI, size: SZ_XIAOSAN, bold: true })],
  });
}

function heading2(text) {
  return new Paragraph({
    spacing: { before: 200, after: 120, ...LINE_SPACING_15 },
    alignment: AlignmentType.LEFT,
    children: [new TextRun({ text, font: FONT_HEI, size: SZ_SI, bold: true })],
  });
}

function emptyLine(size = SZ_XIAOSI) {
  return new Paragraph({
    spacing: { after: 0, line: 240, lineRule: "auto" },
    children: [new TextRun({ text: "", font: FONT_BODY, size })],
  });
}

// 表格
const CELL_BORDER = { style: BorderStyle.SINGLE, size: 1, color: "000000" };
const CELL_BORDERS = { top: CELL_BORDER, bottom: CELL_BORDER, left: CELL_BORDER, right: CELL_BORDER };

function makeCell(text, width, opts = {}) {
  return new TableCell({
    borders: CELL_BORDERS,
    width: { size: width, type: WidthType.DXA },
    shading: opts.header ? { fill: "D9E2F3", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    verticalAlign: "center",
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { after: 0, line: 240, lineRule: "auto" },
      children: [new TextRun({
        text,
        font: FONT_BODY,
        size: SZ_WU,
        bold: opts.header || false,
      })],
    })],
  });
}

function makeTableRow(cells, widths, header = false) {
  return new TableRow({
    children: cells.map((c, i) => makeCell(c, widths[i], { header })),
  });
}

function makeTable(headers, rows, widths) {
  const totalWidth = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: widths,
    alignment: AlignmentType.CENTER,
    rows: [
      makeTableRow(headers, widths, true),
      ...rows.map(r => makeTableRow(r, widths, false)),
    ],
  });
}

// 图片 (docx-js transformation 单位为 96 DPI 像素)
// 内容区宽度约 5.76 inches ≈ 553 px @96DPI
function imagePara(imgPath, captionText, imgWidth) {
  const imgFullPath = path.join(PROJECT, imgPath);
  const imgData = fs.readFileSync(imgFullPath);
  const ext = path.extname(imgPath).slice(1);

  // 从实际像素计算宽高比
  const { execSync } = require('child_process');
  // 直接用已知的图片尺寸 (png 150dpi)
  const imgDims = {
    'outputs/training_comparison.png': [1950, 750],
    'outputs/cm_ResNet18.png': [1200, 1050],
    'outputs/cm_CustomCNN.png': [1200, 1050],
  };
  const [pxW, pxH] = imgDims[imgPath] || [1200, 1050];
  const ratio = pxH / pxW;
  const imgHeight = Math.round(imgWidth * ratio);

  return [
    new Paragraph({
      spacing: { before: 200, after: 0, ...LINE_SPACING_15 },
      alignment: AlignmentType.CENTER,
      children: [new ImageRun({
        type: ext,
        data: imgData,
        transformation: { width: imgWidth, height: imgHeight },
      })],
    }),
    new Paragraph({
      spacing: { before: 60, after: 120, ...LINE_SPACING_15 },
      alignment: AlignmentType.CENTER,
      children: [new TextRun({
        text: captionText,
        font: FONT_BODY,
        size: SZ_XIAOWU,   // 小五号 9pt (图题/表题)
        italics: false,
      })],
    }),
  ];
}

// =============================
// 文档构建
// =============================

const children = [];

// === 标题 ===
children.push(emptyLine(SZ_XIAOER));
children.push(new Paragraph({
  spacing: { after: 80, ...LINE_SPACING_15 },
  alignment: AlignmentType.CENTER,
  children: [new TextRun({
    text: "基于卷积神经网络的垃圾分类识别算法复现研究",
    font: FONT_HEI,
    size: SZ_XIAOER,
    bold: true,
  })],
}));
children.push(emptyLine(SZ_XIAOER));

// === 来源说明 ===
children.push(new Paragraph({
  spacing: { after: 60, ...LINE_SPACING_1 },
  alignment: AlignmentType.LEFT,
  indent: { firstLine: 360 },
  children: [new TextRun({
    text: "本文所有代码已上传至 GitHub 公开仓库（https://github.com/RrEeeake0904/trashnet-classification），包含完整训练脚本、模型定义与实验结果。实验数据集来源于 TrashNet 公开数据集[1]，该数据集由 Gary Thung 与 Mindy Yang 于 2017 年发布在 GitHub 平台，采用 MIT 开源协议。本文代码实现基于 PyTorch 2.5.1 深度学习框架，关键参考资料包括吴恩达《深度学习》课程[2]（Bilibili 转载版）及 PyTorch 官方文档[3]。本文所有图表均为作者依据实验数据自行绘制，文中引用处均标注来源。",
    font: FONT_KAI,
    size: SZ_XIAOWU,
    italics: false,
  })],
}));

// 分隔线
children.push(new Paragraph({
  spacing: { before: 80, after: 80, line: 240, lineRule: "auto" },
  alignment: AlignmentType.CENTER,
  border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: "000000", space: 1 } },
  children: [],
}));

// === 摘要 ===
children.push(heading1("摘要"));
children.push(new Paragraph({
  spacing: { after: 0, ...LINE_SPACING_15 },
  alignment: AlignmentType.JUSTIFIED,
  indent: { firstLine: 480 },
  children: [new TextRun({
    text: "随着城市化进程加速，生活垃圾产量急剧增长，传统的依赖人工分拣的垃圾处理方式效率低下且环境恶劣。利用计算机视觉技术实现垃圾自动分类，是实现智慧城市与绿色可持续发展的重要技术手段。本文基于卷积神经网络（Convolutional Neural Network, CNN），在公开数据集 TrashNet 上对垃圾分类识别算法进行了复现与对比实验研究。TrashNet 数据集包含 6 类垃圾（纸板、玻璃、金属、纸张、塑料、其他垃圾），共计 2,527 张图像。本文分别构建了自定义 CNN 模型与基于 ResNet-18 的迁移学习模型进行对比实验。实验结果表明，自定义 CNN 模型在测试集上达到了 63.68% 的分类准确率，而 ResNet-18 迁移学习模型达到了 76.84% 的分类准确率，较自定义 CNN 提升了 13.16 个百分点。迁移学习方法利用在 ImageNet 大规模数据集上预训练的特征提取能力，有效缓解了小样本数据条件下的过拟合问题，显著提升了分类性能。本文的研究验证了深度学习技术在垃圾分类领域的可行性与有效性，为智能垃圾分类系统的实际部署提供了参考依据。",
    font: FONT_KAI,
    size: SZ_XIAOSI,
  })],
}));
children.push(emptyLine(SZ_XIAOSI));

// 关键词
children.push(multiRunParagraph([
  { text: "关键词", font: FONT_HEI, size: SZ_XIAOSI, bold: true },
  { text: "：图像分类；卷积神经网络；迁移学习；ResNet；垃圾分类；TrashNet 数据集", font: FONT_KAI, size: SZ_XIAOSI },
], { noIndent: true }));

children.push(emptyLine(SZ_XIAOSI));

// === 一、引言 ===
children.push(heading1("一、引言"));

// 1.1
children.push(heading2("1.1 研究背景"));
children.push(bodyText("随着城镇化进程的持续推进，我国城市生活垃圾年产生量已超过 2 亿吨，且仍保持增长态势[4]。传统的垃圾处理方式主要依赖人工分拣，不仅效率有限、成本高昂，还存在健康安全隐患。在此背景下，利用人工智能技术赋能传统环卫行业，实现垃圾的自动化、智能化分类，已成为智慧城市建设中的重要课题，具有显著的社会价值和应用前景。"));

// 1.2
children.push(heading2("1.2 研究意义"));
children.push(bodyText("从技术角度看，垃圾图像分类属于计算机视觉中的细粒度图像识别问题。同一大类下的不同类别（如不同材质垃圾）在视觉特征上可能具有高度的相似性，传统的基于手工特征（如 HOG、SIFT[5]）的机器学习方法难以有效应对此类问题。深度卷积神经网络通过多层非线性变换自动学习层次化特征表示，在处理图像分类问题上具有明显优势[6]。本研究通过对两种典型 CNN 方法（从零训练与迁移学习）的对比实验，旨在探究不同模型策略在小规模数据集下的表现差异，为相关领域研究者和工程师选择合适的技术方案提供实验依据。"));

// 1.3
children.push(heading2("1.3 研究内容"));
children.push(bodyText("本文的主要工作包括：（1）下载并预处理 TrashNet 公开垃圾分类数据集；（2）搭建自定义浅层 CNN 模型从零开始训练；（3）基于 ResNet-18 架构实施迁移学习方法；（4）在两个模型上进行对比实验，评估分类性能；（5）分析模型表现差异的原因，提出改进方向。本文所有实验代码均基于 PyTorch 深度学习框架编写，使用 NVIDIA GeForce RTX 4050 Laptop GPU 作为训练设备。"));

// === 二、相关算法简介 ===
children.push(heading1("二、相关算法简介"));

children.push(heading2("2.1 卷积神经网络基础"));
children.push(bodyText("卷积神经网络（CNN）是一类专门设计用于处理具有网格结构数据（如图像）的深度学习模型[6]。典型的 CNN 由以下几种基本层构成："));
children.push(bodyText("（1）卷积层（Convolutional Layer）：卷积层是 CNN 的核心组件，通过可学习的卷积核（Filter）在输入特征图上滑动，提取局部特征。形式上，对于输入特征图 X 和卷积核 K，输出特征图 Y 的计算公式为：Y_{i,j} = Σ_m Σ_n X_{i+m, j+n} · K_{m,n} + b，其中 b 为偏置项。低层卷积通常提取边缘、纹理等基础特征，高层卷积则组合这些基础特征以识别更复杂的语义模式[7]。"));
children.push(bodyText("（2）池化层（Pooling Layer）：池化层通过对局部区域进行下采样，减小特征图的空间尺寸，降低计算量并增强平移不变性。常用的最大池化（Max Pooling）操作选择局部区域内的最大值作为输出。"));
children.push(bodyText("（3）全连接层（Fully Connected Layer）：全连接层将展平后的高层特征向量映射到类别分数，完成最终的分类决策。为防止过拟合，在全连接层之间通常插入 Dropout 正则化[8]。"));

children.push(heading2("2.2 批归一化与 Dropout"));
children.push(bodyText("批归一化（Batch Normalization, BN）[9] 通过对每层激活值进行标准化处理，使网络对参数初始化的敏感度降低，加速训练收敛，并具有一定的正则化效果。Dropout 技术[8] 则在训练过程中以一定概率随机丢弃神经元输出，迫使网络学习更鲁棒的特征表示，有效缓解过拟合问题。"));

children.push(heading2("2.3 ResNet 与迁移学习"));
children.push(bodyText("深度网络的退化问题（Degradation Problem）表明，简单地堆叠更多层可能导致训练误差反而升高。ResNet（Residual Network）[10] 通过引入残差连接（Skip Connection），将层的输出建模为 H(x) = F(x) + x，其中 F(x) 为残差函数，x 为恒等映射。这一设计使得深层网络的梯度可以绕过中间层直接传播，有效解决了深层网络的训练难题。"));
children.push(bodyText("迁移学习（Transfer Learning）是指将在大规模数据集上预训练的模型参数迁移到目标任务上进行微调[11]。本文使用在 ImageNet（约 120 万张图像，1000 类）上预训练的 ResNet-18 模型，冻结其卷积层参数作为固定特征提取器，仅替换并训练最后的全连接分类层。这一策略在小规模数据集上尤为有效，因为预训练模型已习得通用视觉特征（如边缘、形状、纹理），迁移到新任务时仅需学习任务特定的分类决策边界。"));

// === 三、实验设计与应用场景 ===
children.push(heading1("三、实验设计与应用场景"));

children.push(heading2("3.1 数据集介绍"));
children.push(bodyText("本实验采用 TrashNet 数据集[1]，该数据集由 Gary Thung 和 Mindy Yang 使用智能手机（iPhone 7 Plus、iPhone 5S、iPhone SE）拍摄，将垃圾样本置于白色背景上采集而成。数据集共包含 2,527 张彩色 RGB 图像，分为 6 个类别："));
children.push(emptyLine(SZ_XIAOSI));

// 数据集表格
const dsWidths = [1600, 1400, 1200, 900, 900];
const dsTotal = dsWidths.reduce((a, b) => a + b, 0);
children.push(new Paragraph({
  spacing: { after: 0, ...LINE_SPACING_15 },
  alignment: AlignmentType.CENTER,
  children: [],
}));
children.push(new Table({
  width: { size: dsTotal, type: WidthType.DXA },
  columnWidths: dsWidths,
    alignment: AlignmentType.CENTER,
  rows: [
    makeTableRow(["类别", "英文名称", "图像数量", "占比", ""], dsWidths, true),
    makeTableRow(["纸板", "cardboard", "403", "15.9%", ""], dsWidths),
    makeTableRow(["玻璃", "glass", "501", "19.8%", ""], dsWidths),
    makeTableRow(["金属", "metal", "410", "16.2%", ""], dsWidths),
    makeTableRow(["纸张", "paper", "594", "23.5%", ""], dsWidths),
    makeTableRow(["塑料", "plastic", "482", "19.1%", ""], dsWidths),
    makeTableRow(["其他垃圾", "trash", "137", "5.4%", ""], dsWidths),
  ],
}));
children.push(emptyLine(SZ_XIAOSI));
children.push(bodyText('图像分辨率统一为 512×384 像素。数据集中各类别样本数量存在不均衡，其中“其他垃圾”（trash）类别仅含 137 张图像，占比仅 5.4%，是本实验需要关注的问题。'));

children.push(heading2("3.2 数据预处理与增强"));
children.push(bodyText("为适应 ResNet-18 标准输入要求，所有图像被缩放至 224×224 像素。数据集按 70%:15%:15% 的比例随机划分为训练集（1,768 张）、验证集（379 张）和测试集（380 张）。"));
children.push(bodyText("训练阶段采用以下数据增强策略以提升模型泛化能力：（1）随机水平翻转（概率 0.5）；（2）随机旋转（±15 度）；（3）色彩抖动（亮度、对比度、饱和度各 ±0.2）；（4）ImageNet 标准化（均值 [0.485, 0.456, 0.406]，标准差 [0.229, 0.224, 0.225]）。验证集与测试集仅进行缩放与标准化操作，不做随机增强。"));

children.push(heading2("3.3 模型设计"));
children.push(bodyText("模型一：自定义 CNN。特征提取部分由 4 个卷积块组成，各卷积块包含 2 层 3×3 卷积（带 BatchNorm 和 ReLU 激活），后接 2×2 最大池化。通道数逐块递增为 32→64→128→256。分类部分由 Dropout(0.5) → 全连接(50176→512) → ReLU → Dropout(0.5) → 全连接(512→6) 构成。模型共含约 2,627 万可训练参数，全部从随机初始化开始训练。"));
children.push(bodyText("模型二：ResNet-18 迁移学习。使用在 ImageNet 上预训练的 ResNet-18 模型，冻结所有卷积层参数，替换原始全连接层为：Dropout(0.5) → 全连接(512→256) → ReLU → Dropout(0.3) → 全连接(256→6)。模型总参数量约 1,131 万，其中可训练参数仅约 13.3 万（集中在重新设计的分类层），大幅降低了训练参数量和过拟合风险。"));

children.push(heading2("3.4 训练配置"));
children.push(bodyText("两个模型均使用以下统一超参数设置：优化器为 AdamW（学习率 1×10⁻³，权重衰减 1×10⁻⁴）；学习率调度为 ReduceLROnPlateau（监控验证损失，衰减因子 0.5，耐心值 5 轮）；损失函数为交叉熵损失（CrossEntropyLoss）；批量大小为 32（ResNet-18）和 16（CustomCNN）；最大训练轮数为 30（含早停机制，耐心值 10 轮）；训练设备为 NVIDIA GeForce RTX 4050 Laptop GPU（6GB 显存）。"));

children.push(heading2("3.5 应用场景"));
children.push(bodyText("本实验研究的垃圾分类识别技术具有以下实际应用场景：智能垃圾桶（嵌入边缘计算设备，对投入的垃圾进行实时分类与引导）；垃圾处理厂自动化分拣（结合传送带与机械臂，替代人工完成大批量垃圾的自动分类）；城市环卫管理系统（与物联网传感器网络集成，实现区域垃圾数据的自动统计与分析）；公众环保教育（通过手机应用辅助市民正确分类投放）。"));

// === 四、实验结果与分析 ===
children.push(heading1("四、实验结果与分析"));

children.push(heading2("4.1 模型性能对比"));
children.push(bodyText("两个模型在测试集上的对比结果如下表所示："));
children.push(emptyLine(SZ_XIAOSI));

// 性能对比表
const perfWidths = [2200, 1400, 1200, 1200, 1200];
const perfTotal = perfWidths.reduce((a, b) => a + b, 0);
children.push(new Table({
  width: { size: perfTotal, type: WidthType.DXA },
  columnWidths: perfWidths,
    alignment: AlignmentType.CENTER,
  rows: [
    makeTableRow(["模型", "验证准确率", "测试准确率", "总参数量", "可训练参数"], perfWidths, true),
    makeTableRow(["自定义 CNN", "65.70%", "63.68%", "26,277,286", "26,277,286"], perfWidths),
    makeTableRow(["ResNet-18 迁移学习", "79.16%", "76.84%", "11,309,382", "132,870"], perfWidths),
  ],
}));
children.push(emptyLine(SZ_XIAOSI));
children.push(bodyText("ResNet-18 迁移学习模型在测试集上取得了比自定义 CNN 高 13.16 个百分点的分类准确率（76.84% vs 63.68%），验证了迁移学习在小规模数据集上的显著优势。值得注意的是，ResNet-18 的可训练参数仅约 13.3 万，仅为自定义 CNN 的约 0.5%，却实现了大幅领先的分类性能，充分体现了预训练特征表示的迁移价值。"));

children.push(heading2("4.2 训练过程分析"));
children.push(bodyText("两个模型的训练与验证损失曲线（见图 1）清晰地揭示了两者在收敛行为上的显著差异："));
children.push(bodyText("ResNet-18 的快速收敛：ResNet-18 初始训练损失（epoch 1）仅约 1.41，初始验证准确率即达 67.81%，远高于自定义 CNN 的首轮 30.08%。经过约 5 个 epoch 后，验证准确率稳定在 72% 以上，全程未出现明显的验证损失反弹，最终在 epoch 24 达到最佳验证准确率 79.16%。这说明预训练权重为模型提供了良好的初始特征提取能力，模型仅需少量迭代即可收敛至较优解。"));
children.push(bodyText("自定义 CNN 的缓慢提升：自定义 CNN 从随机初始化出发，训练损失从 epoch 1 的 3.57 缓慢下降，验证准确率仅从 30.08% 逐步提升至 65.70%（epoch 30）。在训练后期（epoch 25 之后），训练准确率已接近 60% 而验证准确率在 62%-65% 区间波动，训练损失持续下降但验证损失未同步改善，表现出一定程度的过拟合趋势。"));
children.push(bodyText("学习率调度效果：ResNet-18 在 epoch 18 触发了一次学习率衰减（1×10⁻³ → 5×10⁻⁴），衰减后验证准确率从 77.31% 进一步提升至 79.16%（epoch 24），说明学习率衰减在训练后期对模型精调起到了积极作用。"));

// 图片: 训练曲线对比 (宽图, 约占5.0英寸)
const trainingImg = imagePara(
  "outputs/training_comparison.png",
  "图 1 训练与验证损失/准确率曲线对比",
  480  // 5.0英寸 @96DPI
);
children.push(...trainingImg);

children.push(heading2("4.3 混淆矩阵分析"));
children.push(bodyText("图 2 与图 3 分别展示了两个模型在测试集上的混淆矩阵。对各类别的分类表现分析如下："));
children.push(bodyText("高准确率类别：cardboard（纸板）和 paper（纸张）两个类别在两个模型上均取得了较高的分类 F1 分数。ResNet-18 对 cardboard 的精确率达 96.23%，召回率 86.44%（F1=0.91）；对 paper 的精确率 82.11%，召回率 84.78%（F1=0.83）。这可能是由于纸板与纸张具有较为统一的视觉纹理特征，易于与其他类别区分。"));
children.push(bodyText("易混淆类别：glass（玻璃）类别在 ResNet-18 上的召回率为 76.92%，但在 CustomCNN 上仅为 50.00%。玻璃的透明特性使得其视觉特征随背景和光照变化较大，对从零训练的模型造成更大挑战。metal（金属）和 plastic（塑料）之间的混淆也较为明显，尤其在 CustomCNN 中，两者精确率和召回率均在 50%-52% 左右。"));
children.push(bodyText("少样本困境：trash（其他垃圾）类别仅有 24 个测试样本（占测试集的 6.3%）。ResNet-18 对该类的召回率为 29.17%（即 24 张中仅正确识别 7 张），CustomCNN 更低至 8.33%（仅识别 2 张）。这直观地反映了类别不平衡问题对分类性能的严重影响——少数类被模型大量错分至多数类中。"));

// 图片: 混淆矩阵 (方图, 各约占3.5英寸)
const cmResNet = imagePara("outputs/cm_ResNet18.png", "图 2 ResNet-18 混淆矩阵", 336);
const cmCNN = imagePara("outputs/cm_CustomCNN.png", "图 3 自定义 CNN 混淆矩阵", 336);
children.push(...cmResNet);
children.push(...cmCNN);

// 各类别性能表
children.push(emptyLine(SZ_XIAOSI));
const clsWidths = [1400, 1200, 1200, 1200, 1000, 1200];
const clsTotal = clsWidths.reduce((a, b) => a + b, 0);
children.push(new Table({
  width: { size: clsTotal, type: WidthType.DXA },
  columnWidths: clsWidths,
    alignment: AlignmentType.CENTER,
  rows: [
    makeTableRow(["类别", "Precision", "Recall", "F1-Score", "样本数", "模型"], clsWidths, true),
    makeTableRow(["cardboard", "96.23%", "86.44%", "0.91", "59", "ResNet-18"], clsWidths),
    makeTableRow(["glass", "70.59%", "76.92%", "0.74", "78", "ResNet-18"], clsWidths),
    makeTableRow(["metal", "64.63%", "86.89%", "0.74", "61", "ResNet-18"], clsWidths),
    makeTableRow(["paper", "82.11%", "84.78%", "0.83", "92", "ResNet-18"], clsWidths),
    makeTableRow(["plastic", "78.18%", "65.15%", "0.71", "66", "ResNet-18"], clsWidths),
    makeTableRow(["trash", "70.00%", "29.17%", "0.41", "24", "ResNet-18"], clsWidths),
  ],
}));
children.push(emptyLine(SZ_XIAOSI));

children.push(heading2("4.4 分析与讨论"));
children.push(bodyText("（1）迁移学习的优势：ResNet-18 预训练模型在 ImageNet 上习得的底层特征（边缘、纹理、形状）在垃圾分类任务中依然有效，仅需少量数据即可微调分类决策边界。这在小规模数据集场景下具有重要的实践意义——收集大规模标注数据成本高昂，而迁移学习提供了一种高效经济的替代方案。"));
children.push(bodyText("（2）类别不平衡问题：TrashNet 数据集中 trash 类仅占 5.4%，造成模型对该类别的识别率偏低。在实际应用中，可通过过采样（Oversampling）、类别加权损失函数（Class-Weighted Loss）或合成少数类样本（如 SMOTE）等方法进行缓解。"));
children.push(bodyText("（3）计算资源约束：本实验在消费级笔记本 GPU（RTX 4050 6GB）上完成，训练时间合理。ResNet-18 的可训练参数仅 13.3 万，显著低于自定义 CNN 的 2,627 万，兼顾了模型性能与硬件开销，适合在嵌入式或边缘设备上部署。"));

// === 五、结论 ===
children.push(heading1("五、结论"));
children.push(bodyText("本文以生活垃圾图像分类为研究对象，基于公开数据集 TrashNet 和 PyTorch 深度学习框架，对自定义 CNN 模型与 ResNet-18 迁移学习模型进行了对比实验复现。实验结果表明，ResNet-18 迁移学习模型的测试准确率达到 76.84%，较大幅度领先于从零训练的自定义 CNN（63.68%），领先幅度达 13.16 个百分点。更重要的是，ResNet-18 仅需训练 13.3 万参数（约为自定义 CNN 的 0.5%），验证了预训练特征表示在小规模数据条件下的重要价值。同时，本文结合混淆矩阵分析了类别不平衡对分类性能的影响，指出了模型在 trash 等少样本类别上识别能力严重不足的问题（ResNet-18 召回率仅 29.17%，CustomCNN 仅 8.33%），并给出了可行的改进方向。"));
children.push(bodyText("本文的研究验证了基于深度学习的图像分类技术在垃圾分类领域的适用性，为智能环卫设备的算法设计提供了实验参考。未来工作可从以下方面展开：（1）引入类别平衡策略和 Focal Loss 等损失函数以应对样本不均衡；（2）尝试 EfficientNet 等更轻量的模型架构以适应边缘部署；（3）构建多模态数据集（深度图、光谱图）以提升难分样本的识别准确率。"));

// === 参考文献 ===
children.push(heading1("参考文献"));
const refs = [
  "[1] Thung G, Yang M. TrashNet: Dataset of images of trash[DS]. GitHub repository, 2017. https://github.com/garythung/trashnet",
  "[2] Ng A. Deep Learning Specialization[EB/OL]. Coursera, 2017. (Bilibili 转载版)",
  "[3] Paszke A, Gross S, Massa F, et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library[C]. Advances in Neural Information Processing Systems (NeurIPS), 2019: 8024-8035.",
  "[4] 国家统计局. 中国统计年鉴 2024[M]. 北京: 中国统计出版社, 2024.",
  "[5] Dalal N, Triggs B. Histograms of oriented gradients for human detection[C]. IEEE CVPR, 2005: 886-893.",
  "[6] LeCun Y, Bengio Y, Hinton G. Deep learning[J]. Nature, 2015, 521(7553): 436-444.",
  "[7] Zeiler M D, Fergus R. Visualizing and Understanding Convolutional Networks[C]. ECCV, 2014: 818-833.",
  "[8] Srivastava N, Hinton G, Krizhevsky A, et al. Dropout: A Simple Way to Prevent Neural Networks from Overfitting[J]. Journal of Machine Learning Research, 2014, 15(1): 1929-1958.",
  "[9] Ioffe S, Szegedy C. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift[C]. ICML, 2015: 448-456.",
  "[10] He K, Zhang X, Ren S, et al. Deep Residual Learning for Image Recognition[C]. IEEE CVPR, 2016: 770-778.",
  "[11] Yosinski J, Clune J, Bengio Y, et al. How transferable are features in deep neural networks?[C]. NeurIPS, 2014: 3320-3328.",
];

refs.forEach((ref, i) => {
  children.push(new Paragraph({
    spacing: { after: 40, ...LINE_SPACING_1 },
    alignment: AlignmentType.LEFT,
    indent: { left: 420, hanging: 420 },
    children: [new TextRun({
      text: ref,
      font: FONT_BODY,
      size: SZ_XIAOWU,
    })],
  }));
});

// =============================
// 生成文档
// =============================

const doc = new Document({
  styles: {
    default: {
      document: {
        run: { font: FONT_BODY, size: SZ_XIAOSI },
        paragraph: { spacing: LINE_SPACING_15 },
      },
    },
  },
  sections: [{
    properties: {
      page: {
        size: { width: pageWidth, height: 16838 }, // A4
        margin: {
          top: marginTop,
          bottom: marginBottom,
          left: marginLeft,
          right: marginRight,
        },
      },
    },
    children,
  }],
});

Packer.toBuffer(doc).then(buffer => {
  const outPath = path.join(PROJECT, "论文_垃圾分类CNN复现.docx");
  fs.writeFileSync(outPath, buffer);
  console.log(`[OK] 论文已生成: ${outPath}`);
  console.log(`文件大小: ${(buffer.length / 1024).toFixed(1)} KB`);
}).catch(err => {
  console.error("生成失败:", err);
  process.exit(1);
});
