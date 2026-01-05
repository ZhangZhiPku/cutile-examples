# cutile-examples

这是一个基于 **cutile** 的 GPU 算子示例仓库。我会不定期往这里添加新的算子实现。  

This repository contains example GPU kernels implemented with **cutile**.  
I will continuously add new kernels from time to time.

---

## 目录 / Table of Contents

1. **Vector Sum with atomic_add**  
   使用 `atomic_add` 实现向量求和，演示并行规约与原子操作的基本用法。  
   Vector summation implemented with `atomic_add`, demonstrating basic reduction and atomic operations.

2. **Vector Norm**  
   计算向量归一化，从向量中减去均值，除以方差。  
   Normalize a vector.

3. **Tensor Per-Channel Quantization (float → int8)**  
   按通道对张量进行量化，将浮点数转换为 int8，适用于模型压缩和推理加速。  
   Per-channel tensor quantization from float to int8, useful for model compression and inference.

4. **Tensor Softmax (Safe Softmax)**  
   数值稳定的 softmax 实现，通过减去最大值避免溢出。  
   Numerically stable softmax implementation that subtracts the max to avoid overflow.

5. **Tensor Random Generator**  
   在 GPU 上生成随机数，用于初始化或数据增强。  
   Random tensor generation on GPU, for initialization or data augmentation.

6. **2D Image to Patches**  
   将 2D 图像切分成 patch（类似 ViT 的 patch embedding 前处理）。  
   Convert 2D images into patches, similar to the patch extraction step in Vision Transformers.

7. **Matmul**  
   基础矩阵乘法实现，演示 tile/block-based 的 GEMM 算法。  
   Basic matrix multiplication, demonstrating a tiled/block-based GEMM implementation.

8. **Matmul with Per-Channel Quantization**  
   在矩阵乘法中融合 per-channel 量化逻辑，用于量化推理场景。  
   Matrix multiplication fused with per-channel quantization for efficient quantized inference.

9. **Fused Attention (Simplified Flash Attention)**  
   融合 QK^T、softmax、与 V 的乘法的简化版 Flash Attention，实现高效注意力计算。  
   A simplified Flash Attention-like kernel fusing QK^T, softmax, and V multiplication for efficient attention.

10. **RMSNorm**  
    Root Mean Square Layer Normalization 的实现，无中心化的归一化算子。  
    Implementation of Root Mean Square Layer Normalization, a non-centering normalization operator.

11. **LayerNorm**  
    标准 Layer Normalization 实现，用于稳定深度网络训练。  
    Standard Layer Normalization implementation for stabilizing deep neural network training.

12. **Rotary Embedding (RoPE)**  
    旋转位置编码实现，常用于 Transformer 中的相对位置感知。  
    Rotary positional embeddings as used in modern Transformers for relative position encoding.

13. **SiLU & Mul**  
    SiLU 激活函数（又称 Swish）及其与输入的融合乘法算子。  
    SiLU (a.k.a. Swish) activation and its fused multiply operator.

14. **MSE Loss Function**  
    Mse 损失函数与其梯度传播

15. **Fused AdamW**  
    AdamW 优化器的融合实现，包含了更新 w, m, v 的操作

16. **Fused Muon**  
    Muon 优化器 Newton-Schulz 迭代的高性能实现，包含了针对对称矩阵的特殊矩阵乘法
---

## 使用方式 / How to Use

1. 克隆仓库 / Clone the repository

```bash
   git clone https://github.com/ZhangZhiPku/cutile-examples
   cd cutile-examples
```

## 联系方式 / Contact

如果你有问题、建议，或者希望看到更多算子示例，请直接在 GitHub Issue 中留言。

If you have questions, suggestions, or requests for more examples,
please leave a note in the GitHub issues.