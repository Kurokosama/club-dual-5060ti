# 双 RTX 5060 Ti 本地跑 LLM

我自己这台机器上的配置和真实验证过的速度。有双 5060 Ti、想试本地大模型的朋友可以参考下，不过**数字是我这台机器跑出来的，抄完最好自己复测**。

This is my personal setup running local LLMs on **2× RTX 5060 Ti 16GB** — hardware, software, and speed I actually measured (not estimated). 下面按「日期 + 模型名」记每次换模型 / 实测，**最新的在最上面**，以后再加新条目就照这个模板往上插。

## 硬件

| 部件 | 配置 |
| --- | --- |
| GPU | 2× NVIDIA GeForce RTX 5060 Ti 16GB（合计 32GB，无 P2P）|
| CPU | AMD Ryzen 5 4500（PCIe 3.0 only）|
| 主板 | MSI B550-A PRO |
| 电源 | Seasonic Focus GX-1000 1000W |
| 内存 | 64GB DDR4 |

测下来真正的瓶颈**不是 PCIe**，而是 **VRAM 带宽 + 跨卡延迟**（PCIe 利用率 <4%、单卡 117/180W、显存时钟顶满）。

**关键：2×16GB 是两块独立卡，不是 32GB 统一池子。** 谁先嗑满谁崩，上限由单卡（16GB）决定；且 GPU0 还要额外扛词表嵌入 + 视觉编码器，往往最先吃紧。

## 软件

| 项 | 配置 |
| --- | --- |
| 运行时 | llama.cpp（CUDA，自编译；Blackwell `sm_120`）|
| 模型 | **Ornith-1.5-35B-A3B · Q4_K_M**（QWEN35MOE，MoE ~3.5B 激活）|
| Tensor split | 1,1（50/50）|
| 投机解码 | 关（纯自回归，见下条 8/19 的优化说明）|
| 上下文 | 256K 原生（实测加载成功）|
| 视觉 | 有（mmproj + `--image-min-tokens 1024`）|
| KV | q4_0（tensor-split 下只能用这个）|
| effort | 固定 low |

> 备注：官方下载的 release 是 **CUDA 版**。用 Vulkan 那套 decode 基本只能单卡、白瞎第二张卡——想用满双卡 tensor-split 得**自己编译 CUDA 版**（Blackwell `sm_120`）。

详细启动命令、做了什么、速度、优化和坑，都写在下方最新的那条「8/19 · Ornith」里。

---

## 8/19 · Ornith-1.5-35B-A3B

**模型**：Ornith-1.5-35B-A3B，量化 **Q4_K_M**（架构 QWEN35MOE，MoE ~3.5B 激活，原生多模态）
**Hugging Face**：[ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)

**干了啥**：把生产从 Qwen3.8-27B 整体换到这台 MoE。定档 **256K 原生上下文 + 视觉 + 关 MTP**。

**启动命令**：
```bash
llama-server \
  -m /path/to/ornith-1.5-35b-q4_k_m.gguf \
  --mmproj /path/to/mmproj-ornith-1.5-35b-bf16.gguf \
  --image-min-tokens 1024 \
  --ctx-size 256000 --flash-attn on \
  --tensor-split 1,1 -ngl 999 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --reasoning-effort low --host 127.0.0.1 --port 8081
```
脚本版：[`examples/llamacpp-ornith-35b-a3b-dual-5060ti.sh`](examples/llamacpp-ornith-35b-a3b-dual-5060ti.sh)

**速度**（实测）：
| 指标 | 数值 |
| --- | --- |
| 纯生成 decode | **~109 tok/s** | 
| （对比旧 Qwen3.8）| ~46 tok/s |

**优化**：关掉了 MTP。上游 **llama.cpp #26750**（CUDA 后端 draft-mtp 接受率崩塌，CUDA 40.7% vs Vulkan 92%，同 GGUF 可复现）。实机 A/B：

| 配置 | 实测 tok/s | 接受率 |
| --- | ---: | ---: |
| MTP n_max=3 | ~73 | 0.24 |
| MTP n_max=4 | ~64 | 0.19 |
| MTP n_max=6 | ~49 | 0.12 |
| MTP n_max=4 + p_min=0.75 | ~43 | 0.83 |
| **纯自回归（关 MTP）** | **~109** | — |

MTP 在 CUDA 上不仅没提速反而拖慢 ~1/3，调 n_max / p_min 都救不回 → 这台机器**必须关 spec**，纯自回归就是天花板（≈1.5×）。等上游修好 #26750 再复测。

**遇到的坑**：
- **Ornith 高量化档**：Q5（25GB）在 **256K+视觉下 OOM**，只能 256K 纯文本或视觉降 ctx；Q6 干脆装不下。→ 256K+视觉下 Q4_K_M 就是这台机器的天花板。
- **DFlash2**（上游 PR #27342，block-diffusion 投机解码，就是有人吹"单卡 4090 90 tok/s"那套）：实测在 **tensor-split 双卡**上直接崩（`GGML_ASSERT ... SPLIT_AXIS_0`）；换成 layer 模式也≈MTP，不如纯自回归。营销数，别信。

---

## 8/18 · Qwen3.8-27B

**模型**：Qwen3.8-27B，量化 **UD-Q5_K_XL**（后来的 Unsloth Dynamic V3 也为同款，见下）
**Hugging Face**：[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)

**干了啥**：这台机器最初的主力配置（在换 Ornith 之前）。留档供回滚。

**启动命令**：
```bash
llama-server \
  -m /path/to/qwen3.8-27b-ud-q5_k_xl.gguf \
  --ctx-size 192000 \
  --tensor-split 50/50 \
  --split-mode layer \
  -ngl 999 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --mtp 3 \
  --reasoning-effort low \
  --host 127.0.0.1 --port 8081
```
脚本版：[`examples/llamacpp-qwen38-27b-dual-5060ti.sh`](examples/llamacpp-qwen38-27b-dual-5060ti.sh)

**速度**（实测）：
| 指标 | 数值 |
| --- | --- |
| 纯生成 decode | **~46 tok/s**（提交输出，同 Ornith 口径；早年记过 ~70 是服务端 eval 数、含投机草稿 token，虚高不可比）|
| 端到端 | 长输出 ~57（1000 token 实测）；短输出更低（prefill 占大头）|

**优化**：MTP-3（内置草稿 token）；`-p-min` 实测是负优化，别开。

**遇到的坑**：
- 上游某个 commit（`51a4f6303`）把 spec 挪进 worker 线程后，这个拓扑下 MTP 全被拒，decode 掉到 ~19 tok/s（后查明与 CUDA MTP 问题同源）。
- tensor-split 下 KV 只能用 q4_0，换 q4_1/q5_0 直接崩（Exit 139）。
- vLLM 上开 MTP 也是负优化。

#### 备选引擎：vLLM + NVFP4
想接标准 OpenAI API（Open WebUI / Copilot 之类）的话，Docker vLLM 跑 NVFP4 也行——有多模态，但**不是最快的**。
```bash
# Docker 版启动，见 examples/vllm-qwen38-27b-nvfp4-dual-5060ti.sh
docker run --rm --gpus all \
  -v /path/to/model:/model:ro -p 8081:8000 \
  vllm/vllm-openai:latest \
  --model /model --tensor-parallel-size 2 \
  --max-model-len 65536 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 --max-num-seqs 1
```
8/14 在同样两台卡上跑过 A/B：

| 方案 | tok/s |
| --- | ---: |
| llama.cpp + MTP-3 | 61.3–67.8（8/14 用 Q4_K_XL 测）✅ |
| vLLM NVFP4（无 MTP）| 31.7 |
| vLLM NVFP4 + MTP-2 | 15.7 |
| SGLang PP=2 | 6.4 |

一句话：**双卡下 llama.cpp 明显是 vLLM 的两倍**；vLLM 上开 MTP 反而更慢（无 P2P，草稿每轮跨一次 PCIe 同步）。（表中 llama.cpp 那行是 8/14 用更轻的 Q4_K_XL 测的；Q5_K_XL 单请求提交输出约 46 tok/s。）

> 注：这条条目里提到的 **Qwen3.8 UD-Q5_K_XL V3**（Unsloth Dynamic 3.0，2026-08-19 同尺寸精度提升、速度同 v2）也测过并短暂换装，因整机切到 Ornith 而未继续使用，文件保留可回滚。

---
MIT License.
