# 双 RTX 5060 Ti 本地跑 LLM

这套双卡配置和速度，是我自己这台机器上实测出来的结果，给同样在折腾双 5060 Ti 的朋友做个参考。所有数字均来自本机实测，不同机器可能有差异，参考后最好自己复测一遍。

> This is my personal dual **2× RTX 5060 Ti 16GB** LLM setup — hardware, software, and speeds I actually measured here (no estimates). The rest is a reverse-chronological log keyed by「日期 + 模型名」: the newest entry sits at the top, and future model swaps/tests follow the same template.

## 硬件

| 部件 | 配置 |
| --- | --- |
| GPU | 2× NVIDIA GeForce RTX 5060 Ti 16GB（合计 32GB，无 P2P）|
| CPU | AMD Ryzen 5 4500（仅 PCIe 3.0）|
| 主板 | MSI B550-A PRO |
| 电源 | Seasonic Focus GX-1000 1000W |
| 内存 | 64GB DDR4 |

实测下来，真正的瓶颈不在 PCIe，而在 **VRAM 带宽 + 跨卡延迟**（PCIe 利用率 <4%、单卡约 117/180W、显存时钟基本满载）。

一个需要留意的点：**2×16GB 是两块独立卡，不是 32GB 的统一池子。** 显存先占满的卡会先出问题，容量上限由单卡（16GB）决定；GPU0 还要额外承担词表嵌入和视觉编码器，通常最先吃紧，配置时一般以 GPU0 为准。

## 软件

| 项 | 配置 |
| --- | --- |
| 运行时 | llama.cpp（CUDA，自编译；Blackwell `sm_120`）|
| 模型 | **Ornith-1.5-35B-A3B · Q4_K_M**（QWEN35MOE，MoE ~3.5B 激活）|
| Tensor split | 1,1（50/50）|
| 投机解码 | 关（纯自回归；原因见 8/19 条目）|
| 上下文 | 256K 原生（实测可加载）|
| 视觉 | 有（mmproj + `--image-min-tokens 1024`）|
| KV | q4_0（这台机器 tensor-split 下暂时只能用这个，换别的会导致崩溃，见 8/18 条目）|
| effort | 固定 low |

> 备注：官方 release 是 **CUDA 版**。用 Vulkan 那套，decode 基本只能发挥单卡性能、用不上第二张卡；想用满双卡 tensor-split，大概率还得自己编译 CUDA 版（Blackwell `sm_120`）。

详细的启动命令、实测数据、优化和遇到的问题，都写在下方最新的「8/19 · Ornith」条目里。

---

## 8/19 · Ornith-1.5-35B-A3B

**模型**：Ornith-1.5-35B-A3B，量化 **Q4_K_M**（架构 QWEN35MOE，MoE ~3.5B 激活，原生多模态）
**Hugging Face**：[ornith-ai/Ornith-1.5-35B-A3B-GGUF](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B-GGUF)

**本次改动**：把生产从 Qwen3.8-27B 整体换到这台 MoE，最终固定为 **256K 原生上下文 + 视觉 + 关 MTP**。

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

**优化**：关闭了 MTP。上游 **llama.cpp #26750**（CUDA 后端 draft-mtp 接受率崩塌：CUDA 40.7% vs Vulkan 92%，同 GGUF 可复现）。实机 A/B 结果：

| 配置 | 实测 tok/s | 接受率 |
| --- | ---: | ---: |
| MTP n_max=3 | ~73 | 0.24 |
| MTP n_max=4 | ~64 | 0.19 |
| MTP n_max=6 | ~49 | 0.12 |
| MTP n_max=4 + p_min=0.75 | ~43 | 0.83 |
| **纯自回归（关 MTP）** | **~109** | — |

在这台机器上，MTP 在 CUDA 下没有提速反而拖慢约 1/3，调整 n_max / p_min 也没能改善，所以**目前先关闭 spec**；纯自回归（约 1.5×）更贴近这台机器的实际上限。等上游修复 #26750 后可再复测。

**遇到的问题**：
- **Ornith 高量化档**：Q5（25GB）在 **256K+视觉下会 OOM**，只能退到 256K 纯文本或视觉降 ctx；Q6 直接放不下。就这台机器而言，256K+视觉下 Q4_K_M 已接近上限。
- **DFlash2**（上游 PR #27342，block-diffusion 投机解码，即有人宣传“单卡 4090 90 tok/s”的方案）：实测在 **tensor-split 双卡**上会崩溃（`GGML_ASSERT ... SPLIT_AXIS_0`）；换成 layer 模式表现也近似 MTP，不如纯自回归。宣传数字仅供参考，建议自行复测。

---

## 8/18 · Qwen3.8-27B

**模型**：Qwen3.8-27B，量化 **UD-Q5_K_XL**（后来的 Unsloth Dynamic V3 也是同款，见下）
**Hugging Face**：[unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)

**本次改动**：这台机器最初的主力配置（在换 Ornith 之前），保留存档供回滚。

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
| 纯生成 decode | **~46 tok/s**（提交输出，同 Ornith 口径；早年记录的 ~70 是服务端 eval 数、含投机草稿 token，偏高、不可直接比）|
| 端到端 | 长输出 ~57（1000 token 实测）；短输出更低（prefill 占大头）|

**优化**：MTP-3（内置草稿 token）；`-p-min` 实测是负优化，这台机器上暂时不建议开启。

**遇到的问题**：
- 上游某个 commit（`51a4f6303`）把 spec 挪进 worker 线程后，这个拓扑下 MTP 被拒，decode 降到 ~19 tok/s（后查明与 CUDA MTP 问题同源）。
- tensor-split 下 KV 只能用 q4_0，换 q4_1/q5_0 会导致崩溃（Exit 139）。
- vLLM 上开 MTP 也是负优化。

#### 备选引擎：vLLM + NVFP4

需要接标准 OpenAI API（Open WebUI / Copilot 等）的话，Docker vLLM 跑 NVFP4 也可以——有多模态，但不是最快的。
```bash
# Docker 版启动，见 examples/vllm-qwen38-27b-nvfp4-dual-5060ti.sh
docker run --rm --gpus all \
  -v /path/to/model:/model:ro -p 8081:8000 \
  vllm/vllm-openai:latest \
  --model /model --tensor-parallel-size 2 \
  --max-model-len 65536 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.95 --max-num-seqs 1
```
8/14 在同样两张卡上跑过 A/B：

| 方案 | tok/s |
| --- | ---: |
| llama.cpp + MTP-3 | 61.3–67.8（8/14 用 Q4_K_XL 测）✅ |
| vLLM NVFP4（无 MTP）| 31.7 |
| vLLM NVFP4 + MTP-2 | 15.7 |
| SGLang PP=2 | 6.4 |

大致结论：**双卡下 llama.cpp 明显比 vLLM 快约一倍**；vLLM 上开 MTP 反而更慢（无 P2P，草稿每轮要跨一次 PCIe 同步）。（表中 llama.cpp 那行是 8/14 用更轻的 Q4_K_XL 测的；Q5_K_XL 单请求提交输出约 46 tok/s。）

> 注：本条提到的 **Qwen3.8 UD-Q5_K_XL V3**（Unsloth Dynamic 3.0，2026-08-19 同尺寸精度略升、速度同 v2）也测过并短暂换装，随后整机切到 Ornith 未继续使用，模型文件保留可回滚。

---

MIT License.
