# 双 RTX 5060 Ti 本地跑 LLM

我自己这台机器上的配置和真实验证过的速度。有双 5060 Ti、想试本地大模型的朋友可以参考下，不过**数字是我这台机器跑出来的，抄完最好自己复测**。

This is my personal setup running local LLMs on **2× RTX 5060 Ti 16GB** — hardware, software, the launch command, and speeds I actually measured (not estimated), plus a running changelog at the top.

## 8/19 更新

一天之内从「跟着别人吹」换成「自己实测」，结论变化不小。核心一句话：**这台机器的配置已经从 Qwen3.8-27B 换成了 Ornith-1.5-35B-A3B（Q4_K_M），256K 原生上下文 + 视觉，并且 MTP 关掉了。**

### 2026-08-19 模型换代

| 项目 | 旧（Qwen3.8-27B） | 新（Ornith-1.5-35B-A3B） |
| --- | --- | --- |
| 架构 | Dense 27B | MoE（~3.5B 激活，QWEN35MOE） |
| 量化 | UD-Q5_K_XL | Q4_K_M（256K+视觉下的上限） |
| 上下文 | 192K（上限） | **256K 原生** |
| 视觉 | 有 | 有（mmproj，加 `--image-min-tokens 1024`） |
| 纯生成 | ~46 tok/s | **~109 tok/s** |

### 期间测过、但没换的
- **Qwen3.8-27B UD-Q5_K_XL V3**（Unsloth Dynamic 3.0）：同尺寸精度更高、速度同 v2，是实在的升级；但因为整机要换到 Ornith 就没继续用，文件留着可回滚。
- **DFlash2**（上游 PR #27342，block-diffusion 投机解码，就是有人吹"单卡 4090 90 tok/s"那套）：实测在这种 **tensor-split 双卡**上直接崩（`GGML_ASSERT ... SPLIT_AXIS_0`）；就算换成 layer 模式也≈MTP，不如纯自回归。营销数，别信。
- Ornith 高量化档：**Q5（25GB）在 256K+视觉下 OOM**，只能 256K 纯文本或视觉降 ctx；Q6 干脆装不下。**单卡 16GB 才是瓶颈（不是 32GB 总和）**——GPU0 还要扛词表嵌入 + 视觉编码器。

### MTP：实测在此机器上是负优化，必须关
上游 **#26750**（CUDA 后端 draft-mtp 接受率崩塌，CUDA 40.7% vs Vulkan 92%，同 GGUF 可复现）。实机 A/B（Ornith Q4@256K，同一批 prompt）：

| 配置 | 实测 tok/s | 接受率 |
| --- | ---: | ---: |
| MTP n_max=3 | ~73 | 0.24 |
| MTP n_max=4 | ~64 | 0.19 |
| MTP n_max=6 | ~49 | 0.12 |
| MTP n_max=4 + p_min=0.75 | ~43 | 0.83 |
| **纯自回归（关 MTP）** | **~109** | — |

MTP 在 CUDA 上不仅没提速，反而拖慢 ~1/3；调 n_max / p_min 都救不回。结论：**这台机器上关掉 spec、纯自回归就是天花板**（≈1.5× 于 MTP）。等上游修好 #26750 再复测。

### 当前启动方式
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
脚本版见 [`examples/llamacpp-ornith-35b-a3b-dual-5060ti.sh`](examples/llamacpp-ornith-35b-a3b-dual-5060ti.sh)。

---

以下是历史配置（Qwen3.8-27B），留档备查。要回滚就按这里的命令走。

## 硬件

| 部件 | 配置 |
| --- | --- |
| GPU | 2× NVIDIA GeForce RTX 5060 Ti 16GB（合计 32GB，无 P2P）|
| CPU | AMD Ryzen 5 4500（PCIe 3.0 only）|
| 主板 | MSI B550-A PRO |
| 电源 | Seasonic Focus GX-1000 1000W |
| 内存 | 64GB DDR4 |

测下来真正的瓶颈**不是 PCIe**，而是 **VRAM 带宽 + 跨卡延迟**（PCIe 利用率 <4%、单卡 117/180W、显存时钟顶满）。以前我也以为是卡槽带宽的问题，折腾半天发现白费。

## 软件

| 项 | 配置 |
| --- | --- |
| 运行时 | llama.cpp（CUDA，自编译）|
| 模型 | Qwen3.8-27B，量化 UD-Q5_K_XL |
| Tensor split | 50/50 |
| MTP | 3（内置草稿 token）|
| 上下文 | 192K（我这边测出来最稳的档）|
| KV | q4_0（tensor-split 下只能用这个）|
| effort | 固定 low |

> 备注：官方下载下来的 release 是 **CUDA 版**。你要是用 Vulkan 那套，decode 基本只能单卡，等于白瞎第二张卡——想用满双卡 tensor-split 得**自己编译 CUDA 版**（我这颗 Blackwell 用的 `sm_120`）。想省事就直接拿我给的命令，别纠结 prebuilt。

## 启动命令

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

脚本版在 [`examples/llamacpp-qwen38-27b-dual-5060ti.sh`](examples/llamacpp-qwen38-27b-dual-5060ti.sh)。

## 速度

| 指标 | 数值 |
| --- | --- |
| 纯生成 decode | ~70 tok/s（服务端 eval 计时实测）|
| 端到端 | 长输出 ~57（1000 token 实测）；短输出更低（prefill 占大头）|

> 温度声明一下：这套 CPU 是 PCIe3-only 的 4500，换个平台数字会变，别拿我的当标准答案。

### 备选：vLLM + NVFP4

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

我 8/14 在同样两台卡上跑过 A/B：

| 方案 | tok/s |
| --- | ---: |
| llama.cpp + MTP-3 | 61.3–67.8（8/14 用 Q4_K_XL 测）✅ |
| vLLM NVFP4（无 MTP）| 31.7 |
| vLLM NVFP4 + MTP-2 | 15.7 |
| SGLang PP=2 | 6.4 |

结论一句话：**双卡这块 llama.cpp 明显是 vLLM 的两倍**；而且 vLLM 上开 MTP 反而更慢（无 P2P，草稿每轮都要跨一次 PCIe 同步）。（表中 llama.cpp 那行是 8/14 用更轻的 Q4_K_XL 测的；当前 Q5_K_XL 单请求 decode 约 70。）

## 踩过的坑

- 上游某个 commit（`51a4f6303`）把 spec 挪进 worker 线程后，我这个拓扑下 MTP 全被拒，decode 掉到 ~19 tok/s。上游还没修，改了以后能白捡 10–15%。
- tensor-split 下 KV 只能用 q4_0，换 q4_1/q5_0 直接崩（Exit 139）。
- `-p-min` 我测出来是负优化，别开。
- vLLM 开 MTP 也是负优化，见上。

---
MIT License.
