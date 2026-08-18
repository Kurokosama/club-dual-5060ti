# 双 RTX 5060 Ti 本地跑 LLM

我自己这台机器上的配置和真实验证过的速度。有双 5060 Ti、想试本地大模型的朋友可以参考下，不过**数字是我这台机器跑出来的，抄完最好自己复测**。

This is my personal setup running local LLMs on **2× RTX 5060 Ti 16GB** — hardware, software, the launch command, and speeds I actually measured (not estimated) on this box.

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
| 运行时 | llama.cpp（CUDA），`ece-0.20.0` |
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
| 纯生成 decode | ~40.7 tok/s（持续单槽）|
| 端到端 | 36–57 tok/s |

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
| llama.cpp + MTP-3 | 61.3–67.8 ✅ |
| vLLM NVFP4（无 MTP）| 31.7 |
| vLLM NVFP4 + MTP-2 | 15.7 |
| SGLang PP=2 | 6.4 |

结论一句话：**双卡这块 llama.cpp 明显是 vLLM 的两倍**；而且 vLLM 上开 MTP 反而更慢（无 P2P，草稿每轮都要跨一次 PCIe 同步）。

## 踩过的坑

- 上游某个 commit（`51a4f6303`）把 spec 挪进 worker 线程后，我这个拓扑下 MTP 全被拒，decode 掉到 ~19 tok/s。上游还没修，改了以后能白捡 10–15%。
- tensor-split 下 KV 只能用 q4_0，换 q4_1/q5_0 直接崩（Exit 139）。
- `-p-min` 我测出来是负优化，别开。
- vLLM 开 MTP 也是负优化，见上。

---
MIT License.
