# AGENTS.md

Personal dual RTX 5060 Ti local LLM setup repo.

## Rules

- Keep claims measured, not estimated. If a number came from one machine, say that.
- Do not commit private IPs, API keys, bearer tokens, or personal paths.
- Sanitize anything environment-identifying.

## README convention — date + model changelog

The README is a **reverse-chronological log keyed by “date · model”** (e.g. `## 8/19 · Ornith-1.5-35B-A3B`). Newest entry goes at the top of the log, right after the `## 软件` section. Each entry is self-contained and follows this shape:

```markdown
## <MM/DD> · <Model Name>

**模型**：<model + quant + arch/param notes>
**Hugging Face**：<link to the GGUF/model repo>

**干了啥**：<one or two lines on why/what changed>

**启动命令**：
```bash
<full llama-server command>
```
脚本版：[`examples/<...>.sh`](examples/<...>.sh)

**速度**（实测）：<table of measured tok/s; always same measurement method>

**优化**：<what was optimized, with measured before/after>

**遇到的坑**：<bulleted pitfalls>
```

- Always use the **same tok/s measurement method across entries** so numbers are comparable. Committed (content+reasoning) output speed, client-measured post-prefill. If an older number used a different method (e.g. server-side eval counting speculative draft tokens, which inflates it), annotate it as non-comparable rather than mixing methods.
- Model file paths go in as `/path/to/…` — never real local paths.
- Recording a speed/claim requires it was actually measured on this machine; label which date/quant if it differs from the live one.
- New tests / new models are new dated entries, not edits to old ones (except fixing a factual error).
