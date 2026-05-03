---
layout: post
title: "Why my INT4 and INT8 KV cache quantization gave bitwise-identical perplexity"
date: 2026-05-04 14:00:00
description: When the standard sliding-window perplexity test produced identical numbers to fifteen decimals, the methodology was the bug, not the quantization.
tags: llm-inference quantization kv-cache benchmarking
categories: machine-learning
giscus_comments: false
related_posts: false
thumbnail: assets/img/posts/two-pass-perplexity/thumbnail.png
toc:
  sidebar: left
---

I quantized the KV cache of Llama 3.2-1B to INT4 and to INT8 separately, expecting a small but meaningful perplexity difference between them. Both runs completed. Both reported perplexity **8.591021385729238**. Identical to fifteen decimal places.

That's not a small numerical difference. That's the same number. Something was wrong with my benchmark.

## The setup

Standard sliding-window perplexity over the `WikiText-2` test split. For each window of 2048 tokens, run a forward pass with `past_key_values=fresh_cache`, compute cross-entropy on the last 512 positions, accumulate negative log-likelihood, divide by total tokens scored, exponentiate. About 560 windows, ~280K tokens scored.

Three cache configurations, otherwise identical:

- **FP16**: standard `DynamicCache` (no quantization).
- **INT8**: HuggingFace `QuantizedCache(backend="hqq", nbits=8, q_group_size=64, residual_length=64)`.
- **INT4**: HuggingFace `QuantizedCache(backend="hqq", nbits=4, q_group_size=64, residual_length=64)`.

Expected pattern: `FP16 ≈ INT8 < INT4`, with a small gap and a slightly larger one.

What I got:

| Variant | Perplexity |
|---|---|
| FP16 | 8.59093961953748 |
| INT8 | 8.591021385729238 |
| INT4 | 8.591021385729238 |

INT8 and INT4 are byte-identical. The FP16 vs quantized gap is ~0.0001 — a rounding error, not a quantization signal.

## The diagnosis

The realization came from thinking about *when* the cache is read vs. written during a forward pass.

In an autoregressive decode loop, the cache is load-bearing: at step *t*, the model computes Q for the current token, then attends over `[K_0, ..., K_{t-1}, K_t]` where the past `K_i` come from the cache. The cache is **read**. If those K values are quantized in storage, they get dequantized for the attention math, and quantization noise enters the logits.

In a single-pass forward over a 2048-token window with an empty starting cache:

1. Model computes Q, K, V for all 2048 input positions directly from the input tokens.
2. Attention runs using these freshly-computed Q, K, V tensors.
3. K, V get written to the cache as a side effect.

Step 3 is where quantization happens. But step 2, which determines the logits, which determine the perplexity uses the **un-cached, freshly-computed** K and V.

**The cache is write-only during a single forward pass.** Quantization affects what's stored. It does not affect what's used.

Hence INT8 and INT4 produce identical attention math (both backed by the same un-quantized K, V from the current input), identical logits, and identical perplexity to fifteen decimals.

## Why FP16 differs by a tiny amount

`DynamicCache` (the FP16 path) and `QuantizedCache` (INT8/INT4) are different Python classes with different `update()` implementations. Even when functionally equivalent for an empty starting cache, they take slightly different code paths — different intermediate tensor allocations, slightly different memory layouts, different order of operations. Floating-point non-associativity does the rest: the order of accumulations can shift the last bit.

The ~0.0001 gap between FP16 and quantized variants has nothing to do with quantization fidelity. It's just *"different class instance → different floating-point round-off."*

## The fix: two-pass evaluation

To exercise the quantized cache, you need a forward pass that *reads* it. Split each window into two passes:

```python
prefix = window[:, :L - score_len]   # 1536 tokens
suffix = window[:, L - score_len:]   # 512 tokens

cache = cache_factory()  # fresh QuantizedCache

# Pass 1: populate the cache (writes get quantized)
with torch.inference_mode():
    _ = model(prefix, past_key_values=cache, use_cache=True, logits_to_keep=1)
    output = model(suffix, past_key_values=cache, use_cache=True)

shift_logits = output.logits[:, :-1, :]
shift_labels = suffix[:, 1:]
loss = F.cross_entropy(
    shift_logits.float().reshape(-1, V),
    shift_labels.reshape(-1),
    reduction='sum',
)
```

In Pass 1, the cache fills up. HQQ's residual buffer flushes every 64 tokens, quantizing those 64 K and V vectors into the main quantized buffer. By the end of Pass 1, the cache holds 1536 tokens in quantized form.

In Pass 2, the suffix's 512 tokens have Q computed fresh from input. But attention needs K and V over the full sequence,meaning it reads the prefix's K and V back from the cache. For `QuantizedCache`, that read **dequantizes** INT4 or INT8 storage back to FP16 on the fly. The reconstructed values carry quantization noise, and the noise propagates into the suffix's attention scores and logits.

INT4's coarser quantization (16 levels vs INT8's 256) produces larger reconstruction error → larger perturbation in attention scores → measurably worse perplexity.

## The corrected numbers

After re-running with two-pass eval:

| Variant | Perplexity | Δ vs FP16 |
|---|---|---|
| FP16 | 8.5916 | — |
| INT8 | 8.5943 | +0.0027 (+0.03%) |
| INT4 | 9.0431 | +0.4515 (**+5.3%**) |

INT8 is essentially free: 0.03% perplexity for ~2× cache compression. 
INT4 pays a real cost: 5.3% perplexity for ~3.6× compression. 
**That's the actual tradeoff** that single-pass eval was hiding.

The numbers broadly agree with literature: KIVI's 2-bit *asymmetric* scheme on Llama-2 stays within ~0.1–0.3 perplexity of FP16. My naive symmetric uniform INT4 lands at +0.45, slightly worse than KIVI's 2-bit because KIVI's asymmetric K/V handling (per-channel K, per-token V) is specifically designed for cache distributions, where K has outliers and V is well-behaved.

## The general lesson

KV cache quantization affects inference **between** forward passes, not **within** a single one. To benchmark its effect on quality, your evaluation has to span at least two forwards that exchange a cache.

Standard sliding-window perplexity, as commonly written in the HuggingFace docs and most tutorials, is a single-pass workload. It's the right way to evaluate base model quality.

How to fix the methodology: 

**Two-pass eval per window**: Each window's cache is populated by a prefix forward, then evaluated by a suffix forward. Fast, simple, costs about one token of evaluation per window.

If you find yourself benchmarking *anything* that modifies the KV cache like quantization, sparsity, eviction, sliding window, check whether your evaluation actually exercises the cache or just writes to it. 

## References

- Liu, Z., et al. (2024). *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.*.
- Hooper, C., et al. (2024). *KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization.*.
- Badri, H., & Shaji, A. *HQQ: Half-Quadratic Quantization for Large Language Models.*.
- HuggingFace Transformers documentation: *Perplexity of fixed-length models.*
