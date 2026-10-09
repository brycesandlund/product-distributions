# Inference time product distributions literature review

Literature reviewed through 2026-10-09.

Frozen language models can be combined at inference time to improve benchmark
performance, without distilling the ensemble or updating its members. Actual
product distributions have direct precedents, including cross-model logit fusion
and same-model context ensembling. There are also positive results combining
frontier-at-the-time 70B/72B models, although the clearest large-model studies use
**arithmetic probability averaging rather than products**.

The narrower question remains less established: can products of distinct,
current-frontier open-weight models consistently beat the strongest constituent
on difficult reasoning and coding tasks, particularly against compute-matched
alternatives? The works below do not settle that question. This is a targeted
review, not an exhaustive claim that such experiments do not exist.

## What counts as a product

For models with compatible token spaces and next-token logits $z_A,z_B$, the
weighted product used in this project is

$$
\mu_\alpha(y_t\mid h_t)
=\operatorname{softmax}\big((1-\alpha)z_A(h_t)+\alpha z_B(h_t)\big)_{y_t}
\propto p_A(y_t\mid h_t)^{1-\alpha}p_B(y_t\mid h_t)^\alpha.
$$

Averaging logits therefore implements a normalized geometric mean. Averaging
probabilities, $(1-\alpha)p_A+\alpha p_B$, implements a mixture instead. Voting,
reranking completed answers, and merging model weights are different operations.
With heterogeneous tokenizers, the alignment procedure is part of the method;
raw token IDs cannot simply be combined.

## Direct product distribution precedents

### PackLLM

[Pack of LLMs: Model Fusion at Test-Time via Perplexity Optimization](https://arxiv.org/html/2404.11531)
(COLM 2024) combines frozen models' next-token **logits**, choosing weights from
input-prompt perplexity. The fusion does not require model-weight training.
Downstream experiments include Llama-2-7B, Mistral-7B, Phi-2, and specialist
models on general knowledge, commonsense, and medical tasks.

PackLLM reports improvements of **1.72–1.89 percentage points over competing
test-time fusion baselines**, averaged over 25 tasks. These are not uniformly
gains over the strongest individual model. Different vocabularies are aligned
using minimum-edit-distance token mappings, so cross-family fusion is approximate.
This is direct prior art for inference-time products, but not a modern
frontier-scale reasoning study.

### Ensembling Language Models with Sequential Monte Carlo

[Ensembling Language Models with Sequential Monte Carlo](https://arxiv.org/html/2603.05432v1)
(March 2026) compares product, minimum, mixture, and maximum ensembles. Experiments
use **Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct, and Phi-4-14B**, both across
models and across prompts of one model. Consensus-seeking products and minima
generally outperform probability averaging, although products are not always best.

The paper distinguishes locally normalized token products from a globally
normalized product of complete-sequence probabilities. These distributions differ
because local normalization introduces prefix-dependent factors. Sequential Monte
Carlo (SMC) targets the global ensemble and supports heterogeneous tokenizations;
local methods are also evaluated.

Evaluation uses **100 examples per task** for JSON-schema generation, BBH word
sorting, and SPIDER text-to-SQL, with particle-based expected-accuracy estimates.
This provides useful mechanistic evidence, not a broad frontier benchmark result.

### ThinkMerge

[Think in Parallel, Answer as One: Logit Averaging for Open-Ended Reasoning](https://arxiv.org/html/2512.02874)
(December 2025) samples independent reasoning traces, then generates one shared
answer by averaging next-token logits across their contexts. Each selected answer
token is fed back into every context. The model remains frozen.

Selected LiveCodeBench overall pass@1 results from Table 4:

| Model | Single-pass baseline | Best reported merge setting |
| --- | ---: | ---: |
| DeepCoder-14B-Preview | 55.32% | 61.09% |
| Qwen3-8B | 57.14% | 59.57% |
| Qwen3-Think-30A3B | 69.30% | 72.04% |

These are best settings within a sweep; other settings sometimes underperform
baseline. The paper also evaluates AIME and GPQA. This is particularly close to
the project's multiple-context sampler, but it merges **answer generation after
independent reasoning**, not necessarily the entire reasoning trajectory. It is
not evidence of combining distinct frontier models.

### SAFE

[When to Ensemble: Identifying Token-Level Points for Stable and Fast LLM Ensembling](https://arxiv.org/html/2510.15346)
(ICLR 2026) introduces selective, tokenizer-aware fusion. Its geometric-mean
ablation explicitly replaces arithmetic averaging with a product-like consensus
operation on aligned distributions.

For **InternLM3-8B-Instruct + Qwen2.5-7B-Instruct**, Table 5 reports:

| Benchmark | Better individual model | Geometric-mean ensemble |
| --- | ---: | ---: |
| MMLU-redux | 76.89% | 78.31% |
| MATH500 | 74.8% | 77.6% |
| GSM8K | 91.81% | 92.27% |

This is selective fusion, not an unmodified product at every token. The negative
result is also important: ordinary every-token UniTE ensembling scores **59.6%
on MATH500**, below both constituents. The authors identify tokenizer-boundary
mismatches as a major source of long-generation instability. That mechanism does
not automatically explain failures in this project's compatible-tokenizer Qwen
experiments.

## Large model evidence from adjacent ensemble methods

### GaC

[Breaking the Ceiling of the LLM Community by Treating Token Generation as a Classification for Ensembling](https://aclanthology.org/2024.findings-emnlp.99.pdf)
(Findings of EMNLP 2024) explicitly ensembles leading open models available at
successive release dates. Its **Qwen2-72B + Llama-3-70B** combination beats both
constituents on all five reported benchmarks in Table 5:

| Model | MMLU | GSM8K | BBH | TriviaQA | NQ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Llama-3-70B-Instruct | 79.68 | 90.00 | 57.13 | 79.12 | 35.57 |
| Qwen2-72B-Instruct | 82.30 | 89.70 | 62.57 | 73.58 | 33.11 |
| Ensemble | 83.54 | 90.91 | 63.99 | 79.29 | 37.65 |

Scores are percentages. The operation is **arithmetic averaging of aligned
probabilities**, not logit averaging. Combining Llama-3-70B with the weaker
Qwen1.5-72B instead slightly reduces average performance. This supports the value
of complementary, comparably capable models, not indiscriminate ensembling.

### UniTE and DeePEn

- [Determine-Then-Ensemble: Necessity of Top-k Union for Large Language Model Ensembling](https://arxiv.org/html/2410.03777)
  (UniTE, ICLR 2025) combines **Qwen1.5-72B + Mixtral-8×7B**, reaching
  **78.84% ARC-C / 81.88% PIQA**, versus the better constituent's
  **73.98% / 74.59%**. It aggregates aligned probabilities rather than implementing
  the project's literal full-vocabulary product.
- [Ensemble Learning for Heterogeneous Large Language Models with Deep Parallel Collaboration](https://proceedings.neurips.cc/paper_files/paper/2024/file/d8a6eb79f8ccaacbe7198a5caf3a0323-Paper-Conference.pdf)
  (DeePEn, NeurIPS 2024) aligns distributions through relative representations.
  **Llama-2-70B + Mixtral-8×7B** improves GSM8K from the better constituent's
  **65.73% to 67.33%**, and PIQA from **71.88% to 75.10%**. This is another
  large-model ensemble precedent, not a clean product-distribution experiment.

An older composition framework,
[Controlled Text Generation via Language Model Arithmetic](https://arxiv.org/abs/2311.14479)
(ICLR 2024), combines models, prompts, and classifiers without retraining. Its
emphasis is controllable generation and toxicity reduction rather than the
general reasoning ceiling.

## Implications for this project

Inference-time products are established prior art. The more specific opportunity
is a careful evaluation of products between strong, complementary models, or
between useful contexts of one model. Published ensemble gains do not establish
that product rollouts will subsequently train a better student.

A discriminating inference-only experiment should compare:

1. Each constituent alone, with the stronger model as the main baseline.
2. Arithmetic probability mixtures and geometric products, with separately tuned
   mixing weights and temperatures on a development split.
3. Compute-matched independent sampling and voting or reranking where applicable.

Keep evaluation prompts, answer extraction, and stopping rules consistent; report
accuracy, response lengths, cap rates, and total compute. Products can suppress
tokens favored by the stronger model when the weaker model assigns them low
probability, so the project's 0.8B × 9B pairing addresses a different question
from combining two comparably capable models. Tokenizer alignment and local versus
global sequence normalization should be explicit parts of any comparison.
