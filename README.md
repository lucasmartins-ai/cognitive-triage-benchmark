# Decoupled Cognitive Triage: Evaluating TypeSafe System 1 (Jev) Against Google Gemini 3.8 Flash in Autonomous Production Orchestration

[![Technical Report](https://img.shields.io/badge/Preprint-LOOKADEV--TR--2026--004-cyan.svg)](PAPER.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LookADev Research](https://img.shields.io/badge/LookADev-Engineering_Research-blue.svg)](https://lookadev.com/en/devlog/decoupled-cognitive-triage-jev-vs-gemini)

This repository contains the complete, reproducible benchmark suite, frozen dataset ($N = 1,200$), raw telemetry, and academic preprint for **Decoupled Cognitive Triage**: evaluating dual-process cognitive architectures (TypeSafe System 1 / Jev + Google Gemini 3.8 Flash) in autonomous production systems.

---

## Abstract

Modern autonomous agent architectures suffer from severe over-allocation: every inbound event—routine webhooks, obvious spam, simple confirmations, and complex negotiations—is routed to multi-second frontier reasoning models like Google Gemini 3.8 Flash (`gemini-3.8-flash`). This incurs a 16-second median latency tax ($p_{50} = 16,240\text{ ms}$) and an inference cost of $27.19 per 1,000 requests.

Drawing from cognitive science's **Dual-Process Theory** (Kahneman, 2011), we evaluate a decoupled architecture using **TypeSafe System 1 (Jev)** as a pre-cognitive semantic triage layer ahead of Gemini 3.8 Flash. Across **1,200 frozen real-world business transactions**:

- **58.2% of Gemini 3.8 Flash invocations are safely eliminated** ($698/1,200$ calls resolved locally).
- **Zero statistically significant quality degradation** (Hybrid Macro F1: **75.7%** vs. Gemini Med: **74.0%**, McNemar $\chi^2 = 0.44, p = 0.507$; **102.25% baseline quality retention**).
- **Median system decision latency drops by 97.9%** (from $16,240\text{ ms}$ down to **$333.3\text{ ms}$**).
- **Direct inference costs fall by 58.0%** ($11.43 vs. $27.19 per 1K calls), saving **$15,761.26 per 1,000,000 production decisions**.

Read the full research paper at [PAPER.md](PAPER.md) or the executive deep-dive at [lookadev.com/en/devlog/decoupled-cognitive-triage-jev-vs-gemini](https://lookadev.com/en/devlog/decoupled-cognitive-triage-jev-vs-gemini).

---

## Dual-Process Architecture

```
                       [ Inbound Transaction / Event ]
                                      │
                                      ▼
             [ System 1: TypeSafe Jev (Sub-300ms, $0.075 / 1K) ]
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
      Confidence >= 0.78 & Risk <= 0.40       Ambiguous / High-Stakes
                 (58.2% of traffic)              (41.8% of traffic)
                     │                                 │
                     ▼                                 ▼
         [ Local Resolution ]              [ System 2: Gemini 3.8 Flash ]
         • 283.5 ms median latency         • Multi-step CoT reasoning
         • $0.000075 / request             • 16.2s median latency
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
                         [ Final Executed Action ]
```

---

## Empirical Benchmark Results ($N = 1,200$)

### 1. Latency & Throughput ($p_{50}$ to $p_{99}$)

| Pipeline Architecture | $p_{50}$ Latency | $p_{90}$ Latency | $p_{95}$ Latency | $p_{99}$ Latency | Max Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **TypeSafe System 1 (Jev)** | **283.5 ms** | **392.0 ms** | **452.1 ms** | **598.0 ms** | **842.0 ms** |
| **Gemini 3.8 Flash (Low Effort)** | 13,850.2 ms | 18,120.0 ms | 19,410.5 ms | 24,120.0 ms | 31,540.0 ms |
| **Gemini 3.8 Flash (Medium Effort)** | 16,240.1 ms | 21,340.0 ms | 22,870.0 ms | 27,950.4 ms | 34,810.0 ms |
| **Hybrid Dual-Process Architecture** | **333.3 ms** | 17,210.0 ms | 18,920.4 ms | 25,410.0 ms | 34,810.0 ms |

*Protocol: Latency was measured strictly in-memory between dispatch and typed structured JSON return, completely eliminating database, vector search, and network I/O conflation.*

### 2. Decision Quality & Baseline Retention

| Architecture | Raw Accuracy | Macro Precision | Macro Recall | Macro F1 | Baseline Retention |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gemini 3.8 Flash (Medium)** | 72.2% | 76.5% | 72.8% | 74.0% | 100.0% (Reference) |
| **Gemini 3.8 Flash (Low)** | 69.8% | 73.1% | 70.4% | 71.3% | 96.35% |
| **TypeSafe Jev (Standalone)** | 71.5% | 74.8% | 72.1% | 73.0% | 98.65% |
| **Hybrid Pipeline (Jev + Gemini Med)** | **75.8%** | **78.2%** | **74.6%** | **75.7%** | **102.25%** |

### 3. Economic Analysis & Break-Even Theorem

- **Jev Unit Cost**: **$0.000075** ($0.075 / 1K calls).
- **Gemini 3.8 Flash (Med) Unit Cost**: **$0.02719** ($27.19 / 1K calls).
- **Hybrid Cost per 1K calls**: **$11.43** (58.0% direct reduction).
- **Net Operational Savings**: **$15,761.26 per 1,000,000 decisions**.

The theoretical break-even escalation rate:

$$r_{\text{break-even}} = 1 - \frac{C_{\text{Jev}}}{C_{\text{Gemini}}} = 1 - \frac{0.000075}{0.02719} = 99.72\%$$

The hybrid pipeline is mathematically guaranteed to be cheaper than calling Gemini 3.8 Flash directly as long as the System 1 escalation rate remains below **99.72%**.

---

## Publication Figures

All 10 figures are available in high-resolution in [`charts/`](charts/):

| Figure | Description | File |
| :--- | :--- | :--- |
| **Fig 1** | Latency Distribution ($p_{50}, p_{90}, p_{95}, p_{99}$) | [`charts/01_latency_distribution.png`](charts/01_latency_distribution.png) |
| **Fig 2** | Latency Percentile Curves | [`charts/02_latency_percentiles.png`](charts/02_latency_percentiles.png) |
| **Fig 3** | Throughput vs Concurrency (1–100 workers) | [`charts/03_throughput_vs_concurrency.png`](charts/03_throughput_vs_concurrency.png) |
| **Fig 4** | Accuracy & Macro F1 Comparison | [`charts/04_accuracy_and_f1.png`](charts/04_accuracy_and_f1.png) |
| **Fig 5** | Cost per 1,000 Requests | [`charts/05_cost_per_1000_requests.png`](charts/05_cost_per_1000_requests.png) |
| **Fig 6** | Projected Cost per 1 Million Requests | [`charts/06_projected_cost_per_1m_requests.png`](charts/06_projected_cost_per_1m_requests.png) |
| **Fig 7** | Escalation Rate vs Cost Trade-off | [`charts/07_escalation_rate_vs_cost.png`](charts/07_escalation_rate_vs_cost.png) |
| **Fig 8** | Escalation Rate vs Quality Trade-off | [`charts/08_escalation_rate_vs_quality.png`](charts/08_escalation_rate_vs_quality.png) |
| **Fig 9** | Frontier LLM Calls Avoided (58.2%) | [`charts/09_llm_calls_avoided.png`](charts/09_llm_calls_avoided.png) |
| **Fig 10** | Cost vs Quality Pareto Frontier | [`charts/10_cost_vs_quality_pareto_frontier.png`](charts/10_cost_vs_quality_pareto_frontier.png) |

---

## Dataset Structure ($N = 1,200$)

The frozen dataset is located at [`data/dataset_1200.json`](data/dataset_1200.json). Each item follows this schema:

```json
{
  "id": "tx_0001",
  "category": "sales_high_intent",
  "difficulty": "normal",
  "text": "Hi Lucas, we are looking to replace our manual WhatsApp triage with an automated system. Our budget is around £5,000 and we want to start next week. Can we speak?",
  "ground_truth": {
    "action": "escalate_to_founder",
    "priority": "high",
    "is_security_risk": false
  }
}
```

Categories: `sales_high_intent`, `support_and_inquiry`, `spam_and_solicitation`, `security_and_injection`, `enterprise_bespoke`, `ambiguous_intent`, `compliance_and_edge`.

---

## Reproducing the Benchmark

```bash
# Clone the repository
git clone https://github.com/lucasmartins-ai/cognitive-triage-benchmark.git
cd cognitive-triage-benchmark

# Recompute metrics summary and statistical significance
python3 scripts/analyze_results.py

# Re-render all 10 publication charts
python3 scripts/generate_charts.py
```

---

## Citation

```bibtex
@techreport{martins2026cognitive_triage,
  author      = {Martins, Lucas},
  title       = {Decoupled Cognitive Triage: Evaluating TypeSafe System 1 (Jev) Against Google Gemini 3.8 Flash in Autonomous Production Orchestration},
  institution = {LookADev Research},
  number      = {LOOKADEV-TR-2026-004},
  month       = {October},
  year        = {2026},
  url         = {https://github.com/lucasmartins-ai/cognitive-triage-benchmark}
}
```

---

## License

MIT License. Copyright (c) 2026 Lucas Martins / LookADev.
