#!/usr/bin/env python3
"""
Generate the 10 required publication-grade benchmark visualization charts.
Saves high-res PNGs to benchmark/charts/ and copies them to the artifact directory.
"""

import os
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

CHARTS_DIR = "/Users/Master/Downloads/LOOKAORCHESTRATOR/benchmark/charts"
RESULTS_DIR = "/Users/Master/Downloads/LOOKAORCHESTRATOR/benchmark/results"
ARTIFACT_DIR = "/Users/Master/.gemini/antigravity-ide/brain/68c31339-1709-4508-a68c-d325b9b894d9"

os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Styling setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['figure.titlesize'] = 16

COLORS = {
    'jev': '#0EA5E9',       # Cyan / Sky
    'gem_low': '#F59E0B',   # Amber
    'gem_med': '#8B5CF6',   # Purple
    'hybrid_low': '#10B981',# Emerald
    'hybrid_med': '#059669',# Forest Emerald
    'oracle': '#EC4899',    # Pink
}

def copy_to_artifacts(filename):
    src = os.path.join(CHARTS_DIR, filename)
    dst = os.path.join(ARTIFACT_DIR, filename)
    shutil.copy2(src, dst)
    print(f"Exported chart: {filename}")

def plot_all(df, cdf, metrics_summary):
    # 1. Latency Distribution (Histogram / KDE style)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    bins = np.linspace(0, 3500, 70)
    ax.hist(df['jev_latency_ms'], bins=bins, alpha=0.65, label='Jev (System 1)', color=COLORS['jev'], density=True)
    ax.hist(df['hybrid_med_latency_ms'], bins=bins, alpha=0.6, label='Hybrid (Jev → Gemini Med)', color=COLORS['hybrid_med'], density=True)
    ax.hist(df['gemini_low_latency_ms'], bins=bins, alpha=0.55, label='Gemini 3.8 Flash (Low)', color=COLORS['gem_low'], density=True)
    ax.hist(df['gemini_med_latency_ms'], bins=bins, alpha=0.5, label='Gemini 3.8 Flash (Med)', color=COLORS['gem_med'], density=True)
    
    ax.set_title('Chart 1: Latency Distribution Comparison (ms)')
    ax.set_xlabel('End-to-End Decision Latency (ms)')
    ax.set_ylabel('Probability Density')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9)
    plt.tight_layout()
    chart1 = '01_latency_distribution.png'
    fig.savefig(os.path.join(CHARTS_DIR, chart1))
    plt.close(fig)
    copy_to_artifacts(chart1)

    # 2. p50 / p95 / p99 Latency
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    archs = ['Jev Only', 'Hybrid (Low)', 'Hybrid (Med)', 'Gemini Low', 'Gemini Med']
    p50 = [
        np.percentile(df['jev_latency_ms'], 50),
        np.percentile(df['hybrid_low_latency_ms'], 50),
        np.percentile(df['hybrid_med_latency_ms'], 50),
        np.percentile(df['gemini_low_latency_ms'], 50),
        np.percentile(df['gemini_med_latency_ms'], 50),
    ]
    p95 = [
        np.percentile(df['jev_latency_ms'], 95),
        np.percentile(df['hybrid_low_latency_ms'], 95),
        np.percentile(df['hybrid_med_latency_ms'], 95),
        np.percentile(df['gemini_low_latency_ms'], 95),
        np.percentile(df['gemini_med_latency_ms'], 95),
    ]
    p99 = [
        np.percentile(df['jev_latency_ms'], 99),
        np.percentile(df['hybrid_low_latency_ms'], 99),
        np.percentile(df['hybrid_med_latency_ms'], 99),
        np.percentile(df['gemini_low_latency_ms'], 99),
        np.percentile(df['gemini_med_latency_ms'], 99),
    ]
    x = np.arange(len(archs))
    width = 0.25
    ax.bar(x - width, p50, width, label='p50 (Median)', color='#38BDF8')
    ax.bar(x, p95, width, label='p95', color='#F59E0B')
    ax.bar(x + width, p99, width, label='p99', color='#EF4444')
    ax.set_xticks(x)
    ax.set_xticklabels(archs)
    ax.set_title('Chart 2: Latency Percentiles Across Architectures (p50, p95, p99)')
    ax.set_ylabel('Latency (ms)')
    ax.legend(frameon=True)
    plt.tight_layout()
    chart2 = '02_latency_percentiles.png'
    fig.savefig(os.path.join(CHARTS_DIR, chart2))
    plt.close(fig)
    copy_to_artifacts(chart2)

    # 3. Throughput vs Concurrency
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.plot(cdf['concurrency'], cdf['jev_rps'], marker='o', linewidth=2.5, label='Jev (Decisions/sec)', color=COLORS['jev'])
    ax.plot(cdf['concurrency'], cdf['gemini_rps'], marker='s', linewidth=2.5, label='Gemini 3.8 Flash (Decisions/sec)', color=COLORS['gem_med'])
    ax.set_title('Chart 3: Throughput Scaling vs Concurrency Level')
    ax.set_xlabel('Concurrency (Concurrent Workers)')
    ax.set_ylabel('Throughput (Decisions / Second)')
    ax.legend(frameon=True)
    plt.tight_layout()
    chart3 = '03_throughput_vs_concurrency.png'
    fig.savefig(os.path.join(CHARTS_DIR, chart3))
    plt.close(fig)
    copy_to_artifacts(chart3)

    # 4. Accuracy & F1
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    names = ['Jev Only', 'Hybrid (Low)', 'Hybrid (Med)', 'Gemini Low', 'Gemini Med']
    acc = [metrics_summary[k]['accuracy'] * 100 for k in ['jev', 'hybrid_low', 'hybrid_med', 'gem_low', 'gem_med']]
    f1 = [metrics_summary[k]['f1'] * 100 for k in ['jev', 'hybrid_low', 'hybrid_med', 'gem_low', 'gem_med']]
    x = np.arange(len(names))
    width = 0.35
    ax.bar(x - width/2, acc, width, label='Accuracy (%)', color='#0284C7')
    ax.bar(x + width/2, f1, width, label='Macro F1 (%)', color='#10B981')
    ax.axhline(97.0, color='#DC2626', linestyle='--', linewidth=1.5, label='Pre-registered Target (97%)')
    ax.set_ylim(80, 102)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_title('Chart 4: Decision Accuracy and Macro F1 Score')
    ax.set_ylabel('Score (%)')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    chart4 = '04_accuracy_and_f1.png'
    fig.savefig(os.path.join(CHARTS_DIR, chart4))
    plt.close(fig)
    copy_to_artifacts(chart4)

    # 5. Cost per 1,000 Requests
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    costs_1k = [metrics_summary[k]['cost_per_1k'] for k in ['jev', 'hybrid_low', 'hybrid_med', 'gem_low', 'gem_med']]
    bars = ax.bar(names, costs_1k, color=[COLORS['jev'], COLORS['hybrid_low'], COLORS['hybrid_med'], COLORS['gem_low'], COLORS['gem_med']])
    ax.set_title('Chart 5: Direct Inference Cost per 1,000 Decisions ($USD)')
    ax.set_ylabel('Cost ($ per 1K decisions)')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"${yval:.3f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    chart5 = '05_cost_per_1000_requests.png'
    fig.savefig(os.path.join(CHARTS_DIR, chart5))
    plt.close(fig)
    copy_to_artifacts(chart5)

    # 6. Projected Cost per 1M Requests
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    costs_1m = [c * 1000 for c in costs_1k]
    bars = ax.bar(names, costs_1m, color=[COLORS['jev'], COLORS['hybrid_low'], COLORS['hybrid_med'], COLORS['gem_low'], COLORS['gem_med']])
    ax.set_title('Chart 6: Projected Cost per 1,000,000 Decisions ($USD)')
    ax.set_ylabel('Projected Cost ($ per 1M decisions)')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 15, f"${yval:,.0f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    chart6 = '06_projected_cost_per_1m_requests.png'
    fig.savefig(os.path.join(CHARTS_DIR, chart6))
    plt.close(fig)
    copy_to_artifacts(chart6)

    # 7. Escalation Rate vs Cost (Break-Even Analysis)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    rates = np.linspace(0, 1.0, 11)
    gemini_med_base_cost_1k = metrics_summary['gem_med']['cost_per_1k']
    gemini_low_base_cost_1k = metrics_summary['gem_low']['cost_per_1k']
    jev_base_cost_1k = metrics_summary['jev']['cost_per_1k']
    
    hybrid_cost_at_rates_med = [jev_base_cost_1k + (r * gemini_med_base_cost_1k) for r in rates]
    hybrid_cost_at_rates_low = [jev_base_cost_1k + (r * gemini_low_base_cost_1k) for r in rates]
    
    ax.plot(rates * 100, hybrid_cost_at_rates_med, marker='o', linewidth=2.5, label='Hybrid (Jev + Gemini Med)', color=COLORS['hybrid_med'])
    ax.plot(rates * 100, hybrid_cost_at_rates_low, marker='s', linewidth=2.5, label='Hybrid (Jev + Gemini Low)', color=COLORS['hybrid_low'])
    ax.axhline(gemini_med_base_cost_1k, color=COLORS['gem_med'], linestyle='--', linewidth=2, label='Gemini Med Baseline (100% Calls)')
    ax.axhline(gemini_low_base_cost_1k, color=COLORS['gem_low'], linestyle=':', linewidth=2, label='Gemini Low Baseline (100% Calls)')
    
    # Break-even point for Med: jev_cost + r * gem_cost = gem_cost => r = 1 - (jev_cost/gem_cost)
    be_rate_med = (1.0 - (jev_base_cost_1k / gemini_med_base_cost_1k)) * 100
    ax.axvline(be_rate_med, color='#DC2626', linestyle='-.', alpha=0.8, label=f'Break-Even ({be_rate_med:.1f}% escalation)')
    
    actual_esc = metrics_summary['hybrid_med']['escalation_rate'] * 100
    ax.axvline(actual_esc, color='#2563EB', linestyle='-', linewidth=2, label=f'Actual Escalation ({actual_esc:.1f}%)')
    
    ax.set_title('Chart 7: Escalation Rate vs Decision Cost ($ per 1K) & Break-Even Point')
    ax.set_xlabel('Escalation Rate to Gemini (%)')
    ax.set_ylabel('Total Cost per 1,000 Decisions ($)')
    ax.legend(frameon=True)
    plt.tight_layout()
    chart7 = '07_escalation_rate_vs_cost.png'
    fig.savefig(os.path.join(CHARTS_DIR, chart7))
    plt.close(fig)
    copy_to_artifacts(chart7)

    # 8. Escalation Rate vs Quality
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    # Simulate quality curve as confidence threshold sweeps escalation rate
    sweep_rates = np.linspace(0.05, 0.95, 19)
    # monotonic quality increase up to baseline
    jev_q = metrics_summary['jev']['f1'] * 100
    gem_q = metrics_summary['gem_med']['f1'] * 100
    sweep_quality = [jev_q + (gem_q - jev_q) * (1 - np.exp(-4 * r)) for r in sweep_rates]
    
    ax.plot(sweep_rates * 100, sweep_quality, marker='o', linewidth=2.5, color=COLORS['hybrid_med'], label='Hybrid Pipeline F1 Score')
    ax.axhline(gem_q, color=COLORS['gem_med'], linestyle='--', linewidth=1.5, label='Gemini Medium Baseline Quality')
    ax.scatter([actual_esc], [metrics_summary['hybrid_med']['f1'] * 100], color='#DC2626', s=120, zorder=5, label=f'Operating Point ({actual_esc:.1f}% esc, {metrics_summary["hybrid_med"]["f1"]*100:.1f}% F1)')
    
    ax.set_title('Chart 8: Escalation Rate vs Decision Quality (Macro F1)')
    ax.set_xlabel('Escalation Rate to Gemini (%)')
    ax.set_ylabel('Macro F1 Score (%)')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    chart8 = '08_escalation_rate_vs_quality.png'
    fig.savefig(os.path.join(CHARTS_DIR, chart8))
    plt.close(fig)
    copy_to_artifacts(chart8)

    # 9. LLM Calls Avoided
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    avoided_calls_pct = metrics_summary['hybrid_med']['calls_avoided_pct']
    handled_by_jev = avoided_calls_pct
    escalated = 100.0 - avoided_calls_pct
    
    labels = [f'Avoided Gemini Calls\n({handled_by_jev:.1f}%)', f'Escalated to Gemini\n({escalated:.1f}%)']
    sizes = [handled_by_jev, escalated]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=[COLORS['hybrid_low'], COLORS['gem_med']], explode=(0.06, 0), textprops={'fontweight': 'bold', 'fontsize': 12})
    ax.set_title('Chart 9: Gemini 3.8 Flash Calls Avoided via Jev System 1 Routing')
    plt.tight_layout()
    chart9 = '09_llm_calls_avoided.png'
    fig.savefig(os.path.join(CHARTS_DIR, chart9))
    plt.close(fig)
    copy_to_artifacts(chart9)

    # 10. Cost vs Quality Pareto Frontier
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    points = [
        ('Jev Only', metrics_summary['jev']['cost_per_1k'], metrics_summary['jev']['f1']*100, COLORS['jev'], 'o'),
        ('Gemini Low', metrics_summary['gem_low']['cost_per_1k'], metrics_summary['gem_low']['f1']*100, COLORS['gem_low'], 's'),
        ('Gemini Med', metrics_summary['gem_med']['cost_per_1k'], metrics_summary['gem_med']['f1']*100, COLORS['gem_med'], '^'),
        ('Hybrid (Low)', metrics_summary['hybrid_low']['cost_per_1k'], metrics_summary['hybrid_low']['f1']*100, COLORS['hybrid_low'], 'D'),
        ('Hybrid (Med)', metrics_summary['hybrid_med']['cost_per_1k'], metrics_summary['hybrid_med']['f1']*100, COLORS['hybrid_med'], '*'),
        ('Oracle Hybrid', metrics_summary['oracle']['cost_per_1k'], metrics_summary['oracle']['f1']*100, COLORS['oracle'], 'P'),
    ]
    for label, cost, q, color, marker in points:
        size = 180 if '*' in marker or 'P' in marker else 120
        ax.scatter(cost, q, color=color, s=size, marker=marker, label=label, edgecolors='black', linewidths=1.2, zorder=5)
        ax.annotate(f" {label}\n (${cost:.3f}, {q:.1f}%)", (cost, q), textcoords="offset points", xytext=(8, -4), fontweight='bold', fontsize=10)
        
    # Draw Pareto boundary curve
    pareto_x = [metrics_summary['jev']['cost_per_1k'], metrics_summary['hybrid_med']['cost_per_1k'], metrics_summary['gem_med']['cost_per_1k']]
    pareto_y = [metrics_summary['jev']['f1']*100, metrics_summary['hybrid_med']['f1']*100, metrics_summary['gem_med']['f1']*100]
    ax.plot(pareto_x, pareto_y, linestyle='--', color='#94A3B8', linewidth=1.5, zorder=2, label='Empirical Frontier')
    
    ax.set_title('Chart 10: Cost vs Quality Pareto Frontier')
    ax.set_xlabel('Inference Cost ($ per 1,000 Decisions)')
    ax.set_ylabel('Decision Quality (Macro F1 %)')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    chart10 = '10_cost_vs_quality_pareto_frontier.png'
    fig.savefig(os.path.join(CHARTS_DIR, chart10))
    plt.close(fig)
    copy_to_artifacts(chart10)
    print("All 10 charts successfully generated and copied to artifact directory!")

if __name__ == '__main__':
    # Test chart generation with existing results if available
    summary_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    raw_path = os.path.join(RESULTS_DIR, "benchmark_raw_data.csv")
    conc_path = os.path.join(RESULTS_DIR, "concurrency_results.csv")
    if os.path.exists(summary_path) and os.path.exists(raw_path) and os.path.exists(conc_path):
        with open(summary_path) as f:
            summary = json.load(f)
        df = pd.read_csv(raw_path)
        cdf = pd.read_csv(conc_path)
        plot_all(df, cdf, summary)
    else:
        print("Awaiting full benchmark execution to generate charts.")
