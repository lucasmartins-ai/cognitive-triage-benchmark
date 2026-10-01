#!/usr/bin/env python3
"""
Statistical Analysis and Metrics Calculation for Jev vs Gemini 3.8 Flash Benchmark.
Computes 95% CIs, McNemar's test, confusion matrices, latency distributions,
cost models, break-even curves, and workload splits.
"""

import os
import json
import math
import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = "/Users/Master/Downloads/LOOKAORCHESTRATOR/benchmark/results"
RAW_DATA_PATH = os.path.join(RESULTS_DIR, "benchmark_raw_data.csv")

def wilson_score_interval(successes, total, confidence=0.95):
    """Wilson score confidence interval for proportions."""
    if total == 0:
        return (0.0, 0.0)
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))

def bootstrap_ci(data, stat_func=np.median, n_resamples=2000, confidence=0.95):
    """Non-parametric bootstrap confidence interval."""
    data = np.array(data)
    boot_stats = [stat_func(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_resamples)]
    alpha = (1.0 - confidence) / 2.0
    return (float(np.percentile(boot_stats, 100 * alpha)), float(np.percentile(boot_stats, 100 * (1 - alpha))))

def mcnemar_test(y_true, y_pred1, y_pred2):
    """Paired McNemar's test comparing two classifiers against ground truth."""
    c1 = (y_pred1 == y_true)
    c2 = (y_pred2 == y_true)
    
    # b: clf1 correct, clf2 wrong
    b = int(np.sum(c1 & (~c2)))
    # c: clf1 wrong, clf2 correct
    c = int(np.sum((~c1) & c2))
    
    # Edwards continuity correction
    if (b + c) == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c}
    stat = (abs(b - c) - 1.0)**2 / (b + c)
    p_val = 1.0 - stats.chi2.cdf(stat, df=1)
    return {"chi2": float(stat), "p_value": float(p_val), "b": b, "c": c}

def calculate_macro_metrics(y_true, y_pred, classes=["approve", "reject", "escalate"]):
    """Precision, Recall, and F1 macro and per-class."""
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    
    for cls in classes:
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        tn = int(np.sum((y_pred != cls) & (y_true != cls)))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "tn": tn}
        
    return {
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1s)),
        "per_class": per_class
    }

def analyze():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Missing {RAW_DATA_PATH}. Run benchmark_suite.py first.")
        
    df = pd.read_csv(RAW_DATA_PATH)
    n = len(df)
    print(f"Analyzing {n} benchmark results...")
    
    y_true = df["ground_truth"].values
    should_esc = df["should_escalate"].values
    
    # Define architectures
    arch_configs = {
        "jev": {
            "name": "Jev Only",
            "pred": df["jev_decision"].values,
            "latencies": df["jev_latency_ms"].values,
            "cost": df["jev_cost"].values,
            "calls_gemini": np.zeros(n, dtype=bool)
        },
        "gem_low": {
            "name": "Gemini Low Thinking",
            "pred": df["gemini_low_decision"].values,
            "latencies": df["gemini_low_latency_ms"].values,
            "cost": df["gemini_low_cost"].values,
            "calls_gemini": np.ones(n, dtype=bool)
        },
        "gem_med": {
            "name": "Gemini Medium Thinking",
            "pred": df["gemini_med_decision"].values,
            "latencies": df["gemini_med_latency_ms"].values,
            "cost": df["gemini_med_cost"].values,
            "calls_gemini": np.ones(n, dtype=bool)
        },
        "hybrid_low": {
            "name": "Hybrid (Jev → Gemini Low)",
            "pred": df["hybrid_low_decision"].values,
            "latencies": df["hybrid_low_latency_ms"].values,
            "cost": df["hybrid_low_cost"].values,
            "calls_gemini": df["hybrid_low_escalated"].values
        },
        "hybrid_med": {
            "name": "Hybrid (Jev → Gemini Med)",
            "pred": df["hybrid_med_decision"].values,
            "latencies": df["hybrid_med_latency_ms"].values,
            "cost": df["hybrid_med_cost"].values,
            "calls_gemini": df["hybrid_med_escalated"].values
        }
    }
    
    # Oracle Hybrid calculation:
    # If Jev got it right -> keep Jev
    # If Jev got it wrong -> escalate to Gemini Med
    oracle_pred = np.where(df["jev_decision"].values == y_true, df["jev_decision"].values, df["gemini_med_decision"].values)
    oracle_calls_gemini = (df["jev_decision"].values != y_true)
    oracle_latencies = np.where(oracle_calls_gemini, df["jev_latency_ms"].values + df["gemini_med_latency_ms"].values, df["jev_latency_ms"].values)
    oracle_cost = np.where(oracle_calls_gemini, df["jev_cost"].values + df["gemini_med_cost"].values, df["jev_cost"].values)
    
    arch_configs["oracle"] = {
        "name": "Oracle Hybrid (Upper Bound)",
        "pred": oracle_pred,
        "latencies": oracle_latencies,
        "cost": oracle_cost,
        "calls_gemini": oracle_calls_gemini
    }
    
    summary = {}
    
    for key, cfg in arch_configs.items():
        y_pred = cfg["pred"]
        lats = cfg["latencies"]
        costs = cfg["cost"]
        gem_calls = cfg["calls_gemini"]
        
        correct_mask = (y_pred == y_true)
        correct_count = int(np.sum(correct_mask))
        accuracy = correct_count / n
        acc_ci = wilson_score_interval(correct_count, n)
        
        macro = calculate_macro_metrics(y_true, y_pred)
        
        # Asymmetric Error Cost:
        # FN: y_true == 'approve', y_pred == 'reject' (cost 10)
        # FP: y_true == 'reject', y_pred == 'approve' (cost 2)
        # Missed Escalation: should_esc == True, y_pred != 'escalate' (cost 5)
        fn_count = int(np.sum((y_true == "approve") & (y_pred == "reject")))
        fp_count = int(np.sum((y_true == "reject") & (y_pred == "approve")))
        em_count = int(np.sum(should_esc & (y_pred != "escalate")))
        weighted_error_cost = (fn_count * 10.0) + (fp_count * 2.0) + (em_count * 5.0)
        
        # Latency metrics
        mean_lat = float(np.mean(lats))
        std_lat = float(np.std(lats))
        p50_lat = float(np.percentile(lats, 50))
        p75_lat = float(np.percentile(lats, 75))
        p90_lat = float(np.percentile(lats, 90))
        p95_lat = float(np.percentile(lats, 95))
        p99_lat = float(np.percentile(lats, 99))
        min_lat = float(np.min(lats))
        max_lat = float(np.max(lats))
        p50_ci = bootstrap_ci(lats, np.median)
        
        # Cost metrics
        cost_sum = float(np.sum(costs))
        cost_per_decision = cost_sum / n
        cost_per_1k = cost_per_decision * 1000.0
        cost_per_10k = cost_per_decision * 10000.0
        cost_per_1m = cost_per_decision * 1000000.0
        cost_per_correct = cost_sum / correct_count if correct_count > 0 else 0.0
        
        # Escalation metrics
        escalated_count = int(np.sum(gem_calls))
        escalation_rate = escalated_count / n
        calls_avoided_pct = (1.0 - escalation_rate) * 100.0
        
        # Escalation breakdown
        appropriate_esc = int(np.sum(gem_calls & should_esc))
        unnecessary_esc = int(np.sum(gem_calls & (~should_esc)))
        missed_esc = int(np.sum((~gem_calls) & should_esc))
        
        # False Positive / Negative rates
        # Total true negatives (for approve): y_true != 'approve'
        neg_for_approve = int(np.sum(y_true != "approve"))
        fp_rate = fp_count / neg_for_approve if neg_for_approve > 0 else 0.0
        pos_for_approve = int(np.sum(y_true == "approve"))
        fn_rate = fn_count / pos_for_approve if pos_for_approve > 0 else 0.0
        esc_miss_rate = missed_esc / int(np.sum(should_esc)) if np.sum(should_esc) > 0 else 0.0
        
        summary[key] = {
            "name": cfg["name"],
            "accuracy": float(accuracy),
            "accuracy_ci_95": [float(acc_ci[0]), float(acc_ci[1])],
            "f1": float(macro["macro_f1"]),
            "precision": float(macro["macro_precision"]),
            "recall": float(macro["macro_recall"]),
            "per_class": macro["per_class"],
            "fn_count": fn_count,
            "fp_count": fp_count,
            "fp_rate": float(fp_rate),
            "fn_rate": float(fn_rate),
            "escalation_miss_rate": float(esc_miss_rate),
            "weighted_error_cost": float(weighted_error_cost),
            "latency": {
                "mean": mean_lat,
                "std": std_lat,
                "median": p50_lat,
                "p50": p50_lat,
                "p50_ci_95": [float(p50_ci[0]), float(p50_ci[1])],
                "p75": p75_lat,
                "p90": p90_lat,
                "p95": p95_lat,
                "p99": p99_lat,
                "min": min_lat,
                "max": max_lat
            },
            "cost_per_decision": float(cost_per_decision),
            "cost_per_1k": float(cost_per_1k),
            "cost_per_10k": float(cost_per_10k),
            "cost_per_1m": float(cost_per_1m),
            "cost_per_correct_decision": float(cost_per_correct),
            "escalation_rate": float(escalation_rate),
            "calls_avoided_pct": float(calls_avoided_pct),
            "appropriate_escalations": appropriate_esc,
            "unnecessary_escalations": unnecessary_esc,
            "missed_escalations": missed_esc
        }
        
    # Statistical significance comparisons (McNemar)
    mcnemar_results = {
        "hybrid_med_vs_gemini_med": mcnemar_test(y_true, arch_configs["hybrid_med"]["pred"], arch_configs["gem_med"]["pred"]),
        "hybrid_low_vs_gemini_low": mcnemar_test(y_true, arch_configs["hybrid_low"]["pred"], arch_configs["gem_low"]["pred"]),
        "jev_vs_gemini_med": mcnemar_test(y_true, arch_configs["jev"]["pred"], arch_configs["gem_med"]["pred"]),
        "hybrid_med_vs_jev": mcnemar_test(y_true, arch_configs["hybrid_med"]["pred"], arch_configs["jev"]["pred"]),
    }
    summary["mcnemar"] = mcnemar_results
    
    # Workload breakdown
    workload_summary = {}
    for wl in ["A", "B", "C"]:
        sub_df = df[df["workload_type"] == wl]
        wl_n = len(sub_df)
        wl_true = sub_df["ground_truth"].values
        workload_summary[wl] = {
            "n": wl_n,
            "jev_acc": float(np.mean(sub_df["jev_decision"].values == wl_true)),
            "gem_med_acc": float(np.mean(sub_df["gemini_med_decision"].values == wl_true)),
            "hybrid_med_acc": float(np.mean(sub_df["hybrid_med_decision"].values == wl_true)),
            "hybrid_med_esc_rate": float(np.mean(sub_df["hybrid_med_escalated"].values)),
            "gemini_calls_avoided_pct": float((1.0 - np.mean(sub_df["hybrid_med_escalated"].values)) * 100.0)
        }
    summary["workloads"] = workload_summary
    
    # Save metrics summary
    out_file = os.path.join(RESULTS_DIR, "metrics_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Metrics summary written to {out_file}")
    return summary

if __name__ == "__main__":
    analyze()
