#!/usr/bin/env python3
import os
import json
import numpy as np

EXPORTS_DIR = "ME2 - Voice Command Model/exports/v3_94class"

def compute_summary():
    seeds = [42, 1337, 2026]
    models = ["bcresnet", "dscnn"]
    summary = {}

    for m in models:
        m_seeds = seeds if m == "bcresnet" else [42]
        runs_star = []
        runs_bal = []
        for s in m_seeds:
            path = os.path.join(EXPORTS_DIR, f"eval_{m}_94class_seed{s}.json")
            d = json.load(open(path))
            runs_star.append(d["test_metrics_tau_star"])
            runs_bal.append(d["test_metrics_tau_bal"])

        def aggregate(run_list):
            cmd_accs = [r["command_level_94"]["accuracy_pct"] for r in run_list]
            int_accs = [r["intent_level_19"]["accuracy_pct"] for r in run_list]
            slot_accs = [r["slot_exact_match"]["exact_match_pct"] for r in run_list]
            far_ooss = [r["pipeline_rates"]["far_oos_speech_pct"] for r in run_list]
            frr_ins = [r["pipeline_rates"]["frr_in_scope_pct"] for r in run_list]
            acc_on_acc = [r["pipeline_rates"]["command_acc_on_accepted_pct"] for r in run_list]
            fil_accs = [r["voice_type_breakdown"]["real_filipino"]["raw_command_acc_pct"] for r in run_list]
            fil_frrs = [r["voice_type_breakdown"]["real_filipino"]["frr_pct"] for r in run_list]
            tau = run_list[0]["pipeline_rates"]["tau"]
            return {
                "tau": tau,
                "command_accuracy_mean": round(float(np.mean(cmd_accs)), 2),
                "command_accuracy_std": round(float(np.std(cmd_accs)), 2),
                "intent_accuracy_mean": round(float(np.mean(int_accs)), 2),
                "intent_accuracy_std": round(float(np.std(int_accs)), 2),
                "slot_exact_match_mean": round(float(np.mean(slot_accs)), 2),
                "slot_exact_match_std": round(float(np.std(slot_accs)), 2),
                "far_oos_mean": round(float(np.mean(far_ooss)), 2),
                "far_oos_std": round(float(np.std(far_ooss)), 2),
                "frr_in_scope_mean": round(float(np.mean(frr_ins)), 2),
                "frr_in_scope_std": round(float(np.std(frr_ins)), 2),
                "command_acc_on_accepted_mean": round(float(np.mean(acc_on_acc)), 2),
                "command_acc_on_accepted_std": round(float(np.std(acc_on_acc)), 2),
                "filipino_command_acc_mean": round(float(np.mean(fil_accs)), 2),
                "filipino_command_acc_std": round(float(np.std(fil_accs)), 2),
                "filipino_frr_mean": round(float(np.mean(fil_frrs)), 2),
                "filipino_frr_std": round(float(np.std(fil_frrs)), 2),
            }

        summary[m] = {
            "seeds": m_seeds,
            "strict_operating_point": aggregate(runs_star),
            "balanced_operating_point": aggregate(runs_bal),
        }

    with open(os.path.join(EXPORTS_DIR, "multi_seed_summary_94class.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Saved multi_seed_summary_94class.json successfully.")

if __name__ == "__main__":
    compute_summary()
