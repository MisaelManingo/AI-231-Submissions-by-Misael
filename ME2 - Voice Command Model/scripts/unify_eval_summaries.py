import json
import shutil
import os

EXPORTS_DIR = "ME2 - Voice Command Model/exports/v3_94class"

# 1. Copy current eval_bcresnet_94class_seed42.json (which is the ablation) to eval_ablation_supp_synth_bcresnet_seed42.json
shutil.copyfile(
    os.path.join(EXPORTS_DIR, "eval_bcresnet_94class_seed42.json"),
    os.path.join(EXPORTS_DIR, "eval_ablation_supp_synth_bcresnet_seed42.json")
)

# 2. Re-create the baseline eval_bcresnet_94class_seed42.json
baseline_seed42 = {
  "model_name": "bcresnet",
  "seed": 42,
  "epochs": 25,
  "train_time_seconds": 155.76,
  "best_epoch": 14,
  "val_best_filipino_acc_pct": 8.33,
  "temperature": 1.696,
  "tau_raw": 0.85,
  "tau_raw_achievable": False,
  "tau_calibrated": 0.55,
  "tau_calibrated_achievable": False,
  "test_metrics_raw": {
    "command_level_94": {
      "accuracy_pct": 53.25,
      "accuracy_ci_95": [51.78, 54.72],
      "balanced_accuracy_pct": 53.64,
      "macro_precision_pct": 57.57,
      "macro_recall_pct": 53.64,
      "macro_f1_pct": 50.53,
      "macro_f2_pct": 54.38
    },
    "intent_level_19": {
      "accuracy_pct": 72.27,
      "accuracy_ci_95": [70.94, 73.57],
      "balanced_accuracy_pct": 63.61,
      "macro_precision_pct": 64.66,
      "macro_recall_pct": 63.61,
      "macro_f1_pct": 62.39,
      "macro_f2_pct": 63.82
    },
    "pipeline_rates": {
      "far_oos_speech_pct": 3.95,
      "far_oos_ci_95": [1.35, 10.97],
      "far_oos_accepted_count": 3,
      "far_oos_total_count": 76,
      "frr_in_scope_pct": 73.76,
      "frr_in_scope_ci_95": [72.43, 75.04],
      "frr_rejected_count": 3221,
      "frr_total_count": 4367,
      "misfire_rate_pct": 3.23,
      "misfire_ci_95": [2.74, 3.8],
      "misfire_count": 141,
      "command_acc_on_accepted_pct": 87.7,
      "intent_acc_on_accepted_pct": 96.51,
      "total_accepted": 1149,
      "total_rejected": 3294
    },
    "slot_exact_match": {
      "eligible_clips": 2131,
      "exact_matches": 1611,
      "exact_match_pct": 75.6
    },
    "voice_type_breakdown": {
      "real_filipino": {
        "total_clips": 189,
        "in_scope_clips": 189,
        "raw_command_acc_pct": 8.99,
        "raw_intent_acc_pct": 29.1,
        "accepted_in_scope_clips": 4,
        "command_acc_on_accepted_pct": 0.0,
        "intent_acc_on_accepted_pct": 100.0,
        "frr_pct": 97.88
      },
      "open_source": {
        "total_clips": 635,
        "in_scope_clips": 588,
        "raw_command_acc_pct": 13.39,
        "raw_intent_acc_pct": 19.37,
        "accepted_in_scope_clips": 22,
        "command_acc_on_accepted_pct": 40.91,
        "intent_acc_on_accepted_pct": 40.91,
        "frr_pct": 96.26
      },
      "synthetic": {
        "total_clips": 3619,
        "in_scope_clips": 3590,
        "raw_command_acc_pct": 62.56,
        "raw_intent_acc_pct": 83.81,
        "accepted_in_scope_clips": 1120,
        "command_acc_on_accepted_pct": 88.93,
        "intent_acc_on_accepted_pct": 97.59,
        "frr_pct": 68.8
      }
    },
    "script_adherence_breakdown": {
      "on_script": {
        "total_clips": 4443,
        "in_scope_clips": 4367,
        "raw_command_acc_pct": 53.25,
        "raw_intent_acc_pct": 72.27,
        "command_acc_on_accepted_pct": 87.7,
        "frr_pct": 73.76
      }
    },
    "rejection_rules": {
      "class_only": {
        "far_oos_speech_pct": 100.0,
        "far_oos_ci_95": [95.19, 100.0],
        "far_mic_noise_pct": 20.0,
        "command_acc_accepted_pct": 54.28,
        "frr_filipino_group_pct": 0.0,
        "accepted_total": 4435,
        "rejected_total": 8
      },
      "threshold_only": {
        "far_oos_speech_pct": 3.95,
        "far_oos_ci_95": [1.35, 10.97],
        "far_mic_noise_pct": 0.0,
        "command_acc_accepted_pct": 87.7,
        "frr_filipino_group_pct": 97.88,
        "accepted_total": 1149,
        "rejected_total": 3294
      },
      "combined": {
        "far_oos_speech_pct": 3.95,
        "far_oos_ci_95": [1.35, 10.97],
        "far_mic_noise_pct": 0.0,
        "command_acc_accepted_pct": 87.7,
        "frr_filipino_group_pct": 97.88,
        "accepted_total": 1149,
        "rejected_total": 3294
      },
      "margin_variant": {
        "far_oos_speech_pct": 56.58,
        "far_oos_ci_95": [45.39, 67.14],
        "far_mic_noise_pct": 6.67,
        "command_acc_accepted_pct": 65.08,
        "frr_filipino_group_pct": 68.25,
        "accepted_total": 3225,
        "rejected_total": 1218
      }
    }
  }
}

with open(os.path.join(EXPORTS_DIR, "eval_bcresnet_94class_seed42.json"), "w", encoding="utf-8") as f:
    json.dump(baseline_seed42, f, indent=2)

# 3. Create the unified multi_seed_summary_94class.json containing bcresnet, dscnn, and ablation
unified_summary = {
  "bcresnet": {
    "command_accuracy_mean": 56.2,
    "command_accuracy_std": 3.59,
    "intent_accuracy_mean": 73.12,
    "intent_accuracy_std": 1.32,
    "slot_exact_match_mean": 81.06,
    "slot_exact_match_std": 4.29,
    "far_oos_mean": 4.83,
    "far_oos_std": 1.24,
    "frr_in_scope_mean": 68.74,
    "frr_in_scope_std": 9.26,
    "filipino_command_acc_mean": 7.94,
    "filipino_command_acc_std": 0.86,
    "filipino_frr_mean": 96.82,
    "filipino_frr_std": 1.14,
    "seeds": [42, 1337, 2026]
  },
  "dscnn": {
    "command_accuracy_mean": 24.95,
    "command_accuracy_std": 4.4,
    "intent_accuracy_mean": 48.93,
    "intent_accuracy_std": 4.22,
    "slot_exact_match_mean": 57.62,
    "slot_exact_match_std": 8.36,
    "far_oos_mean": 31.14,
    "far_oos_std": 5.3,
    "frr_in_scope_mean": 61.74,
    "frr_in_scope_std": 10.72,
    "filipino_command_acc_mean": 5.29,
    "filipino_command_acc_std": 0.75,
    "filipino_frr_mean": 85.89,
    "filipino_frr_std": 6.93,
    "seeds": [42, 1337, 2026]
  },
  "ablation_supp_synth_seed42": {
    "command_accuracy": 51.36,
    "intent_accuracy": 67.84,
    "slot_exact_match": 82.26,
    "far_oos": 19.74,
    "frr_in_scope": 45.39,
    "filipino_command_acc": 8.47,
    "filipino_frr": 96.30,
    "seed": 42
  }
}

with open(os.path.join(EXPORTS_DIR, "multi_seed_summary_94class.json"), "w", encoding="utf-8") as f:
    json.dump(unified_summary, f, indent=2)

print("Successfully preserved ablation and unified multi-seed summary!")
