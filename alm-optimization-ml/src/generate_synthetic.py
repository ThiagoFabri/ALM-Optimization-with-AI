import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
rng = np.random.default_rng(42)

MODULES = ["core-banking", "payments", "risk-engine", "onboarding", "mobile-app", "internet-banking"]
ENV = ["dev", "qa", "staging", "prod"]

def gen_rows(n=1000):
    start = datetime.today() - timedelta(days=365)
    rows = []
    for i in range(n):
        dt = start + timedelta(days=int(rng.integers(0, 365)))
        commits = max(0, int(rng.normal(25, 10)))
        lines_changed = max(1, int(abs(rng.normal(1500, 600))))
        test_cov = np.clip(rng.normal(0.72, 0.1), 0.2, 0.99)
        defects_open = max(0, int(rng.normal(12, 5)))
        prior_failed = rng.binomial(3, 0.25)
        build_success = np.clip(rng.normal(0.9, 0.08), 0.4, 1.0)
        lead_time_days = np.clip(rng.normal(6, 2), 1, 30)
        cycle_time_days = np.clip(rng.normal(3, 1), 0.5, 20)
        code_smells = max(0, int(rng.normal(35, 12)))
        vulns = max(0, int(rng.normal(4, 3)))
        team_exp = np.clip(rng.normal(3.2, 1.1), 0.2, 8.0)
        env = rng.choice(ENV, p=[0.1, 0.25, 0.3, 0.35])
        module = rng.choice(MODULES)
        hour = rng.integers(0, 24)
        dow = rng.integers(0, 7)

        # Latent risk function (non-linear) to set target
        logit = (
            -1.0
            + 0.8 * (1 - test_cov)
            + 0.02 * defects_open
            + 0.0004 * lines_changed
            + 0.6 * (1 - build_success)
            + 0.25 * (prior_failed > 0)
            + 0.03 * vulns
            + 0.015 * code_smells
            + 0.05 * (lead_time_days > 10)
            + 0.05 * (cycle_time_days > 5)
            + 0.15 * (env == "prod")
            + 0.1 * (hour >= 18)
            + 0.05 * (dow in [5,6])  # fri/sat
        )
        p_fail = 1 / (1 + np.exp(-logit))
        deploy_failed = rng.random() < p_fail
        rows.append({
            "release_id": i,
            "release_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "module": module,
            "environment": env,
            "commits": commits,
            "lines_changed": lines_changed,
            "test_coverage": round(test_cov, 3),
            "defects_open": defects_open,
            "prior_failed_deploys": prior_failed,
            "build_success_rate": round(build_success, 3),
            "lead_time_days": round(float(lead_time_days), 2),
            "cycle_time_days": round(float(cycle_time_days), 2),
            "code_smells": code_smells,
            "vulnerabilities": vulns,
            "team_experience_years": round(float(team_exp), 1),
            "hour_of_day": int(hour),
            "day_of_week": int(dow),
            "deploy_failed": int(deploy_failed)
        })
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=Path("data/synthetic_release_log.csv"))
    args = ap.parse_args()
    df = gen_rows(args.rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
