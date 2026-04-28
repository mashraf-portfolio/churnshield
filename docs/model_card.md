# ChurnShield Model Card

## Model Details

- **Name:** ChurnShield
- **Version:** 1.0.0
- **Date trained:** 2026-04-27
- **Model type:** Binary classifier (probability of customer churn)
- **Training algorithm:** XGBoost with Optuna hyperparameter search
- **Underlying estimator:** XGBoost+Optuna
- **Calibration method:** Isotonic regression (post-hoc, fit on holdout)
- **Training set size:** 5,634 customers (80% stratified split)

## Intended Use

Primary use is customer retention scoring for telecom subscribers — flag
high-churn-risk customers so a retention team can target outreach (offers,
plan reviews, service check-ins).

Out of scope: credit decisions, pricing, eligibility for service, and any
legal or regulatory determination. The model has no place in any decision
that affects a customer's access to services or credit.

## Training Data

- **Dataset:** IBM Telco Customer Churn (public, 7,043 customers, ~26%
  churn base rate). Joined from 5 relational files: main, demographics,
  location, services, status. The population file was verified to exist
  but explicitly NOT joined.
- **Features:** 19 raw inputs across account, demographics, services,
  and charges, plus 5 engineered features (`tenure_bucket`,
  `charges_per_month_ratio`, `contract_risk_score`,
  `service_bundle_count`, `high_value_flag`).
- **Leakage prevention:** Churn Reason, Churn Score, CLTV, Customer
  Status, and Churn Category were dropped before training. These are
  post-churn attributes that would leak the target.

## Evaluation Data

20% stratified holdout (1,409 customers). Stratification on the target
preserves the ~26% churn base rate in both splits.

## Metrics

Reported on the calibrated model against the holdout set.

| Metric                | Value  |
| --------------------- | ------ |
| ROC-AUC               | 0.857  |
| PR-AUC                | 0.671  |
| F1 (optimal threshold)| 0.593  |
| Brier score (calibrated) | 0.132 |
| Optimal threshold     | 0.216  |

The optimal threshold is chosen by maximizing F1 along the
precision-recall curve.

## Ethical Considerations

1. **Retention bias / self-fulfilling churn.** Treating predictions as
   destiny can deprioritize customers flagged as likely churners,
   creating the very outcome predicted. Mitigation: predictions feed
   retention *offers*, never service downgrades or worse terms.

2. **Demographic features.** `Gender` and `Senior Citizen` are present
   in the feature set and contribute to predictions (visible in SHAP
   attributions). A full fairness audit — disparate impact across
   protected groups, calibration parity — is recommended before any
   production deployment. No such audit has been conducted on this
   artifact.

3. **Consent and transparency.** Customers should be informed when
   their account data is used for retention scoring, per GDPR/CCPA-style
   norms. The model itself is silent on this; it is the deploying
   organization's responsibility to provide notice and honor opt-outs.

## Caveats and Recommendations

- Trained on US telecom data circa 2020. Do not deploy to other
  regions or industries (banking, GCC markets) without retraining on
  representative data.
- Drift monitoring is required in production. The Live Monitoring tab
  in this repo's Streamlit UI is a starting point — extend with
  Evidently AI or a comparable tool for proper feature- and
  prediction-drift detection.
- Recalibrate quarterly. Probability calibration degrades as the
  underlying customer mix shifts.
- For higher-stakes deployment, consider: human-in-the-loop review for
  predictions in the medium-risk band (where the model is least
  confident), and A/B testing of retention interventions to measure
  causal lift rather than predictive accuracy alone.

## License

MIT. Same as the repo.

## Contact

Mohammad Ashraf Ahmed Hafez — see the repo README for contact details.
