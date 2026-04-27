"""Streamlit UI for ChurnShield churn prediction."""

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="ChurnShield",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ ChurnShield — Customer Churn Prediction")
st.caption(
    "Calibrated XGBoost model with SHAP explanations. "
    "Predicts churn probability for telecom customers."
)

with st.sidebar:
    st.markdown("### About")
    st.markdown(
        "Production ML pipeline with calibrated probabilities, "
        "SHAP explanations, and live monitoring."
    )
    st.markdown(f"**API:** `{API_URL}`")
    st.markdown("---")
    st.markdown("### Links")
    st.markdown("[GitHub](https://github.com/mashraf-portfolio/churnshield)")
    st.markdown("[API Docs](" + API_URL + "/docs)")

tab1, tab2, tab3 = st.tabs(
    [
        "Single Prediction",
        "Batch Prediction",
        "Live Monitoring",
    ]
)

with tab1:
    st.subheader("Single Prediction")
    st.caption(
        "Enter customer details below. The model returns a calibrated "
        "churn probability with risk band and feature-level explanations."
    )

    with st.form("predict_form"):
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.markdown("**Account**")
            customer_id = st.text_input("Customer ID (optional)", value="DEMO-001")
            tenure = st.number_input("Tenure (months)", 0, 72, 12)
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            paperless_billing = st.selectbox("Paperless billing", ["Yes", "No"])

        with col_b:
            st.markdown("**Charges & Demographics**")
            monthly_charges = st.number_input(
                "Monthly charges ($)", 0.0, 200.0, 70.0, 0.5
            )
            total_charges = st.number_input(
                "Total charges ($)", 0.0, 10000.0, 840.0, 10.0
            )
            senior_citizen = st.selectbox("Senior citizen", [0, 1])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            gender = st.selectbox("Gender", ["Male", "Female"])

        with col_c:
            st.markdown("**Services**")
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet service", ["Fiber optic", "DSL", "No"]
            )
            online_security = st.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )
            online_backup = st.selectbox(
                "Online backup", ["No", "Yes", "No internet service"]
            )
            device_protection = st.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech support", ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )

        submitted = st.form_submit_button("Predict churn", type="primary")

    if submitted:
        payload = {
            "customer_id": customer_id or None,
            "tenure": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract,
            "internet_service": internet_service,
            "payment_method": payment_method,
            "senior_citizen": senior_citizen,
            "partner": partner,
            "dependents": dependents,
            "phone_service": phone_service,
            "multiple_lines": multiple_lines,
            "online_security": online_security,
            "online_backup": online_backup,
            "device_protection": device_protection,
            "tech_support": tech_support,
            "streaming_tv": streaming_tv,
            "streaming_movies": streaming_movies,
            "paperless_billing": paperless_billing,
            "gender": gender,
        }

        with st.spinner("Scoring…"):
            try:
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                response.raise_for_status()
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
                st.stop()

        result = response.json()
        proba = result["churn_probability"]
        band = result["risk_band"]
        band_color = {"low": "🟢", "medium": "🟡", "high": "🔴"}[band]

        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability", f"{proba:.1%}")
        m2.metric("Risk band", f"{band_color} {band.title()}")
        m3.metric("Threshold", f"{result['threshold_used']:.4f}")

        st.markdown("### Feature Contributions to Prediction")
        st.caption(
            "How much each feature pushes the prediction toward churn (red) "
            "or away from churn (green). Sorted by impact magnitude."
        )

        shap_items = list(result["shap_values"].items())
        # Order: smallest abs at top, largest at bottom (waterfall convention)
        shap_items.sort(key=lambda kv: abs(kv[1]))
        labels = [k for k, _ in shap_items]
        values = [v for _, v in shap_items]
        colors = ["#EF4444" if v > 0 else "#10B981" for v in values]

        fig = go.Figure(
            go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in values],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
            )
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="SHAP value (impact on churn log-odds)",
            yaxis_title="",
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.add_vline(x=0, line_width=1, line_color="rgba(0,0,0,0.3)")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            f"Model: {result['model_version']} "
            f"({result['calibration_method']}-calibrated) | "
            f"Showing top {len(shap_items)} features by absolute impact"
        )

with tab2:
    st.subheader("Batch Prediction")
    st.caption(
        "Upload a CSV with customer rows. Each row is scored, "
        "predictions are logged, and results are downloadable."
    )

    uploaded = st.file_uploader(
        "CSV file with CustomerInput columns",
        type=["csv"],
        help=(
            "Required columns: customer_id (optional), tenure, "
            "monthly_charges, total_charges, contract, internet_service, "
            "payment_method, senior_citizen, partner, dependents, "
            "phone_service, multiple_lines, online_security, online_backup, "
            "device_protection, tech_support, streaming_tv, streaming_movies, "
            "paperless_billing, gender."
        ),
    )

    if uploaded is None:
        st.info("Upload a CSV to score multiple customers at once.")
    else:
        preview = pd.read_csv(uploaded)
        st.markdown(f"**Preview** — {len(preview):,} rows")
        st.dataframe(preview.head(5), use_container_width=True)

        if st.button("Score all rows", type="primary"):
            uploaded.seek(0)
            files = {"file": (uploaded.name, uploaded.getvalue(), "text/csv")}
            with st.spinner(f"Scoring {len(preview):,} rows…"):
                try:
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        files=files,
                        timeout=120,
                    )
                except requests.RequestException as exc:
                    st.error(f"API request failed: {exc}")
                    st.stop()

            if response.status_code == 413:
                st.warning(f"Batch too large. {response.json().get('detail', '')}")
                st.stop()
            if not response.ok:
                st.error(f"API error {response.status_code}: {response.text}")
                st.stop()

            body = response.json()
            summary = body["summary"]
            preds = body["predictions"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total rows", f"{summary['total']:,}")
            c2.metric("Churners", f"{summary['churners']:,}")
            c3.metric("Churn rate", f"{summary['churn_rate']:.1%}")
            c4.metric("High risk", f"{summary['high_risk']:,}")

            results_df = pd.DataFrame(
                [
                    {
                        "customer_id": p["customer_id"],
                        "churn_probability": p["churn_probability"],
                        "risk_band": p["risk_band"],
                        "churn_prediction": p["churn_prediction"],
                    }
                    for p in preds
                ]
            ).sort_values("churn_probability", ascending=False)

            st.markdown("**Results** — sorted by churn probability")
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv(index=False).encode()
            st.download_button(
                "Download scored CSV",
                data=csv_bytes,
                file_name="churnshield_predictions.csv",
                mime="text/csv",
            )

with tab3:
    st.subheader("Live Monitoring")
    st.caption(
        "Aggregate stats over predictions logged by the API in the last 30 days."
    )

    try:
        response = requests.get(f"{API_URL}/metrics/summary", timeout=5)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        st.error(f"Could not reach API at {API_URL}: {exc}")
        st.stop()

    if data["total_predictions"] == 0:
        st.info(
            "No predictions logged yet. Submit a prediction in Tab 1 "
            "or upload a batch in Tab 2 — this dashboard will populate."
        )
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total predictions", f"{data['total_predictions']:,}")
        col2.metric("Churn rate (30d)", f"{data['churn_rate_last_30d']:.1%}")
        col3.metric("Avg probability", f"{data['avg_probability']:.3f}")
        col4.metric("P95 probability", f"{data['p95_probability']:.3f}")

        if data.get("last_prediction_at"):
            st.caption(f"Last prediction: {data['last_prediction_at']}")
