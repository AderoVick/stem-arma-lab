"""Professional Streamlit dashboard for TextPulse Forecast Lab."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import DataQualityReport, load_text_data  # noqa: E402
from src.forecasting import ForecastBundle, create_forecast  # noqa: E402
from src.text_analysis import (  # noqa: E402
    FREQUENCIES,
    build_topic_timeseries,
    suggest_topics,
)

SAMPLE_DATA = ROOT / "data" / "sample_text.csv"


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1250px;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.20);}
        .hero {
            padding: 1.5rem 1.75rem;
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(65,105,225,.10), rgba(72,209,204,.08));
            margin-bottom: 1.25rem;
        }
        .hero h1 {margin: 0; font-size: 2.15rem;}
        .hero p {margin: .55rem 0 0; font-size: 1.02rem; opacity: .84;}
        .eyebrow {
            font-size: .78rem; font-weight: 700; letter-spacing: .12em;
            text-transform: uppercase; opacity: .68;
        }
        .callout {
            padding: .85rem 1rem;
            border-left: 4px solid #5b78ff;
            background: rgba(91,120,255,.08);
            border-radius: 8px;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,.18);
            padding: .85rem 1rem;
            border-radius: 14px;
            background: rgba(255,255,255,.025);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_source(uploaded_file) -> tuple[pd.DataFrame, DataQualityReport, str]:
    source = uploaded_file if uploaded_file is not None else SAMPLE_DATA
    data, report = load_text_data(source)
    label = uploaded_file.name if uploaded_file is not None else "Built-in demonstration dataset"
    return data, report, label


def parse_topics(value: str) -> list[str]:
    return [topic.strip() for topic in value.split(",") if topic.strip()]


def make_history_chart(topic_frame: pd.DataFrame) -> go.Figure:
    long_frame = topic_frame.reset_index().melt(
        id_vars="date", var_name="Topic", value_name="Mentions"
    )
    figure = px.line(
        long_frame,
        x="date",
        y="Mentions",
        color="Topic",
        markers=True,
        title="Topic activity over time",
    )
    figure.update_layout(
        hovermode="x unified",
        legend_title_text="",
        xaxis_title="Date",
        yaxis_title="Mentions",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return figure


def make_forecast_chart(
    series: pd.Series,
    bundle: ForecastBundle,
    topic: str,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=series.index,
            y=series.values,
            mode="lines+markers",
            name="Historical",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=bundle.forecast.index,
            y=bundle.upper.values,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=bundle.forecast.index,
            y=bundle.lower.values,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(91, 120, 255, 0.18)",
            name="95% confidence interval",
            hoverinfo="skip",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=bundle.forecast.index,
            y=bundle.forecast.values,
            mode="lines+markers",
            name="Forecast",
            line=dict(dash="dash"),
        )
    )
    forecast_boundary = series.index[-1].to_pydatetime()
    figure.add_shape(
        type="line",
        x0=forecast_boundary,
        x1=forecast_boundary,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(dash="dot", width=1),
    )
    figure.add_annotation(
        x=forecast_boundary,
        y=1,
        yref="paper",
        text="Forecast begins",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
    )
    figure.update_layout(
        title=f"Historical and forecast activity for “{topic}”",
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Mentions",
        legend_title_text="",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return figure


def explain_forecast(series: pd.Series, bundle: ForecastBundle, topic: str) -> str:
    recent_window = series.tail(min(7, len(series)))
    split_start = -2 * len(recent_window)
    split_end = -len(recent_window)
    earlier_window = series.iloc[split_start:split_end]
    if len(earlier_window) and recent_window.mean() > earlier_window.mean() * 1.1:
        recent_direction = "increased"
    elif len(earlier_window) and recent_window.mean() < earlier_window.mean() * 0.9:
        recent_direction = "decreased"
    else:
        recent_direction = "remained broadly stable"

    forecast_mean = float(bundle.forecast.mean())
    recent_mean = float(recent_window.mean())
    if forecast_mean > recent_mean * 1.1:
        outlook = "a higher average level"
    elif forecast_mean < recent_mean * 0.9:
        outlook = "a lower average level"
    else:
        outlook = "a similar average level"

    return (
        f"Mentions of **{topic}** have {recent_direction} recently. "
        f"The selected **{bundle.model_name}** model projects {outlook} over the "
        f"forecast horizon. Treat the confidence range as the plausible uncertainty "
        f"around the central estimate, not as a guarantee."
    )


def render_overview(
    data: pd.DataFrame,
    report: DataQualityReport,
    source_label: str,
    topic_frame: pd.DataFrame,
) -> None:
    st.subheader("Dataset overview")
    columns = st.columns(5)
    columns[0].metric("Valid records", f"{report.output_rows:,}")
    columns[1].metric("Time periods", f"{len(topic_frame):,}")
    columns[2].metric("Topics tracked", len(topic_frame.columns))
    columns[3].metric("Total mentions", f"{int(topic_frame.sum().sum()):,}")
    columns[4].metric(
        "Date span",
        f"{(report.end_date - report.start_date).days + 1} days",
    )

    st.plotly_chart(make_history_chart(topic_frame), width="stretch")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("#### Data preview")
        st.dataframe(data.head(12), width="stretch", hide_index=True)
    with right:
        st.markdown("#### Data quality")
        quality = pd.DataFrame(
            {
                "Check": [
                    "Input rows",
                    "Rows retained",
                    "Invalid dates removed",
                    "Blank text removed",
                    "Duplicates removed",
                ],
                "Result": [
                    report.input_rows,
                    report.output_rows,
                    report.invalid_date_rows,
                    report.blank_text_rows,
                    report.duplicate_rows,
                ],
            }
        )
        st.dataframe(quality, width="stretch", hide_index=True)
        st.caption(f"Source: {source_label}")


def render_explorer(data: pd.DataFrame, topic_frame: pd.DataFrame, mode: str) -> None:
    st.subheader("Text and topic explorer")
    suggestions = suggest_topics(data, top_n=20, mode=mode)
    suggestion_frame = pd.DataFrame(suggestions, columns=["Topic", "Occurrences"])
    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("#### Frequent topic candidates")
        bar = px.bar(
            suggestion_frame.sort_values("Occurrences"),
            x="Occurrences",
            y="Topic",
            orientation="h",
            height=520,
        )
        bar.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(bar, width="stretch")
    with right:
        st.markdown("#### Topic summary")
        summary = pd.DataFrame(
            {
                "Total mentions": topic_frame.sum(),
                "Average per period": topic_frame.mean().round(2),
                "Peak mentions": topic_frame.max(),
                "Peak period": [
                    topic_frame[column].idxmax().date().isoformat()
                    for column in topic_frame.columns
                ],
                "Periods with mentions": (topic_frame > 0).sum(),
            }
        )
        st.dataframe(summary, width="stretch")
        st.markdown("#### Aggregated series")
        st.dataframe(
            topic_frame.reset_index().tail(30),
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download aggregated topic data",
            data=topic_frame.reset_index().to_csv(index=False).encode("utf-8"),
            file_name="topic_timeseries.csv",
            mime="text/csv",
        )


def render_forecast(topic_frame: pd.DataFrame) -> None:
    st.subheader("Forecast a selected topic")
    controls, output = st.columns([0.82, 2.18])

    with controls:
        forecast_topic = st.selectbox("Topic to forecast", list(topic_frame.columns))
        automatic = st.toggle("Automatically select the model", value=True)
        steps = st.slider("Forecast periods", 3, 60, 14)
        confidence = st.slider("Confidence level", 0.80, 0.99, 0.95, 0.01)

        order = (1, 0, 1)
        if not automatic:
            st.markdown("**Manual ARIMA order**")
            p = st.number_input("Autoregressive order (p)", 0, 5, 1)
            d = st.number_input("Differencing order (d)", 0, 2, 0)
            q = st.number_input("Moving-average order (q)", 0, 5, 1)
            order = (int(p), int(d), int(q))

        run_forecast = st.button(
            "Run forecast",
            type="primary",
            width="stretch",
        )

    if run_forecast:
        with st.spinner("Fitting and evaluating forecasting models..."):
            try:
                bundle = create_forecast(
                    topic_frame[forecast_topic],
                    steps=steps,
                    order=order,
                    automatic=automatic,
                    confidence=confidence,
                )
                st.session_state["forecast_bundle"] = bundle
                st.session_state["forecast_series"] = topic_frame[forecast_topic]
                st.session_state["forecast_topic"] = forecast_topic
            except Exception as exc:
                st.error(str(exc))
                return

    with output:
        if "forecast_bundle" not in st.session_state:
            st.info("Choose the settings and select **Run forecast** to generate results.")
            return
        bundle: ForecastBundle = st.session_state["forecast_bundle"]
        series: pd.Series = st.session_state["forecast_series"]
        topic = st.session_state["forecast_topic"]

        metric_columns = st.columns(4)
        metric_columns[0].metric("Selected model", bundle.model_name)
        metric_columns[1].metric("Backtest RMSE", f"{bundle.metrics['RMSE']:.2f}")
        metric_columns[2].metric("Backtest MAE", f"{bundle.metrics['MAE']:.2f}")
        metric_columns[3].metric("Backtest sMAPE", f"{bundle.metrics['sMAPE']:.1f}%")
        st.plotly_chart(
            make_forecast_chart(series, bundle, topic),
            width="stretch",
        )
        st.markdown(explain_forecast(series, bundle, topic))

        download = pd.DataFrame(
            {
                "date": bundle.forecast.index,
                "forecast": bundle.forecast.values,
                "lower_interval": bundle.lower.values,
                "upper_interval": bundle.upper.values,
                "model": bundle.model_name,
            }
        )
        st.download_button(
            "Download forecast",
            data=download.to_csv(index=False).encode("utf-8"),
            file_name=f"{topic}_forecast.csv",
            mime="text/csv",
        )


def render_diagnostics() -> None:
    st.subheader("Model diagnostics")
    if "forecast_bundle" not in st.session_state:
        st.info("Run a forecast first to view model comparison and residual diagnostics.")
        return

    bundle: ForecastBundle = st.session_state["forecast_bundle"]
    st.markdown("#### Backtest comparison")
    comparison = bundle.comparison.copy()
    for column in ["MAE", "RMSE", "sMAPE"]:
        if column in comparison:
            comparison[column] = comparison[column].round(3)
    st.dataframe(comparison, width="stretch", hide_index=True)

    if bundle.fitted_model is None:
        st.info("The naive baseline was selected, so ARIMA residual diagnostics are unavailable.")
        return

    residuals = pd.Series(bundle.fitted_model.resid).replace([np.inf, -np.inf], np.nan).dropna()
    left, right = st.columns(2)
    with left:
        residual_figure = px.line(
            x=residuals.index,
            y=residuals.values,
            labels={"x": "Period", "y": "Residual"},
            title="Residuals over time",
        )
        residual_figure.add_hline(y=0, line_dash="dot")
        st.plotly_chart(residual_figure, width="stretch")
    with right:
        histogram = px.histogram(
            residuals,
            nbins=20,
            labels={"value": "Residual"},
            title="Residual distribution",
        )
        histogram.update_layout(showlegend=False)
        st.plotly_chart(histogram, width="stretch")

    info = {
        "AIC": float(bundle.fitted_model.aic),
        "BIC": float(bundle.fitted_model.bic),
        "Residual mean": float(residuals.mean()),
        "Residual standard deviation": float(residuals.std()),
    }
    st.dataframe(
        pd.DataFrame(info.items(), columns=["Diagnostic", "Value"]),
        width="stretch",
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="TextPulse Forecast Lab",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">NLP + Time-Series Analytics</div>
          <h1>TextPulse Forecast Lab</h1>
          <p>Turn dated text into measurable topic trends, compare forecasting models,
          and export decision-ready results.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Analysis settings")
        uploaded_file = st.file_uploader(
            "Upload a CSV",
            type=["csv"],
            help="Required columns: date and text",
        )
        normalization_label = st.radio(
            "Topic matching",
            ["Porter stemming", "Exact word"],
            help="Stemming groups related forms such as delay, delayed, and delays.",
        )
        mode = "stem" if normalization_label == "Porter stemming" else "exact"
        frequency_label = st.selectbox("Aggregation", list(FREQUENCIES.values()))
        frequency = next(key for key, label in FREQUENCIES.items() if label == frequency_label)

    try:
        data, report, source_label = load_source(uploaded_file)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    suggested = [topic for topic, _ in suggest_topics(data, top_n=8, mode=mode)]
    preferred = [topic for topic in ["traffic", "delay", "climate"] if topic in suggested]
    defaults = preferred or suggested[:3]

    with st.sidebar:
        topic_value = st.text_input(
            "Topics",
            value=", ".join(defaults),
            help="Enter one or more comma-separated topics.",
        )
        st.caption("Suggested: " + ", ".join(suggested[:8]))

    try:
        topics = parse_topics(topic_value)
        topic_frame = build_topic_timeseries(
            data,
            topics,
            frequency=frequency,
            mode=mode,
        )
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    overview_tab, explorer_tab, forecast_tab, diagnostics_tab = st.tabs(
        ["Overview", "Explore", "Forecast", "Diagnostics"]
    )
    with overview_tab:
        render_overview(data, report, source_label, topic_frame)
    with explorer_tab:
        render_explorer(data, topic_frame, mode)
    with forecast_tab:
        render_forecast(topic_frame)
    with diagnostics_tab:
        render_diagnostics()

    st.divider()
    st.caption(
        "Forecasts are statistical estimates and should be interpreted alongside domain knowledge."
    )


if __name__ == "__main__":
    main()
