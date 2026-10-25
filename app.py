"""Streamlit entry point. Run with: streamlit run app.py."""
import runpy


if __name__ == "__main__":
    # Streamlit reruns this script; run_module executes the UI on every rerun.
    runpy.run_module("agent_modeling.interface", run_name="__main__")
