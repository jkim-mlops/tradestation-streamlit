import streamlit as st

st.set_page_config(
    page_title="Quantly",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "# Quantitative Trading Strategy Analysis Platform",
    },
)

pg = st.navigation(
    [
        st.Page("./pages/chart.py", title="Chart", icon="📊"),
    ]
)
pg.run()
