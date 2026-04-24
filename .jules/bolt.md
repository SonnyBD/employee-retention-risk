## 2024-04-24 - Cache Expensive ML Initializations in Streamlit
**Learning:** In Streamlit applications, expensive object initializations like `shap.TreeExplainer` block the main thread and slow down interactions if they're placed inside the main execution flow or form submission handlers.
**Action:** Always extract static ML explainers and models into `@st.cache_resource` decorated functions so they are only initialized once upon startup.
