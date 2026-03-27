import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(layout="wide")

st.sidebar.title("📊 Intelligence Platform")
page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📊 Descriptive Analysis",
    "🔍 Diagnostic Analysis",
    "👥 Customer Clustering",
    "🎯 Classification",
    "🔗 Association Rules",
    "💰 Spend Regression",
    "🧪 Predict New Customers"
])

file = st.file_uploader("Upload Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    df_enc = df.copy()
    for col in df_enc.columns:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    if page == "🏠 Overview":
        st.title("Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Users", len(df))
        c2.metric("Top Experience", df["Experience_Type"].mode()[0])
        c3.metric("Top Spend", df["Spend"].mode()[0])
        st.info("This platform uses data-driven analytics to understand and predict customer behavior.")

    elif page == "📊 Descriptive Analysis":
        st.title("Understanding Customers")
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.histogram(df, x="Age", title="Age Distribution"))
            st.plotly_chart(px.pie(df, names="Experience_Type", title="Experience Preferences"))
        with col2:
            st.plotly_chart(px.box(df, x="Income", y="Spend", title="Income vs Spend"))
            st.plotly_chart(px.histogram(df, x="Barriers", title="Customer Barriers"))
        st.success("Insight: Majority users are young and prefer social/food experiences.")

    elif page == "🔍 Diagnostic Analysis":
        st.title("Why Behavior Happens")
        st.plotly_chart(px.scatter(df, x="Income", y="Spend", title="Income vs Spending"))
        st.info("Higher income users tend to spend more.")
        st.info("Barriers like price and time reduce engagement.")

    elif page == "👥 Customer Clustering":
        st.title("Customer Segmentation")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_enc)
        kmeans = KMeans(n_clusters=3)
        df["Cluster"] = kmeans.fit_predict(X_scaled)
        st.plotly_chart(px.scatter(df, x="Age", y="Spend", color="Cluster", size="Frequency"))
        st.success("Segment 1: Low Intent → Discounts")
        st.success("Segment 2: Premium → Upsell")
        st.success("Segment 3: Social → Bundles")

    elif page == "🎯 Classification":
        st.title("Conversion Prediction")
        X = df_enc.drop("Likelihood", axis=1)
        y = df_enc["Likelihood"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", round(accuracy_score(y_test, pred), 2))
        c2.metric("Precision", round(precision_score(y_test, pred, average='weighted'), 2))
        c3.metric("Recall", round(recall_score(y_test, pred, average='weighted'), 2))
        c4.metric("F1", round(f1_score(y_test, pred, average='weighted'), 2))

    elif page == "🔗 Association Rules":
        st.title("Bundling Insights")
        top = df["Experience_Type"].value_counts().head(5)
        st.bar_chart(top)
        st.info("Customers who prefer top experiences are likely to explore similar categories.")

    elif page == "💰 Spend Regression":
        st.title("Spending Prediction")
        y = df_enc["Spend"]
        X = df_enc.drop("Spend", axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        reg = RandomForestRegressor()
        reg.fit(X_train, y_train)
        preds = reg.predict(X_test)
        st.plotly_chart(px.scatter(x=y_test, y=preds, labels={"x":"Actual","y":"Predicted"}))

    elif page == "🧪 Predict New Customers":
        st.title("New Customer Prediction")
        X = df_enc.drop("Likelihood", axis=1)
        y = df_enc["Likelihood"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        user = {}
        for col in X.columns:
            user[col] = st.number_input(col, 0, 100, 0)

        if st.button("Predict"):
            user_df = pd.DataFrame([user]).reindex(columns=X.columns, fill_value=0)
            pred = clf.predict(user_df)
            st.success(f"Prediction: {pred}")
