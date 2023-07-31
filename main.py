import streamlit as st
import streamlit as st
st.set_page_config(
    page_title="USA COVID-19 Data Analysis And Prediction",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded",
)


import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import plotly.graph_objects as go
from sklearn.model_selection import GridSearchCV, cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures



df = pd.read_csv('CleanCovid-19Dataset.csv')

X = df[['actuals.hospitalBeds.currentUsageCovid','actuals.hospitalBeds.currentUsageCovid','actuals.hospitalBeds.weeklyCovidAdmissions']]
y = df['actuals.newCases']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lg=LinearRegression()

poly=PolynomialFeatures(degree=2)

x_train_fit = poly.fit_transform(X_train) #transforming our input data

lg.fit(x_train_fit, y_train)#fit our model

x_test_ = poly.fit_transform(X_test)

predicted = lg.predict(x_test_)



st.title("USA COVID-19 Data Analysis And Prediction")
hospitalBedscurrentusagetotal= st.number_input("Enter the number of hospitalBeds in current Usage Total", value=0)
hospitalBedscurrentusagetotal= st.number_input("Enter the number of hospitalBeds in current Usage Covid", value=0)
hospitalBeds_weeklyCovidAdmissions= st.number_input("Enter the number of hospitalBeds_weekly Covid Admissions", value=0)


new_input = [[hospitalBedscurrentusagetotal,hospitalBedscurrentusagetotal, hospitalBeds_weeklyCovidAdmissions]]  # New input value as a list of features (e.g., hospitalBeds.currentUsageCovid, hospitalBeds.currentUsageCovid, hospitalBeds.weeklyCovidAdmissions)
new_input_transformed = poly.transform(new_input)
predicted_output = lg.predict(new_input_transformed)

print("Predicted output for new input:", predicted_output[0])
button=st.button("Predict",use_container_width=True)

if button:
    new_input = [[hospitalBedscurrentusagetotal, hospitalBedscurrentusagetotal, hospitalBeds_weeklyCovidAdmissions]]  # New input value as a list of features (e.g., hospitalBeds.currentUsageCovid, hospitalBeds.currentUsageCovid, hospitalBeds.weeklyCovidAdmissions)
    new_input_transformed = poly.transform(new_input)
    predicted_output = lg.predict(new_input_transformed)
    st.write("Predicted output for new input:", predicted_output[0])

# dashboard title
st.title("USA Covid-19 Dashboard")


# top-level filters
state_filter = st.selectbox("Select the State", pd.unique(df["state"]))

# creating a single-element container
placeholder = st.empty()

# dataframe filter
df = df[df["state"] == state_filter]

# near real-time / live feed simulation
for seconds in range(200):
    # creating KPIs
    avg_cases = np.mean(df["actuals.cases"])

    avg_positive_test = np.mean(df["actuals.positiveTests"])

    avg_negative_test = np.mean(df["actuals.negativeTests"])

    avg_deaths = np.mean(df["actuals.deaths"])

    with placeholder.container():
        # create three columns
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Cases 🦠",
            value=int(avg_cases),
            delta=round(avg_cases) - 10,
        )

        kpi2.metric(

            label="Deaths ☠️",
            value=round(avg_deaths),
            delta=round(avg_deaths) - 10,
        )

        kpi3.metric(
            label="Positive Test ➕",
            value=round(avg_positive_test),
            delta=round(avg_positive_test) - 10,

        )

        kpi4.metric(
            label="Negative Test ➖",
            value=round(avg_negative_test),
            delta=round(avg_negative_test) - 10,
        )

        # create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Distribution Of Cases For Each State")
            fig=px.histogram(data_frame=df, x="actuals.cases")
            st.write(fig)

        with fig_col2:
            st.markdown("### Distribution Of Deaths For Each State")
            fig2 = px.histogram(data_frame=df, x="actuals.deaths")
            st.write(fig2)

        st.markdown("### Detailed Data View")
        # df=df.drop('Unnamed: 0',axis=1)
        st.dataframe(df)
        time.sleep(1)













