import streamlit as st
import plotly.express as px
import polars as pl
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="collapsed",

)

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://i.postimg.cc/4xgNnkfX/Untitled-design.png");
background-size: cover;
background-position: center center;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
.center-content {{
    display: flex;
    justify-content: center;
    align-items: center;
    flex-direction: column;
}}
.metric-container {{
    display: flex;
    justify-content: center;
    gap: 20px;
    margin-bottom: 40px;
}}
.metric {{
    text-align: center;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)



@st.cache_data
def load_data():
    df = pl.read_csv("data/data.csv")
    # converte a data
    df.with_columns(
        pl.col("date").str.to_date("%Y %B %d",
                                   strict=False))
    return df


# carrega a data
data = load_data()
# st.dataframe(data)

header_l, header_main, header_r = st.columns([1, 2, 1], gap="large")
with header_main:
    st.title('Painel de Marketing Conversão Inteligente')

 


with st.sidebar:
   
    filter_campaign = st.multiselect(label="Select Campaign",
                                     options=data["campaign"].to_pandas().unique(),
                                     default=data["campaign"].to_pandas().unique())
    
    filter_age_group = st.multiselect(label="Select Age Group",
                                      options=data["age"].to_pandas().unique(),
                                      default=data["age"].to_pandas().unique())
    
    filter_gender = st.multiselect(label="Select Gender Group",
                                   options=data["gender"].to_pandas().unique(),
                                   default=data["gender"].to_pandas().unique())

df1 = data.to_pandas().query(
    "campaign == @filter_campaign & age == @filter_age_group & gender == @filter_gender"
)


total_impressions = float(
    df1['Impressions'].sum())
total_clicks = float(
    df1['Clicks'].sum())
total_spent = float(
    df1['Spent'].sum())
total_conversions = float(
    df1['Total_Conversion'].sum())
total_approved_conversions = float(
    df1['Approved_Conversion'].sum())


metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5, gap="large")

with metric_1:
    st.markdown(f"""
    ### Total de impressões
    ## {total_impressions}""")

with metric_2:
    st.markdown(f"""
    ### Total de clicks
    ## {total_clicks}""")

with metric_3:
    st.markdown(f"""
    ### Total gasto
    ## {round(total_spent, 2)}""")

with metric_4:
    st.markdown(f"""
    ### Total de conversões
    ## {total_conversions}""")


quad1, quad2 = st.columns(2, gap="large")

with quad1:
   
    df2 = df1.groupby(
        by="campaign"
    ).sum()[["Impressions", "Clicks"]]
    df2 = df2.reset_index("campaign")
  
    df2["CTR"] = round((df2["Clicks"] / df2["Impressions"]) * 100, 2)

    CTR_by_campaign = px.bar(df2,
                             x="campaign",
                             y="CTR",
                             color="campaign",
                             color_discrete_sequence=px.colors.qualitative.Set1,
                             title="Taxa de cliques")
    CTR_by_campaign.update_layout(title={"x": 0.5},
                                  showlegend=False,
                                  plot_bgcolor="rgba(0,0,0,0)",
                                  xaxis=(dict(showgrid=False)),
                                  yaxis=(dict(showgrid=False)))
    st.plotly_chart(CTR_by_campaign, use_container_width=True)

with quad2:
    
    impressions_by_day = px.line(df1,
                                 x="date",
                                 y="Impressions",
                                 color="campaign",
                                 color_discrete_sequence=px.colors.qualitative.Set1,
                                 title="Impressões por dia")
    impressions_by_day.update_xaxes(rangeslider_visible=True)
    impressions_by_day.update_layout(title={"x": 0.5},
                                     showlegend=False,
                                     plot_bgcolor="rgba(0,0,0,0)",
                                     xaxis=(dict(showgrid=False)),
                                     yaxis=(dict(showgrid=False)),
                                     xaxis_range=['2021-01-01', '2021-01-31']
                                     )
    st.plotly_chart(impressions_by_day, use_container_width=True)

quad3, quad4 = st.columns(2, gap="large")

with quad3:
    
    df3 = df1.groupby(by="gender") \
        .sum()["Spent"] \
        .reset_index()
    spent_by_gender = px.pie(data_frame=df3,
                             names="gender",
                             values="Spent",
                             color="gender",
                             color_discrete_sequence=px.colors.qualitative.Set1,
                             title="Anúncio gasto por gênero")
    spent_by_gender.update_layout(
        title={'x': 0.5},
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(spent_by_gender, use_container_width=True)
with quad4:
    
    df4 = df1.groupby(
        by="age"
    ).sum()[["Spent", "Total_Conversion"]].reset_index()
    df4["CPC"] = round(df4["Spent"] / df4["Total_Conversion"], 2)
    CPC_by_age = px.bar(df4,
                        x="age",
                        y="CPC",
                        title="Custo por conversão por faixa etária",
                        color="age",
                        color_discrete_sequence=px.colors.qualitative.Set1)
    CPC_by_age.update_layout(
        title={'x': 0.5},
        xaxis=(dict(showgrid=False)),
        yaxis=(dict(showgrid = False)),
        plot_bgcolor="rgba(0,0,0,0)"
    )
    st.plotly_chart(CPC_by_age, use_container_width=True)
