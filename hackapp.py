import logging
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import snscrape.modules.twitter as sntwitter
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import plotly.graph_objects as go


logging.basicConfig(level=logging.INFO)


def get_tweets(keyword):
    locs = []
    contents = []
    for i, tweet in enumerate(
        sntwitter.TwitterSearchScraper(keyword + ' lang:"en"').get_items()
    ):  #  + ' since:2022-01-01 until:2022-01-10, lang:"en"'
        if i >= 100:
            break
        contents.append(tweet.content)
        locs.append(tweet.user.location)

    return contents, locs


def sentiment(contents, locs):
    model = AutoModelForSequenceClassification.from_pretrained(
        "finiteautomata/bertweet-base-sentiment-analysis"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "finiteautomata/bertweet-base-sentiment-analysis"
    )

    with torch.no_grad():
        features = tokenizer(
            contents, padding=True, truncation=True, return_tensors="pt"
        )
        logits = model(**features).logits
        logits = logits.cpu().detach().numpy()
        label_mapping = ["NEG", "NEU", "POS"]
        labels = [label_mapping[label_id] for label_id in np.argmax(logits, axis=1)]

    posCount = labels.count("POS")
    negCount = labels.count("NEG")
    neuCount = labels.count("NEU")
    rate = labels.count("POS") / (labels.count("POS") + labels.count("NEG"))

    frequency = {}

    # iterating over the list
    for item in locs:
        # checking the element in dictionary
        if item in frequency:
            # incrementing the counr
            frequency[item] += 1
        else:
            # initializing the count
            frequency[item] = 1

    frequency["unknown"] = frequency.pop("")

    sort_orders = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    # printing the frequency
    # print(frequency)

    return posCount, negCount, neuCount, rate, sort_orders


st.title("ESG Analyser")

option = st.selectbox("Which company?", ("Tesla", "Ford Motor", "General Electric"))


options = st.multiselect(
    "Select modules",
    ["Twitter Analysis", "Gender", "Carbon Credit", "Stock Price"],
    ["Twitter Analysis"],
)

if st.button("Analyse"):
    if "Twitter Analysis" in options:
        contents, locs = get_tweets(option)
        posCount, negCount, neuCount, rate, sort_orders = sentiment(contents, locs)
        st.write(posCount)

        col1_1, col1_2 = st.columns(2)

        with col1_1:
            st.header("Twitter Analysis")
            """ 
            ### Sentiment Analysis   
            #         
            """
            st.write("Positive Tweet count: " + str(posCount))
            st.write("Neutral Tweet count: " + str(neuCount))
            st.write("Negative Tweet count: " + str(negCount))
            st.write("Ratio: " + str(rate))

        with col1_2:
            """
            #
            ### Most tweeted top 5 locations:
            #
            """
            st.write(" ")
            st.write(str(sort_orders[0][0]) + " : " + str(sort_orders[0][1]))
            st.write(str(sort_orders[1][0]) + " : " + str(sort_orders[1][1]))
            st.write(str(sort_orders[2][0]) + " : " + str(sort_orders[2][1]))
            st.write(str(sort_orders[3][0]) + " : " + str(sort_orders[3][1]))
            st.write(str(sort_orders[4][0]) + " : " + str(sort_orders[4][1]))

        for i in range(0, 5):
            st.write(contents[i])
            st.write("--------------------------------")
        st.write(" ")

    if "Gender" in options:
        labels = "Male", "Female"
        if option == "Tesla":
            sizes = [49, 51]
        if option == "Ford Motor":
            sizes = [66, 34]
        if option == "General Electric":
            sizes = [76, 24]

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

        (
            col2_1,
            col2_2,
        ) = st.columns(2)

        with col2_1:
            st.header("Gender Diversity")
            st.pyplot(fig1)

        with col2_2:
            st.header(" ")
            st.write(" ")
            st.write(" ")
            """
            #  
            # 
            #            
            """
            st.write("Male ratio is: " + str(sizes[0]))
            st.write("Female ratio is: " + str(sizes[1]))
        st.write(" ")

    if "Carbon Credit" in options:
        st.header("Carbon Credit")
        if option == "Tesla":
            st.write("Carbon Emission: " + str(-0.518096))
        if option == "Ford Motor":
            st.write("Carbon Emission: " + str(0.272218))
        if option == "General Electric":
            st.write("Carbon Emission: " + str(-0.279007))
        st.write(" ")

    if "Stock Price" in options:
        st.header("Stock Price Chart")
        if option == "Tesla":
            stock_df = pd.read_csv("TSLA_Historical_Data.csv")[::-1]
        if option == "Ford Motor":
            stock_df = pd.read_csv("F_Historical_Data.csv")[::-1]
        if option == "General Electric":
            stock_df = pd.read_csv("GE_Historical_Data.csv")[::-1]

        # fig = go.Figure([go.Scatter(x=stock_df['Date'], y=stock_df['Price'])])
        # st.plotly_chart(fig)
        month_map = {
            "Jan": 1,
            "Feb": 2,
            "Mar": 3,
            "Apr": 4,
            "May": 5,
            "Jun": 6,
            "Jul": 7,
            "Aug": 8,
            "Sep": 9,
            "Oct": 10,
            "Nov": 11,
            "Dec": 12,
        }

        stock_df["Date"] = pd.to_datetime(
            stock_df["Date"].apply(
                lambda x: str(month_map[x.split()[0]]) + ".15." + "20" + x.split()[1]
            ),
            infer_datetime_format=True,
        )
        stock_df = stock_df.sort_values(by=["Date"])
        # Using graph_objects

        fig = go.Figure([go.Scatter(x=stock_df["Date"], y=stock_df["Price"])])
        fig.update_yaxes(type="linear")
        st.plotly_chart(fig)
        st.write(" ")

else:
    st.write("Goodbye")
