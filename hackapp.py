import logging
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import snscrape.modules.twitter as sntwitter
from transformers import pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import plotly.graph_objects as go
import json
from itertools import product


logging.basicConfig(level=logging.INFO)

def get_tweets(keyword):
    locs = []
    contents = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(keyword + ' lang:"en"').get_items()): #  + ' since:2022-01-01 until:2022-01-10, lang:"en"'
        if i >= 100:
           break
        #print(tweet.user.username)
        #print(tweet.content)
        contents.append(tweet.content)
        #print(tweet.date)
        #print(tweet.user.location)
        locs.append(tweet.user.location)
        #print("\n")

    return contents, locs

def sentiment(contents, locs):
    model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
    tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

    # create pipeline

    # classifier = pipeline("sentiment-analysis", tokenizer=tokenizer, model=model)

    #classifier(contents)
    #st.write(contents)
    #classifier = pipeline("sentiment-analysis")
    #a = classifier(contents)
    with torch.no_grad():
        features = tokenizer(
                    contents, padding=True, truncation=True, return_tensors="pt"
                )
        logits = model(**features).logits
        logits = logits.cpu().detach().numpy()
        label_mapping = ["NEG", "NEU", "POS"]
        print(logits.shape)
        print(np.argmax(logits, axis=1))
        labels = [label_mapping[label_id] for label_id in np.argmax(logits, axis=1)]

    print(labels)
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
    
    frequency['unknown'] = frequency.pop("")

    sort_orders = sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    # printing the frequency
    #print(frequency)

    return posCount, negCount, neuCount, rate, sort_orders

st.title("Industrial Sustainability Analyser")
st.write(" ")
st.write(" ")

option = st.selectbox(
     'Which company?',
     ('Tesla', 'Ford', 'General Electric'))


options = st.multiselect(
     'Select modules',
     ['Twitter Analysis', 'Gender', 'Carbon Footprint', 'Stock Price', 'News Analysis', 'Sustainability Score'],
     ['Twitter Analysis'])

if st.button('Analyse'):
    st.write(" ")
    st.write(" ")
    st.write(" ")
    if 'Gender' in options:
        labels = 'Male', 'Female'
        if option == 'Tesla': sizes = [49, 51]            
        if option == 'Ford':  sizes = [66, 34]          
        if option == 'General Electric': sizes = [76, 24]        

        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        startangle=90)
        ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        col2_1, col2_2,  = st.columns(2)

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
        st.write(" ")
        st.write(" ")
    
    if 'Carbon Footprint' in options:
        st.header("Carbon Footprint")
        if option == 'Tesla': st.subheader('Carbon Emission is ' + str(51.8096) + " percent smaller than average Co2 consumption in the industry")
        if option == 'Ford': st.subheader('Carbon Emission is: ' + str(27.2218) + " percent bigger than average Co2 consumption in the industry" )
        if option == 'General Electric': st.subheader('Carbon Emission: ' + str(27.9007)+ " percent smaller than average Co2 consumption in the industry")
        st.write(" ")
        st.write(" ")
        st.write(" ")
    
    if 'News Analysis' in options:
        st.header("News Analysis")
        f = open('output.json',)
        employee_dict = json.loads(f.read())
        col3_1, col3_2, col3_3, col3_4, col3_5 = st.columns(5)
        col_list = [col3_1, col3_2, col3_3, col3_4, col3_5]

        for key,value in employee_dict.items():
           if key == option.lower():
               for inner_key,inner_value in value.items(): 
                   st.subheader(str(inner_key) + ": " + str(round(inner_value[0][1] / (inner_value[0][1]+inner_value[1][1]), 2)))                  
                   #for values in inner_value:                                           
                        #st.write(str(values[0]) + ": " +str(values[1]))
        st.write(" ")
        st.write(" ")
        st.write(" ")

    if 'Twitter Analysis' in options:        
        contents, locs = get_tweets(option)
        posCount, negCount, neuCount, rate, sort_orders = sentiment(contents, locs)
        st.write(posCount)

        col1_1, col1_2 = st.columns(2)

        with col1_1:
            st.header("Twitter Analysis")
            """ 
            ##
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
            #            
            ### Most tweeted top 5 locations:            
            """
            st.write(" ")
            st.write(str(sort_orders[0][0])+ ' : '+ str(sort_orders[0][1]))
            st.write(str(sort_orders[1][0])+ ' : '+ str(sort_orders[1][1]))
            st.write(str(sort_orders[2][0])+ ' : '+ str(sort_orders[2][1]))
            st.write(str(sort_orders[3][0])+ ' : '+ str(sort_orders[3][1]))
            st.write(str(sort_orders[4][0])+ ' : '+ str(sort_orders[4][1]))
        
        st.write("--------------------------------") 

        for i in range(0,5):
            st.write(contents[i])   
            st.write("--------------------------------")   
        st.write(" ")
        f = open('output_tweets.json',)
        tweet_dict = json.loads(f.read())
        col3_1, col3_2, col3_3, col3_4, col3_5 = st.columns(5)
        col_list = [col3_1, col3_2, col3_3, col3_4, col3_5]

        for key,value in tweet_dict.items():
           if key == option.lower():
               for inner_key,inner_value in value.items(): 
                   st.subheader(str(inner_key) + ": " + str(round(inner_value[0][1] / (inner_value[0][1]+inner_value[1][1]), 2)))                  
                   #for values in inner_value:                                           
                        #st.write(str(values[0]) + ": " +str(values[1]))
        st.write(" ")
        st.write(" ")
        st.write(" ")
          
    
    #0.3 x GenderEquality + 0.3 x AvarageOfSentimentAnalysis + 0.1 x TwitterTrend + 0.3 x CarbonFootPrint
    if 'Sustainability Score' in options:
        st.header("Sustainability Score")
        if option == 'Tesla': st.subheader('Sustainability Score of Tesla Company is: ' + str(20+28+21+6+15))
        if option == 'Ford': st.subheader('Sustainability Score of Ford Company is: ' + str(20+21+22+5-9))
        if option == 'General Electric': st.subheader('Sustainability Score of General Company is: ' + str(20+18+25+4+9))
        st.write(" ")
        st.write(" ")
        st.write(" ")

    if 'Stock Price' in options:
        st.header("Stock Price Chart")
        if option == 'Tesla': 
            stock_df = pd.read_csv('TSLA_Historical_Data.csv')[::-1] 
        if option == 'Ford': 
            stock_df = pd.read_csv('F_Historical_Data.csv')[::-1]
        if option == 'General Electric': 
            stock_df = pd.read_csv('GE_Historical_Data.csv')[::-1]
        
      
        #fig = go.Figure([go.Scatter(x=stock_df['Date'], y=stock_df['Price'])])
        #st.plotly_chart(fig)
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
                
        stock_df["Date"] = pd.to_datetime(stock_df["Date"].apply(lambda x: str(month_map[x.split()[0]]) + ".15." + "20" + x.split()[1]), infer_datetime_format=True)
        stock_df = stock_df.sort_values(by=["Date"])
        # Using graph_objects
        
        fig = go.Figure([go.Scatter(x=stock_df['Date'], y=stock_df['Price'])])
        fig.update_yaxes(type="linear")
        st.plotly_chart(fig)
        

    




                    
                            
               
                        
                           
                               
                         
                
                       
                          
                  
               
                      
                         
            
                       
                        
                    
                                

                           
else:
    st.write('Goodbye')










