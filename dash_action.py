# -*- coding: utf-8 -*-
"""
Created on Thu May 26 11:31:03 2022

@author: arnau
"""

### frameworks ###

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import yfinance as yf
from scipy.optimize import minimize

### header one (title) ###

st.set_page_config(page_title="Dashboard StockPrice")
st.header("Portfolio on the best CAC40 stocks")

### DarkMode & Style ###

darkmode = """
<style>
body {
  background-color: black;
  color: white;
}
</style>
"""

style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(style, unsafe_allow_html=True)

### dropdown ###

st.subheader("Drop-Down")
tickers = ['CAP.PA', 'OR.PA', 'AIR.PA', 'SAN.PA', 'SU.PA']

# ticker detail

st.caption("Ticker Description")

dico_tickers = { 'Investment Title':['CapGemini',"L'oréal",'Airbus','Sanofi','Schneider Electric'],
                'Tickers':tickers

    }

df_tickers_description = pd.DataFrame(dico_tickers)
st.dataframe(df_tickers_description)

dropdown = st.multiselect(
    
'Chosen the different tickers (please select the five tickers from the drop-down list to see the investment gains for this portfolio)',

 tickers,default=['CAP.PA', 'OR.PA', 'AIR.PA', 'SAN.PA', 'SU.PA'])

start= st.date_input('Start Date (you can choose a start date, please click on the drop-down list)',value=pd.to_datetime('2023-01-02'))
end = st.date_input('End Date (you can choose an end date, please click on the drop-down list)',value=pd.to_datetime('today'))

### first condition ###

if len(dropdown)>0:
    
    ### Connexion (yahoo finance) ###
    
    data = yf.download(dropdown,start,end)['Adj Close']
    
    ## graph 1 (Evolution of Prices) ##
    
    st.subheader('Evolution of Prices')
    st.caption("Last Price (please select at least 2 titles to see the latest prices)")
    
    if len(dropdown)>1:
    
        last_price = round(data.iloc[-1].sort_values(ascending=False),2)
        st.dataframe(last_price)
        
    graph_one = px.line(data)
    graph_one.update_layout(yaxis={"title": "Price"},legend_title="Tickers",
    title={'text':'Stock Prices','y':0.92,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    graph_one['layout']['title']['font'] = dict(size=15)
    st.plotly_chart(graph_one)
    
    ## graph 2  (Cumulative return) ##
    
    st.subheader("Cumulative Return")
    st.caption("A title whose percentage is lower than 1 is not good")
    df_cum = data/data.iloc[0]
    df_cum['lIM']= np.linspace(1,1,len(df_cum))
    graph_two = px.line(df_cum)
    graph_two.update_layout(yaxis={"title":"(%) Percentage"},legend_title="Tickers",
    title={'text':'Calculation of cumulative returns','y':0.92,'x':0.5,'xanchor': 'center','yanchor': 'top'})
    graph_two['layout']['title']['font'] = dict(size=15)
    st.plotly_chart(graph_two)
    
    ### Simulation Monte-Carlo (portfolio optimisation) ###
    
    ### second condition ###
    
    if len(dropdown)>4:

        np.random.seed(101)
    
        num_ports = 500
        all_weights = np.zeros((num_ports,len(data.columns)))
        ret_arr = np.zeros(num_ports) 
        vol_arr = np.zeros(num_ports)
        sharpe_arr = np.zeros(num_ports)
        
        # cumulatif return calculation copy
        
        df_cum_second = data/data.iloc[0]
    
        for ind in range(num_ports):
            
            log_ret = np.log(data/data.shift(1))
    
            # weights
            
            weights = np.array(np.random.random(5))
            weights = weights/np.sum(weights)
            
            # save weights
            
            all_weights[ind,:] = weights
                
            # Global daily return
            
            ret_arr[ind] = np.sum(log_ret.mean() * weights * 252)
            
            # Global volatility
            
            vol_arr[ind] = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
            
            # ration sharpe
            
            sharpe_arr[ind] = ret_arr[ind]/vol_arr[ind]
            
        # dataframe (vol & ret & ratio sharpe)
            
        vol = pd.DataFrame(vol_arr,columns=['Volatility'])
        ret = pd.DataFrame(ret_arr,columns=['Return'])
        sharpe = pd.DataFrame(sharpe_arr,columns=['(%) Sharpe_ratio'])
        
        # best ratio sharpe
    
        best_result = round(sharpe['(%) Sharpe_ratio'].max(),2)
        
        # joins (vol & ret & ratio sharpe)
        
        vol['pk_index'] = vol.index
        ret['pk_index'] = ret.index
        sharpe['pk_index'] = sharpe.index
        df_vol_ret = pd.merge(vol,ret,on='pk_index')
        df_vol_ret_sharpe=pd.merge(df_vol_ret,sharpe,on='pk_index')
        
        ## graph 3 ##
        
        st.subheader("Monte Carlo Simulation")
        
        dico_sharpe = {
            
            'Ratio (%)':['0 < Sharpe Ratio < 1','Sharpe Ratio > 1'],
            'Comment':["The return obtained is superior to that of a risk-free investment, but it remains insufficient",
                       "The performance achieved is better than the risk-free investment rate"]
                    
                     }
        
        sharpe_comment=pd.DataFrame(dico_sharpe)
        
        st.caption("Interpretation of the sharpe ratio value obtained")
        st.dataframe(sharpe_comment)
        
        st.caption('')
        
        st.markdown(f'*Best Annualized Sharpe ratio (%) : {best_result}*')
        
        graph_three = px.scatter(df_vol_ret_sharpe,x='Volatility',y='Return',
        color='(%) Sharpe_ratio',width=740, height=570)
        st.plotly_chart(graph_three)
        
        ### Correlation & volatility Analysis (daily return) ###
        
        ## graph 4 ##
        
        st.subheader("Correlation & Volatility analysis of daily returns")
        st.caption("Correlation Coefficient (the titles are strongly linked, when the value of the coefficient is closer to 1 or - 1)")
        st.dataframe(round(log_ret.corr(),2))
        st.caption("Correlation Matrix")
        graph_four = px.scatter_matrix(log_ret)
        graph_four.update_layout(width=675,height=600)
        graph_four.update_traces(marker_line_width=1,marker_line_color="black")
        st.plotly_chart(graph_four)
        
        st.caption("Box-Plot (volatility is lower for a title when the box-plot is tighter and when the standard deviation is smaller)")
        
        st.dataframe(round(log_ret.std().sort_values(ascending=False).rename('Standard Deviation (%) Daily Return'),3))
        
        graph_four_bis = px.box(log_ret)
        graph_four_bis.update_layout(yaxis={"title":"(%) Daily Returns"},xaxis={"title":"Title"},
                                     width=675,height=600)
        graph_four_bis.update_traces(marker_line_width=1,marker_line_color="black")
        st.plotly_chart(graph_four_bis)
        
        
        ### Portfolio optimization ###
        
        st.subheader("Portfolio optimization")
        
        def get_ret_vol_sr(weights):
            weights = np.array(weights)
            ret = np.sum(log_ret.mean()*weights)*252
            vol = np.sqrt(np.dot(weights.T,np.dot(log_ret.cov()*252,weights)))
            sr = ret/vol
            return np.array([ret,vol,sr])


        def neg_sharpe(weights):
            return get_ret_vol_sr(weights)[2]*-1
         

        def check_sum(weights):      
            return np.sum(weights)-1


        cons = ({'type':'eq','fun':check_sum})
        bounds = ((0,1),(0,1),(0,1),(0,1),(0,1))
        init_guess = [0.25,0.25,0.25,0.25,0.25]

        opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,
                              constraints=cons)


        result = [i for i in opt_results.x]
        key = [i for i in log_ret.columns]
        
        st.caption("Best Optimization")
        
        optimization = get_ret_vol_sr(opt_results.x)
        optimization = [i for i in optimization]
        key_optimization = ['Return','Volatility','Annualized Sharpe ratio (%)']
        dico_result_optimization = {'Critical Portfolio Parameters':key_optimization,'Values':optimization}
        
        df_optimization = pd.DataFrame(dico_result_optimization)
        
        df_optimization = round(df_optimization,2)
        
        st.dataframe(df_optimization)
        
        st.caption("Best Weights (Better distribution of money between the 5 stocks when investing)")

        dico_result_weights = {'Title':key,'Weights (%)':result}
        df_weights = pd.DataFrame(dico_result_weights)
        df_weights_order_by = round(df_weights.sort_values(by=['Weights (%)'],ascending=False),2)
        st.dataframe(df_weights_order_by)
        
        ## Show the gains ## 
        
        st.subheader("Simulation of the gains on this investment")
        
        list_of_price=list(range(0,100000,1000))
        
        dropdown_price = st.selectbox('Select the investment price (date of the beginning of the investment : ' + str(start) + ')',(list_of_price))       
        
        # Allocation calculation

        df_allocation = df_cum_second * [i for i in df_weights['Weights (%)']]

        # Position calculation 

        df_position = df_allocation * dropdown_price
        df_total_position = np.sum(df_position,axis=1)
        
        ## graph 5 ## 
        
        st.caption("Total Position")
        
        graph_five = px.line(df_total_position)
        graph_five.update_layout(yaxis={"title": "Price"},legend_title="Total Position",
        title={'text':"Total gain obtained between " +  str(start)  + " and " +  str(end) +" : "+str(round(df_total_position.iloc[-1]-df_total_position.iloc[0],1)) + " €",
        'y':0.92,'x':0.5,'xanchor': 'center','yanchor': 'top'})
        graph_five['layout']['title']['font'] = dict(size=15)
        st.plotly_chart(graph_five)
               
        ## graph 6 ## 
        
        st.caption("Detail Position")
        
        graph_six = px.line(df_position)
        graph_six.update_layout(yaxis={"title": "Price"},legend_title="Tickers",
        title={'text':"Total gain obtained between " +  str(start)  + " and " +  str(end) +" : "+str(round(df_total_position.iloc[-1]-df_total_position.iloc[0],1)) + " €",
        'y':0.92,'x':0.5,'xanchor': 'center','yanchor': 'top'})
        graph_six['layout']['title']['font'] = dict(size=15)
        st.plotly_chart(graph_six)
        
        
        
        
    

        
        

    
    
    