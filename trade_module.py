#!/usr/bin/env python
# coding: utf-8

# trade_module.py

# Imports

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle

import datetime
from scipy.stats import linregress
from IPython.display import Image
from statistics import mean
import random

# Plot Imports
import matplotlib.pyplot as plt 
import matplotlib.dates as mpl_dates
import plotly.graph_objects as go
import datetime

pd.set_option("display.max_rows", None, "display.max_columns", None)

''' ---------------------------------------------------------------------------------------------------------------------------'''

def Closing(df):
    '''Function that provides Close_Up/Close_Dw depending on how it closed in regards to the open.

    INPUTS:

     - df

    OUTPUTS:

     - ['Closing'] ==> output: col with 2 vbles: 1 if Close_Up & -1 if Close_Dw.

    '''
    df['Closing'] = df.apply(lambda x : 1 if x['Open'] <= x['Close'] else -1, axis=1)

''' ---------------------------------------------------------------------------------------------------------------------------'''

def Market_Structure(df,clean_sweep,swing_max,perc_95):
    
    '''Function that provides the Market Structure on the df adding the following cols outputs:

    INPUTS:

     - df

    OUTPUTS:

     - MS_H      ==> output:  value of Market structure High for each data point
     - MS_L      ==> output:  value of Market structure Low  for each data point
     - MS_Sit    ==> outputs are categorical vbles: 'MS', 'Settling','Up_Break', 'Dw_Break'
    
    perc_95 used for H1 is: 126.6 pips
    perc_95 used for H4 is: 259.1 pips
    perc_95 used for D1 is: 577.2 pips    
    '''
    
# Initiates 1st row;

    for i in range (0,10*swing_max):
        
        df.loc[df.index[i],'MS_L'] = df.loc[df.index[i],'Low']   
        df.loc[df.index[i],'MS_H'] = df.loc[df.index[i],'High']
        df.loc[df.index[i],'MS_Sit'] = 'MS'
 
    for i in range(10*swing_max,len(df)):
        
        i = i + df.index[0]
    
    # From 'MS'/'Dw_break' => 'Up_break' situation
        if (df.loc[i-1,'MS_Sit'] not in ('Up_Break')) & (df.loc[i,'Close'] > df.loc[i-1,'MS_H'] + 0.0001*clean_sweep):
            
            df.loc[i,'MS_Sit'] = 'Up_Break'
            df.loc[i,'MS_H']   = df.loc[i,'High']
            #df.loc[i,'MS_N']   = df.loc[i-1,'MS_N'] + 1

    # Id 1st Low swing before the break
            for j in range (0,1000):
                
                if (df.loc[(i-j),'Low'] >= df.loc[(i-j) -1 ,'Low']):
                    continue
                else:
                    if df.loc[(i-j),'Low'] < df.loc[i-j-swing_max:i-j-1,'Low'].min():
                        df.loc[i,'MS_L'] = df.loc[(i-j),'Low']
                        
    # Test that MS_range is under limit perc_95
                        if (10000 * (df.loc[i,'MS_H'] - df.loc[i,'MS_L'])) >= perc_95:
                            df.loc[i,'MS_L'] = df.loc[i,'MS_H'] - 0.0001 * perc_95
                        else:
                            pass
                        break
                    
                    else:
                        continue
        
    # From 'MS'/'Up_break' => 'Dw_break' situation
        elif (df.loc[i-1,'MS_Sit'] not in ('Dw_Break')) & (df.loc[i,'Close'] < df.loc[i-1,'MS_L']- 0.0001*clean_sweep):
            
            df.loc[i,'MS_Sit'] = 'Dw_Break'
            df.loc[i,'MS_L']   = df.loc[i,'Low']

    # Id 1st High swing before the break
            for j in range (0,1000):
                if (df.loc[(i-j),'High'] <= df.loc[(i-j) -1 ,'High']):
                    continue
                else:
                    if df.loc[(i-j),'High'] > df.loc[i-j-swing_max:i-j-1,'High'].max():
                        df.loc[i,'MS_H'] = df.loc[(i-j),'High']
                        
    # Test that MS_range is under limit perc_95
                        if (10000 * (df.loc[i,'MS_H'] - df.loc[i,'MS_L'])) >= perc_95:
                            df.loc[i,'MS_H'] = df.loc[i,'MS_L'] + 0.0001 * perc_95
                        else:
                            pass
                        break
                    else:
                        continue

    # From Up_Break => Up_Break / Settling_Up       
        elif (df.loc[i-1,'MS_Sit'] in ('Up_Break','Settling_Up')) & (df.loc[i,'High'] > df.loc[i-1,'High']):
            
            df.loc[i,'MS_Sit'] = 'Settling_Up'
            df.loc[i,'MS_H']   = df.loc[i,'High']
            
    # Test that MS_range is under limit perc_95
            if (10000 * (df.loc[i,'MS_H'] - df.loc[i-1,'MS_L'])) <= perc_95:
                df.loc[i,'MS_L']   = df.loc[i-1,'MS_L']   
            else:
                df.loc[i,'MS_L'] = df.loc[i,'MS_H'] - 0.0001 * perc_95

    # From Dw_Break => Dw_Break / Settling_Dw situation           
        elif (df.loc[i-1,'MS_Sit'] in ('Dw_Break','Settling_Dw')) & (df.loc[i,'Low'] < df.loc[i-1,'Low']):
            
            df.loc[i,'MS_Sit'] = 'Settling_Dw'
            df.loc[i,'MS_L']   = df.loc[i,'Low']
            
    # Test that MS_range is under limit perc_95
            if (10000 * (df.loc[i-1,'MS_H'] - df.loc[i,'MS_L'])) <= perc_95:
                df.loc[i,'MS_H']   = df.loc[i-1,'MS_H']   
            else:
                df.loc[i,'MS_H'] = df.loc[i,'MS_L'] + 0.0001 * perc_95

    # From MS => MS situation             
        else:
            
            df.loc[i,'MS_Sit'] = 'MS'
            df.loc[i,'MS_H']   = df.loc[i-1,'MS_H']
            df.loc[i,'MS_L']   = df.loc[i-1,'MS_L']
                
''' ---------------------------------------------------------------------------------------------------------------------------'''

def Trend(df):

    '''Function that calculates the trend based on the previous MS break:

    INPUTS:

     - df

    OUTPUTS:

     - Trend  ==> outputs are categorical vbles: 'Up', 'Dw' (depending on the value of the previous Break) 

    '''
    
    df.loc[df['MS_Sit'] == 'Up_Break','Trend'] =  1
    df.loc[df['MS_Sit'] == 'Dw_Break','Trend'] = -1
    
    for i in range(1,len(df)):
        
        i = i + df.index[0]
        
        if df.loc[i,'MS_Sit'] not in ('Dw_Break','Up_Break'):
            df.loc[i,'Trend'] = df.loc[i-1,'Trend']
            
        df['Trend'] = df['Trend'].fillna('none')

''' ---------------------------------------------------------------------------------------------------------------------------'''
        
def N_breaks(df):

    '''Function that calculates the number of previous breaks in the same direction:

    INPUTS:

     - df

    OUTPUTS:

     - N_Breaks ==> value of the the number of previous breaks in the same direction

    '''
    
    # Correction of ('Dw_Break','Up_Break')
    
    for i in range (1,len(df)):
        
        i = i + df.index[0]
        
        if (df.loc[i,'MS_Sit'] == 'Dw_Break') & (df.loc[i-1,'MS_Sit'] == 'Settling_Dw'):
            df.loc[i,'MS_Sit'] = 'Settling_Dw'
            
        elif (df.loc[i,'MS_Sit'] == 'Up_Break') & (df.loc[i-1,'MS_Sit'] == 'Settling_Up'):
            df.loc[i,'MS_Sit'] = 'Settling_Up'
        
        else:
            pass
            
    
    df['N_Breaks'] = 0
    
    for i in range (1,len(df)):
        
        i = i + df.index[0]
        
        if df.loc[i,'MS_Sit'] not in ('Dw_Break','Up_Break'):
            df.loc[i,'N_Breaks'] = df.loc[i-1,'N_Breaks']
            
        elif (df.loc[i,'MS_Sit'] == 'Up_Break'):
            if (df.loc[i-1,'Trend'] == -1):
                df.loc[i,'N_Breaks'] = 1
            else:
                df.loc[i,'N_Breaks'] = df.loc[i-1,'N_Breaks'] + 1
        
        elif (df.loc[i,'MS_Sit'] == 'Dw_Break'):
            if (df.loc[i-1,'Trend'] == 1):
                df.loc[i,'N_Breaks'] = -1
            else:
                df.loc[i,'N_Breaks'] = df.loc[i-1,'N_Breaks'] - 1
                
    df['N_Breaks'] = df['N_Breaks'].abs()

''' ---------------------------------------------------------------------------------------------------------------------------'''

def MS_periods(df):
    
    '''Function that calculates the number of periods the price remains in MS:

    INPUTS:

     - df

    OUTPUTS:

     - N_periods_MS ==> value of the range within the same MS

    '''
    
    df.loc[df.index[0],'MS_Pds'] = 0
    
    for i in range (1,len(df)):
        
        i = i + df.index[0]
        
        if (df.loc[i-1,'MS_Sit'] == 'MS') & (df.loc[i,'MS_Sit'] == 'MS'):
            df.loc[i,'MS_Pds'] = df.loc[i-1,'MS_Pds'] + 1
        
        elif (df.loc[i-1,'MS_Sit'] != 'MS') & (df.loc[i,'MS_Sit'] == 'MS'):
            df.loc[i,'MS_Pds'] = 1
        
        else:
            df.loc[i,'MS_Pds'] = 0
    
    # Converts column into integers
    df['MS_Pds'] = df['MS_Pds'].apply(np.int64)

''' ---------------------------------------------------------------------------------------------------------------------------'''

def MS_range(df):

    '''Function that calculates the number of pips between MS_High & MS_Low within MS:

    INPUTS:

     - df

    OUTPUTS:

     - MS_range ==> value of the range (MS_H - MS_L) within the same MS

    '''
    
    df['MS_range'] = 10000 * (df['MS_H'] - df['MS_L'])
                
''' ---------------------------------------------------------------------------------------------------------------------------'''

def MS_N(df):
    
    df.loc[df.index[0],'MS_N'] = 0
    for i in range(1,len(df)):

        i = i + df.index[0]

        if df.loc[i,'MS_Sit'] in ('Dw_Break','Up_Break'):
            df.loc[i,'MS_N']   = df.loc[i-1,'MS_N'] + 1

        else:
            df.loc[i,'MS_N']   = df.loc[i-1,'MS_N']

''' ---------------------------------------------------------------------------------------------------------------------------'''       
        
def Read_prep_df(File, n, sd, ed, clean_sweep, swing_max, perc_95):
    
    df = pd.read_excel(File)
    df = df.dropna()
    
    if sd != None:
        df = df.loc[df[df['Date'] == sd].index[0]:df[df['Date'] == ed].index[0]]
        df.reset_index(drop=True, inplace=True)
    else:
        pass   
    
    if n == 0:
        
        print(len(df))
        
        Closing(df) 
        Market_Structure(df, clean_sweep, swing_max, perc_95)
        Trend(df)
        N_breaks(df)
        MS_periods(df)
        MS_range(df)
        #MS_retracement(df)
        MS_N(df)
        Indicators(df)

    else:
        
        print(len(df))
        
        for i in range(0,1 + int(len(df)/n)):
        
            if i == 0:
            
                dfn = df.loc[(i*n) : ((i+1)*n) + 499,:].copy()
                
                Closing(dfn) 
                Market_Structure(dfn, clean_sweep, swing_max, perc_95)
                Trend(dfn)
                N_breaks(dfn)
                MS_periods(dfn)
                MS_range(dfn)
                #MS_retracement(dfn)
            
                df_out = dfn.copy()
                print(i)
                
            elif (i > 0) & (i < (1 + int(len(df)/n))):

                dfn = df.loc[i*n:((i+1)*n)+499,:].copy()

                Closing(dfn) 
                Market_Structure(dfn, clean_sweep, swing_max, perc_95)
                Trend(dfn)
                N_breaks(dfn)
                MS_periods(dfn)
                MS_range(dfn)
                #MS_retracement(dfn)

                dfn = dfn.loc[(i*n) + 500 : ((i+1)*n) + 499,:].copy()

                df_out = pd.concat([df_out,dfn])
                print(i)                
                
            else:

                dfn = df.loc[i*n:,:].copy()

                Closing(dfn) 
                Market_Structure(dfn, clean_sweep, swing_max, perc_95)
                Trend(dfn)
                N_breaks(dfn)
                MS_periods(dfn)
                MS_range(dfn)
                #MS_retracement(dfn)

                dfn = dfn.loc[(i*n) + 500 :,:].copy()

                df_out = pd.concat([df_out,dfn])
                print(i)
                
        df = df_out.copy()
        MS_N(df)
        Indicators(df)
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df

''' ---------------------------------------------------------------------------------------------------------------------------'''

def Rename_df(df,suffix):
    
    '''Function that renames all columns adding the suffix at the end

    INPUTS:

     - df     : Dataframe
     - suffix : '1D', '4H' or '15M' to rename the columns

    OUTPUTS:

     - New df with renamed columns

    '''

    keys = df.columns
    values = keys + '_' + suffix
    dictionary = dict(zip(keys, values))
    
    df = df.rename(columns=dictionary)
    
    return df

''' ---------------------------------------------------------------------------------------------------------------------------'''

def Merge_shift_df_H4_D1(df1,df2):
    
    '''Function that merges de 4H & 1D dataframes (df1 & df2)

    INPUTS:

     - df1    : df4H
     - df2    : df1D

    OUTPUTS:

     - New merged df

    '''

    df1.iloc[:, 0] = pd.to_datetime(df1.iloc[:, 0], dayfirst=True)
    df2.iloc[:, 0] = pd.to_datetime(df2.iloc[:, 0], dayfirst=True)
    
    df1['Date_del'] = df1.iloc[:, 0] + dt.timedelta(hours=4) 
    df2['Date_del'] = df2.iloc[:, 0] + dt.timedelta(days=1)

    df = pd.merge(df1, df2, how='left', on='Date_del')
    
    df1.drop(['Date_del'],axis=1,inplace=True)
    df2.drop(['Date_del'],axis=1,inplace=True)
    df. drop(['Date_del'],axis=1,inplace=True)
    
    df = df.ffill(axis=0)
    
    return df

''' ---------------------------------------------------------------------------------------------------------------------------'''

def Merge_shift_df_H1_H4(df1,df2):
    
    '''Function that merges de 15M & 4H dataframes (df1 & df2)

    INPUTS:

     - df1    : df1H
     - df2    : df4H or (dfH1_H4)

    OUTPUTS:

     - New merged df

    '''
    
    df1.iloc[:, 0] = pd.to_datetime(df1.iloc[:, 0], dayfirst=True)
    df2.iloc[:, 0] = pd.to_datetime(df2.iloc[:, 0], dayfirst=True)
    
    df1['Date_del'] = df1.iloc[:, 0] + dt.timedelta(hours=1) 
    df2['Date_del'] = df2.iloc[:, 0] + dt.timedelta(hours=4)
   
    df = pd.merge(df1, df2, how='left', on='Date_del')
    
    df['Date_H4'] = df['Date_H4'].ffill(axis=0)
    
    df1.drop(['Date_del'],axis=1,inplace=True)
    df2.drop(['Date_del'],axis=1,inplace=True)
    df. drop(['Date_del'],axis=1,inplace=True)
    
    df = df.ffill(axis=0)
    
    return df

''' ---------------------------------------------------------------------------------------------------------------------------'''

def Merge_shift_df_M15_H1(df1,df2):
    
    '''Function that merges de 15M & 4H dataframes (df1 & df2)

    INPUTS:

     - df1    : dfM15
     - df2    : dfH1 or (dfM15_H1)

    OUTPUTS:

     - New merged df

    '''
    
    df1.iloc[:, 0] = pd.to_datetime(df1.iloc[:, 0], dayfirst=True)
    df2.iloc[:, 0] = pd.to_datetime(df2.iloc[:, 0], dayfirst=True)
    
    df1['Date_del'] = df1.iloc[:, 0] + dt.timedelta(minutes=15) 
    df2['Date_del'] = df2.iloc[:, 0] + dt.timedelta(hours=1)
   
    df = pd.merge(df1, df2, how='left', on='Date_del')
    
    df['Date_H1'] = df['Date_H1'].ffill(axis=0)
    
    df1.drop(['Date_del'],axis=1,inplace=True)
    df2.drop(['Date_del'],axis=1,inplace=True)
    df. drop(['Date_del'],axis=1,inplace=True)
    
    df = df.ffill(axis=0)
    
    return df

''' ---------------------------------------------------------------------------------------------------------------------------'''

def MS_retracement(df):

    '''Function that calculates that the price has retraced (0,100)% between MS_High & MS_Low depending on its trend:

    INPUTS:

     - df

    OUTPUTS:

     - MS_retracement ==> value of the range (MS_H - MS_L) within the same MS
    '''
    
    df['MS_retracement_D1']  = df.apply(lambda x: 100 * (x['Close_M15'] - x['MS_L_D1'])  /(x['MS_H_D1']  - x['MS_L_D1'])  if x['Trend_D1']  == -1 else (100 * (x['MS_H_D1']  - x['Close_M15'])/(x['MS_H_D1']  - x['MS_L_D1'])),  axis=1).round(1)
    df['MS_retracement_H4']  = df.apply(lambda x: 100 * (x['Close_M15'] - x['MS_L_H4'])  /(x['MS_H_H4']  - x['MS_L_H4'])  if x['Trend_H4']  == -1 else (100 * (x['MS_H_H4']  - x['Close_M15'])/(x['MS_H_H4']  - x['MS_L_H4'])),  axis=1).round(1)
    df['MS_retracement_H1']  = df.apply(lambda x: 100 * (x['Close_M15'] - x['MS_L_H1'])  /(x['MS_H_H1']  - x['MS_L_H1'])  if x['Trend_H1']  == -1 else (100 * (x['MS_H_H1']  - x['Close_M15'])/(x['MS_H_H1']  - x['MS_L_H1'])),  axis=1).round(1)
    df['MS_retracement_M15'] = df.apply(lambda x: 100 * (x['Close_M15'] - x['MS_L_M15']) /(x['MS_H_M15'] - x['MS_L_M15']) if x['Trend_M15'] == -1 else (100 * (x['MS_H_M15'] - x['Close_M15'])/(x['MS_H_M15'] - x['MS_L_M15'])), axis=1).round(1)  

''' ---------------------------------------------------------------------------------------------------------------------------'''

def clean_data(df):
    
    from scipy import stats
    
# Drop non required columns
    RSI_SMA_1, RSI_EMA_1 = 14, 14

    non_req_cols = ['U_Move_M15','U_Move_H1','U_Move_H4','U_Move_D1','D_Move_M15','D_Move_H1','D_Move_H4','D_Move_D1',
                    'Close_im1_M15','Close_im1_H1','Close_im1_H4','Close_im1_D1',
                    str(RSI_SMA_1) + '_Avg_U_M15',str(RSI_SMA_1) + '_Avg_U_H1',str(RSI_SMA_1) + '_Avg_U_H4',str(RSI_SMA_1) + '_Avg_U_D1',
                    str(RSI_SMA_1) + '_Avg_D_M15',str(RSI_SMA_1) + '_Avg_D_H1',str(RSI_SMA_1) + '_Avg_D_H4',str(RSI_SMA_1) + '_Avg_D_D1',
                    str(RSI_EMA_1) + '_Avg_U_EMA_M15',str(RSI_EMA_1) + '_Avg_U_EMA_H1',str(RSI_EMA_1) + '_Avg_U_EMA_H4',str(RSI_EMA_1) + '_Avg_U_EMA_D1',
                    str(RSI_EMA_1) + '_Avg_D_EMA_M15',str(RSI_EMA_1) + '_Avg_D_EMA_H1',str(RSI_EMA_1) + '_Avg_D_EMA_H4',str(RSI_EMA_1) + '_Avg_D_EMA_D1',
                    #'Open_M15','High_M15','Low_M15','Close_M15','MS_L_M15','MS_H_M15','MS_Sit_M15',
                    #'Open_H1' ,'High_H1' ,'Low_H1' ,'Close_H1', 'MS_L_H1' ,'MS_H_H1' ,'MS_Sit_H1',
                    #'Open_H4' ,'High_H4' ,'Low_H4' ,'Close_H4', 'MS_L_H4' ,'MS_H_H4' ,'MS_Sit_H4',
                    #'Open_D1' ,'High_D1' ,'Low_D1' ,'Close_D1', 'MS_L_D1' ,'MS_H_D1' ,'MS_Sit_D1',
                    'B_Win_Idx_M15_H4','B_Lose_Idx_M15_H4','S_Win_Idx_M15_H4','S_Lose_Idx_M15_H4',
                    'B_Win_Idx_M15_H1','B_Lose_Idx_M15_H1','S_Win_Idx_M15_H1','S_Lose_Idx_M15_H1']
    
    print('No. columns BEFORE dropping:',df.shape[1])
    df.drop(non_req_cols,inplace = True, axis = 1)
    print('No. columns AFTER dropping:',df.shape[1])
     
# Drop None´s
    print('Length BEFORE removing None´s:',len(df))
    last_none_idx = df[df.isin(['none']).any(axis=1)].index[-1]
    print('Last row containing a None:',last_none_idx)
    df.drop(df.index[:last_none_idx],inplace=True)
    print('Length AFTER removing None´s:',len(df))
    
# Drop Nan´s
    print('Length BEFORE removing NaNs:',len(df))
    print('Number rows containing NaNs:',df.isnull().any(axis=1).sum())
    df.dropna(inplace=True)
    print('Length AFTER removing NaNs:',len(df))
    
# Remove Ouliars z = 3 (3 standard deviations)

    gen_cols_outliars = ['RSI_14_SMA_M15','RSI_14_SMA_H1','RSI_14_SMA_H4','RSI_14_SMA_D1',
                         'RSI_14_EMA_M15','RSI_14_EMA_H1','RSI_14_EMA_H4','RSI_14_EMA_D1',
                         'EMA_12_macd_M15','EMA_12_macd_H1','EMA_12_macd_H4','EMA_12_macd_D1',
                         'EMA_26_macd_M15','EMA_26_macd_H1','EMA_26_macd_H4','EMA_26_macd_D1',
                         'MACD_12_26_M15','MACD_12_26_H1', 'MACD_12_26_H4', 'MACD_12_26_D1',
                         'EMA_MACD_12_26_9_M15','EMA_MACD_12_26_9_H1', 'EMA_MACD_12_26_9_H4', 'EMA_MACD_12_26_9_D1',
                         'Hist_MACD_12_26_9_M15','Hist_MACD_12_26_9_H1', 'Hist_MACD_12_26_9_H4', 'Hist_MACD_12_26_9_D1',
                         'MACD_signal_M15','MACD_signal_H1', 'MACD_signal_H4', 'MACD_signal_D1',
                         'Boll_SMA_20_M15','Boll_SMA_20_H1', 'Boll_SMA_20_H4', 'Boll_SMA_20_D1',
                         'Boll_SMA_20_Var_-2_M15','Boll_SMA_20_Var_-2_H1', 'Boll_SMA_20_Var_-2_H4', 'Boll_SMA_20_Var_-2_D1',
                         'Boll_SMA_20_Var_+2_M15','Boll_SMA_20_Var_+2_H1', 'Boll_SMA_20_Var_+2_H4', 'Boll_SMA_20_Var_+2_D1',
                         'Dist_BollB_SMA_20_Var_+2_M15','Dist_BollB_SMA_20_Var_+2_H1', 'Dist_BollB_SMA_20_Var_+2_H4', 'Dist_BollB_SMA_20_Var_+2_D1',
                         'Dist_BollB_SMA_20_Var_-2_M15','Dist_BollB_SMA_20_Var_-2_H1', 'Dist_BollB_SMA_20_Var_-2_H4', 'Dist_BollB_SMA_20_Var_-2_D1',
                         'BollB_Wideness_M15','BollB_Wideness_H1', 'BollB_Wideness_H4', 'BollB_Wideness_D1',]

    print('Length BEFORE removing Ouliars:',len(df))
    df = df[(np.abs(stats.zscore(df[gen_cols_outliars])) < 3).all(axis=1)]
    print('Length AFTER removing Ouliars:',len(df))

    df = df.reset_index()

    return df
        
''' ---------------------------------------------------------------------------------------------------------------------------'''

def plot_MS(df, date, rng):
    
    p = df[df['Date'] == date].index[0]
    i = p
    j = p + rng
    
    dfp = df.loc[i:j,['Date', 'Open', 'High', 'Low', 'Close']]
    
    fig = go.Figure(data=[go.Candlestick(x=dfp['Date'],
                    open=dfp['Open'],
                    high=dfp['High'],
                    low=dfp['Low'],
                    close=dfp['Close'])])

    fig.update_layout(width=1000, height=1000,margin=dict(l=0, r=20, b=100, t=20, pad=4))
    
    for k in range(i,j):
        
        fig.add_shape(
                # Line Horizontal - ['MS_H']
                    type="line",
                    x0 = df.loc[k,'Date'],
                    y0 = df.loc[k,'MS_H'],
                    x1 = df.loc[k+1,'Date'],
                    y1 = df.loc[k,'MS_H'],
                    line=dict(
                        color="Green",
                        width=4,
                        dash="dashdot",
                    ))

        fig.add_shape(
                # Line Horizontal - ['MS_L']
                    type="line",
                    x0 = df.loc[k,'Date'],
                    y0 = df.loc[k,'MS_L'],
                    x1 = df.loc[k+1,'Date'],
                    y1 = df.loc[k,'MS_L'],
                    line=dict(
                        color="Blue",
                        width=4,
                        dash="dashdot",
                    ))
    
    print(df.loc[p,'Date'])
    fig.show()

''' ---------------------------------------------------------------------------------------------------------------------------'''

def plot_MS_all(df, date, rng):
    
    p = df[df['Date_M15'] == date].index[0]
    i = p - rng
    j = p + rng
    
    dfp = df.loc[i:j,['Date_M15', 'Open_M15', 'High_M15', 'Low_M15', 'Close_M15']]
    
    fig = go.Figure(data=[go.Candlestick(x=dfp['Date_M15'],
                    open=dfp['Open_M15'],
                    high=dfp['High_M15'],
                    low=dfp['Low_M15'],
                    close=dfp['Close_M15'])])

    fig.update_layout(width=1000, height=2000,margin=dict(l=0, r=20, b=100, t=20, pad=4))    
    
    for k in range(i,j):
        
        fig.add_shape(
                # Line Horizontal - ['MS_H_M15']
                    type="line",
                    x0 = df.loc[k,'Date_M15'],
                    y0 = df.loc[k,'MS_H_M15'],
                    x1 = df.loc[k+1,'Date_M15'],
                    y1 = df.loc[k,'MS_H_M15'],
                    line=dict(
                        color="Black",
                        width=4,
                        dash="dashdot",
                    ))

        fig.add_shape(
                # Line Horizontal - ['MS_L_M15']
                    type="line",
                    x0 = df.loc[k,'Date_M15'],
                    y0 = df.loc[k,'MS_L_M15'],
                    x1 = df.loc[k+1,'Date_M15'],
                    y1 = df.loc[k,'MS_L_M15'],
                    line=dict(
                        color="Black",
                        width=4,
                        dash="dashdot",
                    ))
        
        fig.add_shape(
                # Line Horizontal - ['MS_H_H1']
                    type="line",
                    x0 = df.loc[k,'Date_M15'],
                    y0 = df.loc[k,'MS_H_H1'],
                    x1 = df.loc[k+1,'Date_M15'],
                    y1 = df.loc[k,'MS_H_H1'],
                    line=dict(
                        color="Blue",
                        width=4,
                        dash="dashdot",
                    ))

        fig.add_shape(
                # Line Horizontal - ['MS_L_H1']
                    type="line",
                    x0 = df.loc[k,'Date_M15'],
                    y0 = df.loc[k,'MS_L_H1'],
                    x1 = df.loc[k+1,'Date_M15'],
                    y1 = df.loc[k,'MS_L_H1'],
                    line=dict(
                        color="Blue",
                        width=4,
                        dash="dashdot",
                    ))
        
        fig.add_shape(
                # Line Horizontal - ['MS_H_H4']
                    type="line",
                    x0 = df.loc[k,'Date_M15'],
                    y0 = df.loc[k,'MS_H_H4'],
                    x1 = df.loc[k+1,'Date_M15'],
                    y1 = df.loc[k,'MS_H_H4'],
                    line=dict(
                        color="Red",
                        width=4,
                        dash="dashdot",
                    ))

        fig.add_shape(
                # Line Horizontal - ['MS_L_H4']
                    type="line",
                    x0 = df.loc[k,'Date_M15'],
                    y0 = df.loc[k,'MS_L_H4'],
                    x1 = df.loc[k+1,'Date_M15'],
                    y1 = df.loc[k,'MS_L_H4'],
                    line=dict(
                        color="Red",
                        width=4,
                        dash="dashdot",
                    ))
        
        fig.add_shape(
                # Line Horizontal - ['MS_H_D1']
                    type="line",
                    x0 = df.loc[k,'Date_M15'],
                    y0 = df.loc[k,'MS_H_D1'],
                    x1 = df.loc[k+1,'Date_M15'],
                    y1 = df.loc[k,'MS_H_D1'],
                    line=dict(
                        color="Yellow",
                        width=4,
                        dash="dashdot",
                    ))

        fig.add_shape(
                # Line Horizontal - ['MS_L_D1']
                    type="line",
                    x0 = df.loc[k,'Date_M15'],
                    y0 = df.loc[k,'MS_L_D1'],
                    x1 = df.loc[k+1,'Date_M15'],
                    y1 = df.loc[k,'MS_L_D1'],
                    line=dict(
                        color="Yellow",
                        width=4,
                        dash="dashdot",
                    ))
    
    print(df.loc[p,'Date_H1'])
    fig.show()

''' ---------------------------------------------------------------------------------------------------------------------------'''

def Price_M15_H1(df,k,ratio,pip_min,pip_max,pip_over):
    
    '''Function that prices all M15 candles based on [Close_M15].shift() value and calcs. its stop loss &
                limits based on MS_H1.
    INPUTS:

     - df    : Dataframe
     - ratio : limit:stop_loss ratio (normally 3)

    OUTPUTS:

     - Price_M15          ==> output: value of [Close_M15].shift()
     - B_Stop_Loss_M15_H1 ==> output: # pips between ([Price_M15] - [MS_L_H1])
     - B_Limit_M15_H1     ==> output: # pips between (ratio x [B_Stop_Loss_M15_H1])
     - S_Stop_Loss_M15_H1 ==> output: # pips between ([MS_H_H1] - [Price_M15])
     - S_Limit_M15_H1     ==> output: # pips between (ratio x [S_Stop_Loss_M15_H1])
     - Labelb_M15         ==> output: 1 or 0 whether the action of buying  was succesful or not
     - Labels_M15         ==> output: 1 or 0 whether the action of selling was succesful or not    
    '''
    
# Price
    df['Price_M15']          = df['Close_M15']
    df['B_Stop_Loss_M15_H1'] = (10000 * (df['Price_M15'] - df['MS_L_H1'])) - pip_over
    df['B_Stop_Loss_M15_H1'] = df.apply(lambda x: x['B_Stop_Loss_M15_H1'] if x['B_Stop_Loss_M15_H1'] < pip_max else pip_max ,axis = 1)
    df['B_Stop_Loss_M15_H1'] = df.apply(lambda x: x['B_Stop_Loss_M15_H1'] if x['B_Stop_Loss_M15_H1'] > pip_min else pip_min ,axis = 1)
    df['B_Limit_M15_H1']     = ratio * df['B_Stop_Loss_M15_H1']
    df['S_Stop_Loss_M15_H1'] = 10000 * (df['MS_H_H1'] - df['Price_M15']) - pip_over
    df['S_Stop_Loss_M15_H1'] = df.apply(lambda x: x['S_Stop_Loss_M15_H1'] if x['S_Stop_Loss_M15_H1'] < pip_max else pip_max ,axis = 1)
    df['S_Stop_Loss_M15_H1'] = df.apply(lambda x: x['S_Stop_Loss_M15_H1'] if x['S_Stop_Loss_M15_H1'] > pip_min else pip_min ,axis = 1)
    df['S_Limit_M15_H1']     = ratio * df['S_Stop_Loss_M15_H1']

    m = 0
    for i, row in df.iterrows():
        
        if i < len(df)-k:
            m = k
        else:
            m = len(df) - i - 3
        print('Price_M15_H1 - ',int(100*((i/20000)*20000/len(df))),'%') if ((i > 0) & (i%20000 == 0)) else ''

# BUY - Losing Index
        
        if len(np.where((df.loc[i+1:i+m,'Low_M15'] <= row['Price_M15'] - (0.0001*(row['B_Stop_Loss_M15_H1']))))[0]) == 0:
            df.loc[i, 'B_Lose_Idx_M15_H1'] = 1000000
        else:
            df.loc[i, 'B_Lose_Idx_M15_H1'] = np.where((df.loc[i+1:i+m,'Low_M15'] <= row['Price_M15'] - (0.0001*(row['B_Stop_Loss_M15_H1']))))[0][0]       
                       
# BUY - Winning Index
        
        if len(np.where((df.loc[i+1:i+m,'High_M15'] >= row['Price_M15'] + (0.0001*(row['B_Limit_M15_H1']))))[0]) == 0:
            df.loc[i, 'B_Win_Idx_M15_H1'] = 1000000

        else:
            df.loc[i, 'B_Win_Idx_M15_H1'] = np.where((df.loc[i+1:i+m,'High_M15'] >= row['Price_M15'] + (0.0001*(row['B_Limit_M15_H1']))))[0][0]

# SELL - Losing Index
        
        if len(np.where((df.loc[i+1:i+m,'High_M15'] >= row['Price_M15'] + (0.0001*(row['S_Stop_Loss_M15_H1']))))[0]) == 0:
            df.loc[i, 'S_Lose_Idx_M15_H1'] = 1000000

        else:
            df.loc[i, 'S_Lose_Idx_M15_H1'] = np.where((df.loc[i+1:i+m,'High_M15'] >= row['Price_M15'] + (0.0001*(row['S_Stop_Loss_M15_H1']))))[0][0]

# SELL - Winning Index
        
        if len(np.where((df.loc[i+1:i+m,'Low_M15'] <= row['Price_M15'] - (0.0001*(row['S_Limit_M15_H1']))))[0]) == 0:
            df.loc[i, 'S_Win_Idx_M15_H1'] = 1000000

        else:
            df.loc[i, 'S_Win_Idx_M15_H1'] = np.where((df.loc[i+1:i+m,'Low_M15'] <= row['Price_M15'] - (0.0001*(row['S_Limit_M15_H1']))))[0][0]

# Label
# 1 if B_Win Index < B_Lose Index
    df['Labelb_M15_H1'] = df.apply(lambda x: 1 if x['B_Win_Idx_M15_H1'] < x['B_Lose_Idx_M15_H1'] else 0, axis=1)

# -1 if S_Win Index < S_Lose Index  
    df['Labels_M15_H1'] = df.apply(lambda x: 1 if x['S_Win_Idx_M15_H1'] < x['S_Lose_Idx_M15_H1'] else 0, axis=1) 
    
# Drops all index    
    #df.drop(['B_Lose_Idx_M15_H1','B_Win_Idx_M15_H1','S_Lose_Idx_M15_H1','S_Win_Idx_M15_H1'],axis = 1, inplace =True)
    
    return df

''' ---------------------------------------------------------------------------------------------------------------------------'''

def Price_M15_H4(df,k,ratio,pip_min,pip_max,pip_over):
    
    '''Function that prices all M15 candles based on [Close_M15].shift() value and calcs. its stop loss &
                limits based on MS_H1.
    INPUTS:

     - df    : Dataframe
     - ratio : limit:stop_loss ratio (normally 3)

    OUTPUTS:

     - Price_m15          ==> output: value of [Close_M15].shift()
     - B_Stop_Loss_M15_H4 ==> output: # pips between ([Price_M15] - [MS_L_H4])
     - B_Limit_M15_H4     ==> output: # pips between (ratio x [B_Stop_Loss_M15_H4])
     - S_Stop_Loss_M15_H4 ==> output: # pips between ([MS_H_H4] - [Price_M15])
     - S_Limit_M15_H4     ==> output: # pips between (ratio x [S_Stop_Loss_M15_H4])
     - Labelb_M15         ==> output: 1 or 0 whether the action of buying  was succesful or not
     - Labels_M15         ==> output: 1 or 0 whether the action of selling was succesful or not    
    '''
    
# Price
    df['Price_M15']          = df['Close_M15']
    df['B_Stop_Loss_M15_H4'] = (10000 * (df['Price_M15'] - df['MS_L_H4'])) - pip_over
    df['B_Stop_Loss_M15_H4'] = df.apply(lambda x: x['B_Stop_Loss_M15_H4'] if x['B_Stop_Loss_M15_H4'] < pip_max else pip_max ,axis = 1)
    df['B_Stop_Loss_M15_H4'] = df.apply(lambda x: x['B_Stop_Loss_M15_H4'] if x['B_Stop_Loss_M15_H4'] > pip_min else pip_min ,axis = 1)
    df['B_Limit_M15_H4']     = ratio * df['B_Stop_Loss_M15_H4']
    df['S_Stop_Loss_M15_H4'] = 10000 * (df['MS_H_H4'] - df['Price_M15']) - pip_over
    df['S_Stop_Loss_M15_H4'] = df.apply(lambda x: x['S_Stop_Loss_M15_H4'] if x['S_Stop_Loss_M15_H4'] < pip_max else pip_max ,axis = 1)
    df['S_Stop_Loss_M15_H4'] = df.apply(lambda x: x['S_Stop_Loss_M15_H4'] if x['S_Stop_Loss_M15_H4'] > pip_min else pip_min ,axis = 1)
    df['S_Limit_M15_H4']     = ratio * df['S_Stop_Loss_M15_H4']

    m = 0
    for i, row in df.iterrows():
        
        if i < len(df)-k:
            m = k
        else:
            m = len(df) - i - 3
        print('Price_M15_H4 - ',int(100*((i/10000)*10000/len(df))),'%') if ((i > 0) & (i%10000 == 0)) else ''

# BUY - Losing Index
        
        if len(np.where((df.loc[i+1:i+m,'Low_M15'] <= row['Price_M15'] - (0.0001*(row['B_Stop_Loss_M15_H4']))))[0]) == 0:
            df.loc[i, 'B_Lose_Idx_M15_H4'] = 1000000
        else:
            df.loc[i, 'B_Lose_Idx_M15_H4'] = np.where((df.loc[i+1:i+m,'Low_M15'] <= row['Price_M15'] - (0.0001*(row['B_Stop_Loss_M15_H4']))))[0][0]       
                       
# BUY - Winning Index
        
        if len(np.where((df.loc[i+1:i+m,'High_M15'] >= row['Price_M15'] + (0.0001*(row['B_Limit_M15_H4']))))[0]) == 0:
            df.loc[i, 'B_Win_Idx_M15_H4'] = 1000000

        else:
            df.loc[i, 'B_Win_Idx_M15_H4'] = np.where((df.loc[i+1:i+m,'High_M15'] >= row['Price_M15'] + (0.0001*(row['B_Limit_M15_H4']))))[0][0]

# SELL - Losing Index
        
        if len(np.where((df.loc[i+1:i+m,'High_M15'] >= row['Price_M15'] + (0.0001*(row['S_Stop_Loss_M15_H4']))))[0]) == 0:
            df.loc[i, 'S_Lose_Idx_M15_H4'] = 1000000

        else:
            df.loc[i, 'S_Lose_Idx_M15_H4'] = np.where((df.loc[i+1:i+m,'High_M15'] >= row['Price_M15'] + (0.0001*(row['S_Stop_Loss_M15_H4']))))[0][0]

# SELL - Winning Index
        
        if len(np.where((df.loc[i+1:i+m,'Low_M15'] <= row['Price_M15'] - (0.0001*(row['S_Limit_M15_H4']))))[0]) == 0:
            df.loc[i, 'S_Win_Idx_M15_H4'] = 1000000

        else:
            df.loc[i, 'S_Win_Idx_M15_H4'] = np.where((df.loc[i+1:i+m,'Low_M15'] <= row['Price_M15'] - (0.0001*(row['S_Limit_M15_H4']))))[0][0]

# Label
# 1 if B_Win Index < B_Lose Index
    df['Labelb_M15_H4'] = df.apply(lambda x: 1 if x['B_Win_Idx_M15_H4'] < x['B_Lose_Idx_M15_H4'] else 0, axis=1)

# -1 if S_Win Index < S_Lose Index  
    df['Labels_M15_H4'] = df.apply(lambda x: 1 if x['S_Win_Idx_M15_H4'] < x['S_Lose_Idx_M15_H4'] else 0, axis=1) 
    
# Drops all index    
    #df.drop(['B_Lose_Idx_M15_H4','B_Win_Idx_M15_H4','S_Lose_Idx_M15_H4','S_Win_Idx_M15_H4'],axis = 1, inplace =True)
    
    return df

''' ---------------------------------------------------------------------------------------------------------------------------'''

def Auto_Analysis(df, itr, M15_H1, prt_ret, a1b, a2b, b1b, b2b, c1b, c2b, d1b, d2b, r1, r2, s1, s2, t1, t2, u1, u2, n1, n2, o1, o2, p1, p2, q1, q2): 
    
    a1s, a2s, b1s, b2s, c1s, c2s, d1s, d2s = -a1b, -a2b, -b1b, -b2b, -c1b, -c2b, -d1b, -d2b   
    df3, df2, df1 = split3_dfs(df)    

    list_idx_1 = df1.index.to_list()
    list_idx_2 = df2.index.to_list()
    list_idx_3 = df3.index.to_list()

    # Initialising lists

    xpa1, xca1, xpb1, xcb1 = [], [], [], []
    xpa2, xca2, xpb2, xcb2 = [], [], [], []
    xpa3, xca3, xpb3, xcb3 = [], [], [], []

    ypa1, yca1, ypb1, ycb1 = [], [], [], []
    ypa2, yca2, ypb2, ycb2 = [], [], [], []
    ypa3, yca3, ypb3, ycb3 = [], [], [], []

    def analysis_b(df):

        x = df[(df['MS_retracement_M15'].between(-1000000, 1000000)) & 
           ((df['Trend_D1']  == a1b) | (df['Trend_D1']  == a2b)) & 
           ((df['Trend_H4']  == b1b) | (df['Trend_H4']  == b2b)) &
           ((df['Trend_H1']  == c1b) | (df['Trend_H1']  == c2b)) &
           ((df['Trend_M15'] == d1b) | (df['Trend_M15'] == d2b)) &
           #((df['Closing_D1'] == e1b) | (df['Closing_D1'] == e2b)) &
           #((df['Closing_H4'] == f1b) | (df['Closing_H4'] == f2b)) &
           #((df['Closing_H1'] == g1b) | (df['Closing_H1'] == g2b)) &
           #((df['Closing_H1'] == h1b) | (df['Closing_H1'] == h2b)) &
           (df['MS_retracement_D1'] .between(r1, r2)) &
           (df['MS_retracement_H4'] .between(s1, s2)) & 
           (df['MS_retracement_H1'] .between(t1, t2)) &
           (df['MS_retracement_M15'].between(u1, u2)) &
           #(df['N_Breaks_D1'] .between(k1, k2)) &
           #(df['N_Breaks_H4'] .between(l1, l2)) & 
           #(df['N_Breaks_H1'] .between(m1, m2)) &
           (df['MS_range_D1'] .between(n1, n2)) & 
           (df['MS_range_H4'] .between(o1, o2)) & 
           (df['MS_range_H1'] .between(p1, p2)) &
           (df['MS_range_M15'].between(q1, q2)) &
           (df['MS_retracement_M15'].between(-1000000, 1000000))]['Labelb_'+str(M15_H1)].describe()[:2]

        return x

    def analysis_s(df):
        
        y = df[(df['MS_retracement_M15'].between(-1000000, 1000000)) & 
           ((df['Trend_D1']  == a1s) | (df['Trend_D1']  == a2s)) & 
           ((df['Trend_H4']  == b1s) | (df['Trend_H4']  == b2s)) &
           ((df['Trend_H1']  == c1s) | (df['Trend_H1']  == c2s)) &
           ((df['Trend_M15'] == d1s) | (df['Trend_M15'] == d2s)) &
           #((df['Closing_D4'] == e1s) | (df['Closing_D1'] == e2s)) &
           #((df['Closing_H4'] == f1s) | (df['Closing_H4'] == f2s)) &
           #((df['Closing_H1'] == g1s) | (df['Closing_H1'] == g2s)) &
           #((df['Closing_H1'] == h1s) | (df['Closing_H1'] == h2s)) &
           (df['MS_retracement_D1'] .between(r1, r2)) &
           (df['MS_retracement_H4'] .between(s1, s2)) & 
           (df['MS_retracement_H1'] .between(t1, t2)) &
           (df['MS_retracement_M15'].between(u1, u2)) &
           #(df['N_Breaks_D1'] .between(k1, k2)) &
           #(df['N_Breaks_H4'] .between(l1, l2)) & 
           #(df['N_Breaks_H1'] .between(m1, m2)) &
           (df['MS_range_D1'] .between(n1, n2)) & 
           (df['MS_range_H4'] .between(o1, o2)) & 
           (df['MS_range_H1'] .between(p1, p2)) &
           (df['MS_range_M15'].between(q1, q2)) &
           (df['MS_retracement_M15'].between(-1000000, 1000000))]['Labels_'+str(M15_H1)].describe()[:2]

        return y
    
    if prt_ret == 'print':

        for i in range(0,itr):

            df1a = df1.loc[Space_dfa(df1,10),:]
            df1b = df1.loc[Space_dfb(list_idx_1,int(len(df1)/10)),:]
            df2a = df2.loc[Space_dfa(df2,10),:]
            df2b = df2.loc[Space_dfb(list_idx_2,int(len(df2)/10)),:]
            df3a = df3.loc[Space_dfa(df3,10),:]
            df3b = df3.loc[Space_dfb(list_idx_3,int(len(df3)/10)),:]

            xa1, xb1 = analysis_b(df1a), analysis_b(df1b)
            xa2, xb2 = analysis_b(df2a), analysis_b(df2b)
            xa3, xb3 = analysis_b(df3a), analysis_b(df3b)

            ya1, yb1 = analysis_s(df1a), analysis_s(df1b)
            ya2, yb2 = analysis_s(df2a), analysis_s(df2b)
            ya3, yb3 = analysis_s(df3a), analysis_s(df3b)

            xpa1.append(xa1[1])
            xca1.append(xa1[0])
            xpb1.append(xb1[1])
            xcb1.append(xb1[0])

            xpa2.append(xa2[1])
            xca2.append(xa2[0])
            xpb2.append(xb2[1])
            xcb2.append(xb2[0])

            xpa3.append(xa3[1])
            xca3.append(xa3[0])
            xpb3.append(xb3[1])
            xcb3.append(xb3[0])

            ypa1.append(ya1[1])
            yca1.append(ya1[0])
            ypb1.append(yb1[1])
            ycb1.append(yb1[0])

            ypa2.append(ya2[1])
            yca2.append(ya2[0])
            ypb2.append(yb2[1])
            ycb2.append(yb2[0])

            ypa3.append(ya3[1])
            yca3.append(ya3[0])
            ypb3.append(yb3[1])
            ycb3.append(yb3[0])

        count_b = round((mean(xca1)+mean(xca2)+mean(xca3)+mean(xcb1)+mean(xcb2)+mean(xcb3))/6,2)
        count_s = round((mean(yca1)+mean(yca2)+mean(yca3)+mean(ycb1)+mean(ycb2)+mean(ycb3))/6,2)

        avg_b  = round((mean(xpa1)*mean(xca1) + mean(xpa2)*mean(xca2) + mean(xpa3)*mean(xca3)+mean(xpb1)*mean(xcb1) + mean(xpb2)*mean(xcb2) + mean(xpb3)*mean(xcb3))/(6*count_b),2)   
        avg_s  = round((mean(ypa1)*mean(yca1) + mean(ypa2)*mean(yca2) + mean(ypa3)*mean(yca3)+mean(ypb1)*mean(ycb1) + mean(ypb2)*mean(ycb2) + mean(ypb3)*mean(ycb3))/(6*count_s),2)   

        print('TOTAL SCORE BUY  --- MEAN:',avg_b,' --- COUNT:',count_b)
        print('TOTAL SCORE SELL --- MEAN:',avg_s,' --- COUNT:',count_s)
        print('BUY VBLES   :','(',a1b,a2b,')(',b1b,b2b,')(',c1b,c2b,')(',d1b,d2b,')')
        print('SELL VBLES  :','(',a1s,a2s,')(',b1s,b2s,')(',c1s,c2s,')(',d1s,d2s,')')
        print('COMMON VBLES:','(',r1,r2,')(',s1,s2,')(',t1,t2,')(',u1,u2,')n(',n1,n2,')o(',o1,o2,')p(',p1,p2,')q(',q1,q2,')')
        print('BUY DFRAMES:')
        print('df1a',round(mean(xpa1),2),round(mean(xca1),2),' - df1b',round(mean(xpb1),2),round(mean(xcb1),2))
        print('df2a',round(mean(xpa2),2),round(mean(xca2),2),' - df2b',round(mean(xpb2),2),round(mean(xcb2),2))
        print('df3a',round(mean(xpa3),2),round(mean(xca3),2),' - df3b',round(mean(xpb3),2),round(mean(xcb3),2))
        print('Tot score:')
        print('mean:',avg_b,'   count:',count_b)
        print('SELL DFRAMES:')
        print('df1a',round(mean(ypa1),2),round(mean(yca1),2),' - df1b',round(mean(ypb1),2),round(mean(ycb1),2))
        print('df2a',round(mean(ypa2),2),round(mean(yca2),2),' - df2b',round(mean(ypb2),2),round(mean(ycb2),2))
        print('df3a',round(mean(ypa3),2),round(mean(yca3),2),' - df3b',round(mean(ypb3),2),round(mean(ycb3),2))
        print('Tot score:')
        print('mean:',avg_s,'   count:',count_s)
        
    else:
        x = df[(df['MS_retracement_M15'].between(-1000000, 1000000)) & 
           ((df['Trend_D1']  == a1b) | (df['Trend_D1']  == a2b)) & 
           ((df['Trend_H4']  == b1b) | (df['Trend_H4']  == b2b)) &
           ((df['Trend_H1']  == c1b) | (df['Trend_H1']  == c2b)) &
           ((df['Trend_M15'] == d1b) | (df['Trend_M15'] == d2b)) &
           #((df['Closing_D4'] == e1b) | (df['Closing_D1'] == e2b)) &
           #((df['Closing_H4'] == f1b) | (df['Closing_H4'] == f2b)) &
           #((df['Closing_H1'] == g1b) | (df['Closing_H1'] == g2b)) &
           #((df['Closing_H1'] == h1b) | (df['Closing_H1'] == h2b)) &
           (df['MS_retracement_D1'] .between(r1, r2)) &
           (df['MS_retracement_H4'] .between(s1, s2)) & 
           (df['MS_retracement_H1'] .between(t1, t2)) &
           (df['MS_retracement_M15'].between(u1, u2)) &
           #(df['N_Breaks_D1'] .between(k1, k2)) &
           #(df['N_Breaks_H4'] .between(l1, l2)) & 
           #(df['N_Breaks_H1'] .between(m1, m2)) &
           (df['MS_range_D1'] .between(n1, n2)) & 
           (df['MS_range_H4'] .between(o1, o2)) & 
           (df['MS_range_H1'] .between(p1, p2)) &
           (df['MS_range_M15'].between(q1, q2)) &
           (df['MS_retracement_M15'].between(-1000000, 1000000))]

        y = df[(df['MS_retracement_M15'].between(-1000000, 1000000)) & 
           ((df['Trend_D1']  == a1s) | (df['Trend_D1']  == a2s)) & 
           ((df['Trend_H4']  == b1s) | (df['Trend_H4']  == b2s)) &
           ((df['Trend_H1']  == c1s) | (df['Trend_H1']  == c2s)) &
           ((df['Trend_M15'] == d1s) | (df['Trend_M15'] == d2s)) &
           #((df['Closing_D4'] == e1s) | (df['Closing_D1'] == e2s)) &
           #((df['Closing_H4'] == f1s) | (df['Closing_H4'] == f2s)) &
           #((df['Closing_H1'] == g1s) | (df['Closing_H1'] == g2s)) &
           #((df['Closing_H1'] == h1s) | (df['Closing_H1'] == h2s)) &
           (df['MS_retracement_D1'] .between(r1, r2)) &
           (df['MS_retracement_H4'] .between(s1, s2)) & 
           (df['MS_retracement_H1'] .between(t1, t2)) &
           (df['MS_retracement_M15'].between(u1, u2)) &
           #(df['N_Breaks_D1'] .between(k1, k2)) &
           #(df['N_Breaks_H4'] .between(l1, l2)) & 
           #(df['N_Breaks_H1'] .between(m1, m2)) &
           (df['MS_range_D1'] .between(n1, n2)) & 
           (df['MS_range_H4'] .between(o1, o2)) & 
           (df['MS_range_H1'] .between(p1, p2)) &
           (df['MS_range_M15'].between(q1, q2)) &
           (df['MS_retracement_M15'].between(-1000000, 1000000))]
        
        return x, y

''' ---------------------------------------------------------------------------------------------------------------------------'''

def split3_dfs(df):
    
    df1 = df.loc[0:int(len(df)/3),:].copy()
    df2 = df.loc[int(  len(df)/3) +1:int(2*len(df)/3),:].copy()
    df3 = df.loc[int(2*len(df)/3) +1:,:].copy()
    
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df3 = df3.reset_index(drop=True)
    
    return df1, df2, df3

''' ---------------------------------------------------------------------------------------------------------------------------'''

# Space_dfa - Function that spaces dataframe rows by a distance k

import random

def Space_dfa(df,k):
    
    l1 = df.index.to_list()
    r  = random.randint(0, k)
    l2 = [r]
    
    for i in range(r+1,len(df),k):
        l2.append(i)
    
    return l2

''' ---------------------------------------------------------------------------------------------------------------------------'''

# Space_dfb - Function that picks 'size' number of random rows from dataframe

import random

def Space_dfb(l1,size):

    l2 = random.sample(l1, size)
    l2.sort()
    return l2

''' ---------------------------------------------------------------------------------------------------------------------------'''

# Identify when these point happen

def N_months(df):
    
    return pd.to_datetime(pd.Series(df['Date_M15']), format = '%Y%m%d').apply(lambda x: x.strftime('%Y-%m')).nunique()
    
def diff_month(df):
    
    return ((df.reset_index().loc[len(df.reset_index())-1,'Date_M15'].year - df.reset_index().loc[0,'Date_M15'].year) * 12) + (df.reset_index().loc[len(df.reset_index())-1,'Date_M15'].month - df.reset_index().loc[0,'Date_M15'].month)    
    
def N_months_concat(df1, df2):
    
    s1 = pd.to_datetime(pd.Series(df1['Date_M15']), format = '%Y%m%d').apply(lambda x: x.strftime('%Y-%m'))
    s2 = pd.to_datetime(pd.Series(df2['Date_M15']), format = '%Y%m%d').apply(lambda x: x.strftime('%Y-%m'))
    
    return s1.append(s2).nunique()

def N_months_concat4(df1, df2, df3, df4):
    
    s1 = pd.to_datetime(pd.Series(df1['Date_M15']), format = '%Y%m%d').apply(lambda x: x.strftime('%Y-%m'))
    s2 = pd.to_datetime(pd.Series(df2['Date_M15']), format = '%Y%m%d').apply(lambda x: x.strftime('%Y-%m'))
    s3 = pd.to_datetime(pd.Series(df3['Date_M15']), format = '%Y%m%d').apply(lambda x: x.strftime('%Y-%m'))
    s4 = pd.to_datetime(pd.Series(df4['Date_M15']), format = '%Y%m%d').apply(lambda x: x.strftime('%Y-%m'))
    
    return s1.append(s2).append(s3).append(s4).nunique()

''' ---------------------------------------------------------------------------------------------------------------------------'''

'''PARAMETERS USED:

Market_Structure(df,clean_sweep,x,perc_95)

    D1  --> clean_sweep = 1; x irrelevant; p_100 = 100000
    H4  --> clean_sweep = 1; x irrelevant; p_95  = 480
    H1  --> clean_sweep = 1; x irrelevant; p_95  = 220
    M15 --> clean_sweep = 1; x irrelevant; p_x   = 220

Price_M15_H4(df_M15_H1, k = 5000, ratio = 3, pip_min = 15, pip_max = 125, pip_over = -3)
Price_M15_H1(df_M15_H1, k = 1500, ratio = 3, pip_min = 15, pip_max = 90 , pip_over = -3)
'''

def MS_index_M15(df):
    
# Show data analitics based on individual Market structures (without taking into account repetitions)
    idx_list_break = []
    for i in range (1,len(df)-1):
    
        if (df.loc[i,'MS_Sit_M15'] == 'MS') & ((df.loc[i+1,'MS_Sit_M15'] == 'Up_Break') | (df.loc[i+1,'MS_Sit_M15'] == 'Dw_Break')):
            idx_list_break.append(i)
        else:
            continue
            
    return idx_list_break

def MS_index_H1(df):
    
# Show data analitics based on individual Market structures (without taking into account repetitions)
    idx_list_break = []
    for i in range (1,len(df)-1):
    
        if (df.loc[i,'MS_Sit_H1'] == 'MS') & ((df.loc[i+1,'MS_Sit_H1'] == 'Up_Break') | (df.loc[i+1,'MS_Sit_H1'] == 'Dw_Break')):
            idx_list_break.append(i)
        else:
            continue
            
    return idx_list_break

def MS_index_H4(df):
    
# Show data analitics based on individual Market structures (without taking into account repetitions)
    idx_list_break = []
    for i in range (1,len(df)-1):
    
        if (df.loc[i,'MS_Sit_H4'] == 'MS') & ((df.loc[i+1,'MS_Sit_H4'] == 'Up_Break') | (df.loc[i+1,'MS_Sit_H4'] == 'Dw_Break')):
            idx_list_break.append(i)
        else:
            continue
            
    return idx_list_break

def MS_index_D1(df):
    
# Show data analitics based on individual Market structures (without taking into account repetitions)
    idx_list_break = []
    for i in range (1,len(df)-1):
    
        if (df.loc[i,'MS_Sit_D1'] == 'MS') & ((df.loc[i+1,'MS_Sit_D1'] == 'Up_Break') | (df.loc[i+1,'MS_Sit_D1'] == 'Dw_Break')):
            idx_list_break.append(i)
        else:
            continue
            
    return idx_list_break

''' ---------------------------------------------------------------------------------------------------------------------------'''

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, n):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

''' ---------------------------------------------------------------------------------------------------------------------------'''

def roc(model, X_test, y_test):

    import numpy as np
    from sklearn.metrics import roc_curve, roc_auc_score
    from matplotlib import pyplot

    ## generate a no skill prediction (majority class)
    ns_probs = [0 for _ in range(len(y_test))]

    ## predict probabilities
    lr_probs = model.predict_proba(X_test)
    ## keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]
    ## calculate scores
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)
    ## summarize scores
    print('No Skill   : ROC AUC=%.3f' % (ns_auc))
    print('Rand Forest: ROC AUC=%.3f' % (lr_auc))
    ## calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
    ## plot the roc curve for the model
    pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Rand Forest')
    ## axis labels
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    ## show the legend
    pyplot.legend()
    ## show the plot
    pyplot.show()

''' ---------------------------------------------------------------------------------------------------------------------------'''

def split(df1,df2,df3,Label):
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

    df_frames = [df1, df2]
    frames = pd.concat(df_frames)

    X = frames.drop([Label],axis=1)[:]
    y = frames[Label][:]

    X2 = df3.drop([Label],axis=1)
    y2 = df3[Label]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.99, shuffle = True, random_state = 101)
    
    return X_train, X_test, y_train, y_test, X2, y2

''' ---------------------------------------------------------------------------------------------------------------------------'''

def model_rfc_random(X_train,y_train):
    
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
    from sklearn.datasets import make_classification

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 4)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num = 4)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 10]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]
    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    print(random_grid)

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rfc = RandomForestClassifier()
    # Random search of parameters, using 3 fold cross validation, 
    # search across 100 different combinations, and use all available cores
    rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
    
    # Fit the random search model
    rfc_random.fit(X_train,y_train)
    
    print('Best Parameters:', rfc_random.best_params_)
    
    return rfc_random.best_estimator_

''' ---------------------------------------------------------------------------------------------------------------------------'''

def predictions(model, X_test, y_test, thres):

    from sklearn.metrics import classification_report,confusion_matrix

    print('Predictions - Threshold >=',thres)
    predicted_proba = model.predict_proba(X_test)
    predictions = (predicted_proba [:,1] >= thres).astype('int')
    print(classification_report(y_test,predictions))
    print(confusion_matrix(y_test,predictions))

''' ---------------------------------------------------------------------------------------------------------------------------'''

def top_cols(model, X_train, threshold):

    feature_names = [f'{col}' for col in X_train.columns]
    importances = model.feature_importances_
    forest_importances = pd.Series(importances, index=feature_names)
    
    return list(forest_importances[forest_importances > threshold].sort_values(ascending = False).index[:])

''' ---------------------------------------------------------------------------------------------------------------------------'''

def remove_corr(df, ind_cols, th):

    corr_list_top   = get_top_abs_correlations(df[ind_cols], 150)
    corr_list_top_f = corr_list_top[corr_list_top > th].index[:]
    
    corr_list =[]
    
    for i in range(0,len(corr_list_top_f)):
        corr_list.append(corr_list_top_f[i][1])
    
    corr_list = list(set(corr_list))
    ret_list  = [x for x in ind_cols if x in corr_list]
    
    return ret_list

''' ---------------------------------------------------------------------------------------------------------------------------'''

def Pre_rec_curve(model, X_test, y_test, thres):
    
    from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, average_precision_score
    from sklearn.metrics import plot_precision_recall_curve
    import matplotlib.pyplot as plt

    predicted_proba = model.predict_proba(X_test)
    predicted = (predicted_proba [:,1] >= thres).astype('int')

    accuracy  = accuracy_score(y_test, predicted)
    precision = precision_score(y_test, predicted)

    disp = plot_precision_recall_curve(model, X_test, y_test)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(precision))

''' ---------------------------------------------------------------------------------------------------------------------------'''

def opt_thres(model, X_test1, y_test1, X_test2, y_test2, X_test3, y_test3, pr):
    
    #from sklearn.metrics import accuracy_score, precision_score, average_precision_score
    
    for i in np.arange(0,1,0.02):
        
        predicted_proba1 = model.predict_proba(X_test1)
        predicted1 = (predicted_proba1 [:,1] >= i).astype('int')
        if precision_score(y_test1, predicted1) > pr:
            break
            
    print(i, precision_score(y_test1, predicted1))

    for j in np.arange(0,1,0.02):
        
        predicted_proba2 = model.predict_proba(X_test2)
        predicted2 = (predicted_proba2 [:,1] >= j).astype('int')
        if precision_score(y_test2, predicted2) > pr:
            break
            
    print(j, precision_score(y_test2, predicted2))

    for k in np.arange(0,1,0.02):
        
        predicted_proba3 = model.predict_proba(X_test3)
        predicted3 = (predicted_proba3 [:,1] >= k).astype('int')
        if precision_score(y_test3, predicted3) > pr:
            break
            
    print(k, precision_score(y_test3, predicted3))
    
    th = max(i,j,k)
    print ('Threshold:',t) 
            
    return th

''' ---------------------------------------------------------------------------------------------------------------------------'''