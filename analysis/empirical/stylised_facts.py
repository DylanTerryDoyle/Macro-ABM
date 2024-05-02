import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
# add parent directory
sys.path.append('..\\..')
from src.utils import *

### matplotlib settings ###

# change matplotlib font to serif
plt.rcParams['font.family'] = ['serif']
# figure size
x_figsize = 8
y_figsize = 5
# fontsize
fontsize = 18

### paths to folders & data ###

# show current databases in data folder
cwd = os.getcwd()
# analysis folder path
analysis_path = os.path.abspath(os.path.join(cwd, os.pardir))
# parent path of analysis
parent_path = os.path.abspath(os.path.join(analysis_path, os.pardir))
# figure path
figure_path = f"{cwd}\\figures"

### read USA data ###

# real gdp data 
real_gdp = pd.read_csv('data/GDPC1.csv')
# consumption data
consumption = pd.read_csv('data/PCECC96.csv')
# investment data
investment = pd.read_csv('data/GPDIC1.csv')
# employment data
employment = pd.read_csv('data/CE16OV.csv')
# unemployment rate data
unrate = pd.read_csv('data/UNRATE.csv')
# corporate debt data
debt = pd.read_csv('data/TBSDODNS.csv')

### create dataframe to store data ###

macro = pd.DataFrame()
# real gross domestic product
macro['rgdp'] = real_gdp['GDPC1']
# real consumption
macro['consumption'] = consumption['PCECC96']
# real investment
macro['investment'] = investment['GPDIC1']
# total level of employment
macro['employment'] = employment['CE16OV']
# unemployment rate
macro['unrate'] = unrate['UNRATE']/100
# total corporate debt
macro['debt'] = debt['TBSDODNS'] 
# macroframe index
macro.index = real_gdp['DATE']
# linearly fill missing values
macro = macro.interpolate('linear')

### define new series ###

# normalised productivity 
macro['productivity'] = (macro['rgdp']/macro['employment'])/(macro['rgdp'].iloc[0]/macro['employment'].iloc[0])

### plot autocorrelation of time series ###

# autocorrelation lags
autocorr_lags = 20
# Hodrick-Prescott filter sensitivity parameters (quarterly)
hp_lamda = 1600

# real gdp autocorrelation
plot_autocorrelation(macro['rgdp'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\autocorr_rgdp.png', lags=autocorr_lags, lamda=hp_lamda)
# consumption autocorrelation
plot_autocorrelation(macro['consumption'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\autocorr_consumption.png', lags=autocorr_lags, lamda=hp_lamda)
# investment autocorrelation 
plot_autocorrelation(macro['investment'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\autocorr_investment.png', lags=autocorr_lags, lamda=hp_lamda)
# productivity autocorrelation
plot_autocorrelation(macro['productivity'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\autocorr_productivity.png', lags=autocorr_lags, lamda=hp_lamda)
# unemployment rate autocorrelation
plot_autocorrelation(macro['unrate'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\autocorr_unemployment.png', lags=autocorr_lags, lamda=hp_lamda)
# debt autocorrelation
plot_autocorrelation(macro['debt'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\autocorr_debt.png', lags=autocorr_lags, lamda=hp_lamda)

### plot cross correlation ###

# cross correlation total lags 
xcorr_lags = 10

# cross correlation of real gdp with real gdp
plot_cross_correlation(macro['rgdp'], macro['rgdp'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\xcorr_rdgp.png', lags=xcorr_lags, lamda=hp_lamda)
# cross correlation of real gdp with consumption
plot_cross_correlation(macro['rgdp'], macro['consumption'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\xcorr_consumption.png', lags=xcorr_lags, lamda=hp_lamda)
# cross correlation of real gdp with investment
plot_cross_correlation(macro['rgdp'], macro['investment'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\xcorr_investment.png', lags=xcorr_lags, lamda=hp_lamda)
# cross correlation of real gdp with debt
plot_cross_correlation(macro['rgdp'], macro['debt'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\xcorr_debt.png', lags=xcorr_lags, lamda=hp_lamda)
# cross correlation of real gdp with unemployment rate
plot_cross_correlation(macro['rgdp'], macro['unrate'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\xcorr_unemployment.png', lags=xcorr_lags, lamda=hp_lamda)
# cross correlation of debt with unemployment_rate
plot_cross_correlation(macro['debt'], macro['unrate'], figsize=(x_figsize,y_figsize), fontsize=fontsize, savefig=f'{figure_path}\\xcorr_debt_unemployment.png', lags=xcorr_lags, lamda=hp_lamda)