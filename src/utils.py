import yaml
import numpy as np
import powerlaw as pl
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import statsmodels.api as sm
from statsmodels.graphics import tsaplots

def load_parameters(filename: str):
    """
    Load YAML file as dictionary.
    
    Parameters
    ----------
        filename : str
            name of YAML file to load
    
    Returns
    -------
        file_dict : dict
            YAML file loaded as dictionary 
    """
    with open(filename, 'r') as file:
        file_dict = yaml.safe_load(file)
    return file_dict

def logscale_ticks(low: float, high: float, num: int) -> np.ndarray:
    """
    Get ticks for either x or y axis from data, rounded to first digit spaced by num in log 10.
    
    Parameters
    ----------
        series : pd.Series
            time series
            
        num : int 
            number of ticks to return
    
    Returns
    -------
        ticks : numpy array
            axis ticks spaced by num in log 10
    """
    log_arr = np.logspace(np.log10(low),np.log10(high), num)
    lengths = np.vectorize(len)(np.char.mod('%d', log_arr))
    factor = 10 ** (lengths - 1)
    round_log_arr = np.int64(np.round(log_arr.astype(int) / factor) * factor)
    return round_log_arr

def plot_autocorrelation(data, figsize: tuple, fontsize: int, savefig: str, lags: int, lamda: int=1600) -> None:
    """
    Plots the autocorrelation function for a given series for a given number of lags.
    
    Parameters
    ----------
        data : series like
            time series
        
        figsize : tuple (int, int) 
            size of figure (x-axis, y-axis)
            
        fontsize : int
            fontsize of legend and ticks
            
        savefig : str
            path and figure name 
        
        lags: int
            number of autocorrelation lags
        
        lamda : int (default = 1600)
            HP lambda parameter (quarterly => lambda=1600)
    """
    # Hodrick-Prescot filter
    cycle, trend = sm.tsa.filters.hpfilter(data, lamda)
    # figure
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(111)
    # autocorrelation plot
    tsaplots.plot_acf(cycle, lags=lags, ax=ax1, color='k', vlines_kwargs={"colors": 'k'})
    # change shaded colour
    for item in ax1.collections:
        if type(item)==PolyCollection:
            item.set_facecolor('k')
    # ticks
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # remove title
    plt.title("")
    # save figure
    plt.savefig(savefig, bbox_inches='tight')
    plt.show()

def plot_cross_correlation(x, y, figsize: tuple, fontsize: int, savefig: str, lags: int, lamda: int=1600):
    """
    Plots the cross correlation between series x and series y, from lags to -lags lags.
    
    Parameters
    ----------
        x : series like
            time series
            
        y : series like
            time series
        
        figsize : tuple (int, int) 
            size of figure (x-axis, y-axis)
            
        fontsize : int
            fontsize of legend and ticks
            
        savefig : str
            path and figure name 
        
        lags: int
            number of correlation lags
        
        lamda : int (default = 1600)
            HP lambda parameter (quarterly => lambda=1600)
    """
    # Hodrick-Prescot filter
    xcycle, xtrend = sm.tsa.filters.hpfilter(x, lamda)
    ycycle, ytrend = sm.tsa.filters.hpfilter(y, lamda)
    # figure
    plt.figure(figsize=figsize)
    corr = plt.xcorr(xcycle, ycycle, color='k', maxlags=lags)
    plt.scatter(corr[0], corr[1], color='k')
    plt.axhline(0, color="k")
    # ticks
    plt.xticks([-lags, -lags/2, 0, lags/2, lags], fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # save figure
    plt.savefig(savefig, bbox_inches='tight')
    plt.show()

def plot_ccdf(data, figsize: tuple, fontsize: int, savefig: str, dp: int=0) -> None:
    """
    Plots the complementary cumulative distribution function (CCDF) for a given series 
    and prints the power law exponent, cut-off value, and compares the distribution to a lognormal.
    
    Parameters
    ----------
        data : pd.Series or numpy array
            series to plot CCDF
            
        figsize : tuple (int, int) 
            size of figure (x-axis, y-axis)
            
        fontsize : int
            fontsize of legend and ticks
            
        savefig : str
            path and figure name 
        dp : int 
            power law fit x minimum number decimal points
    """
    # power law fit results 
    results = pl.Fit(data)
    a, m = results.alpha, results.xmin
    # complementary cdf 
    plt.figure(figsize=figsize)
    # x values 
    x = np.sort(data)
    # complementary cdf (ccdf)
    cdf = np.arange(1,len(data)+1)/(len(data))
    ccdf = 1 - cdf
    plt.scatter(x, ccdf, color='skyblue', edgecolors='k', alpha=0.7, s=40, label='CCDF')
    # power law fit
    # => rescale to start fit from cut off
    index = np.where(x == m)[0][0]
    rescale = ccdf[index]
    power_law_fit = np.where(x >= m, np.power((m)/x,a-1)*rescale, np.nan)
    plt.plot(x, power_law_fit, color='limegreen', linewidth=3, label='Power-Law Fit')
    # power law cut off (mF)
    plt.axvline(m, color='k', linestyle='--', label=r'$m$'f' = {round(m, dp)}') # type: ignore
    # lognormal distribution
    estimates = stats.lognorm.fit(data)
    cdf = stats.lognorm.cdf(x, estimates[0], estimates[1], estimates[2])
    plt.plot(x, 1 - cdf, color='r', linestyle='-.', linewidth=2, label='Log-Normal Fit')
    # log-log axis
    plt.loglog()
    # legend
    plt.legend(loc='lower left', fontsize=fontsize)
    # tick size
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    # save figure
    plt.savefig(savefig, bbox_inches='tight')
    plt.show()
    print(f'Power law exponent = {a}')
    print(f'Power law minimum = {m}')
    print(f"Distribution compare = {results.distribution_compare('power_law', 'lognormal')}\n")