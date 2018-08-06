import bitcoinPlotTable, ethPlotTable
import pandas as pd

bitcoin_market_info = bitcoinPlotTable.getBitcoinMarketInfo()
eth_market_info = ethPlotTable.getEthMarketInfo()

def createDataSet():
    market_info = pd.merge(bitcoin_market_info, eth_market_info, on=['Date'])
    market_info = market_info[market_info['Date'] >= '2016-01-01'] 
    for coins in ['_x', '_y']:
        kwargs = { 'day_diff' + coins: lambda x: (x['Close**' + coins] - x['Open*' + coins]) / x['Open*' + coins]}
        market_info = market_info.assign(**kwargs)
    print("**********ビットコイン&イーサリアムの情報**************")
    print(market_info.head())
    return market_info

# print("**********データセット作成完了*********")
# createDataSet()