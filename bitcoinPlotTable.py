import pandas as pd
import time
import datetime

def getBitcoinMarketInfo():
    # コインマーケットキャップからデータをスクレイピング
    bitcoin_market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
    # Dateを文字列から日付フォーマットへ変換
    bitcoin_market_info = bitcoin_market_info.assign(Date=pd.to_datetime(bitcoin_market_info['Date']))
    # 取引高が'-'の欠損データを0へ変換
    bitcoin_market_info.loc[bitcoin_market_info['Volume']=='-', 'Volume'] = 0
    # intへ変換
    bitcoin_market_info['Volume'] = bitcoin_market_info['Volume'].astype('int64')
    print("**********ビットコインの情報**************")
    print(bitcoin_market_info.head())
    return bitcoin_market_info

# # データセットのヘッド情報の確認
# print("\n*****************データセットのヘッダ情報****************\n")
# print(getBitcoinMarketInfo().head())