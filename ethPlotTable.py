import pandas as pd
import time
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np

# イーサリアムの価格データを所定のURLからスクレイピング
eth_market_info = pd.read_html("https://coinmarketcap.com/currencies/ethereum/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
# データの処理とヘッド情報の表示
eth_market_info = eth_market_info.assign(Date=pd.to_datetime(eth_market_info['Date']))
print("\n*****************データセットのヘッダ情報****************\n")
print(eth_market_info.head())