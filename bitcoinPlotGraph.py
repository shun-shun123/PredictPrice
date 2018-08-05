# ビットコインとイーサリアムのロゴを取得しましょう
import sys
from PIL import Image
import io
import urllib
import matplotlib.pyplot as plt
import datetime
import bitcoinPlotTable

bt_img = urllib.request.urlopen("http://logok.org/wp-content/uploads/2016/10/Bitcoin-Logo-640x480.png")
image_file = io.BytesIO(bt_img.read())
bitcoin_im = Image.open(image_file)

bitcoin_market_info = bitcoinPlotTable.getBitcoinMarketInfo()

# ビットコインとイーサリアムのデータフレームのカラム名を変更
bitcoin_market_info.columns = [bitcoin_market_info.columns[0]] + ['bt_' + i for i in bitcoin_market_info.columns[1:]]
print(bitcoin_market_info.columns)
# ビットコインの価格をプロッティング
fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
ax1.set_ylabel('Closing Price（$）', fontsize=12)
ax2.set_ylabel('Volume（$ bn）', fontsize=12)
ax2.set_yticks([int('%d000000000'%i) for i in range(10)])
ax2.set_yticklabels(range(10))
ax1.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax1.set_xticklabels('')
ax2.set_xticks([datetime.date(i,j,1) for i in range(2013,2019) for j in [1,7]])
ax2.set_xticklabels([datetime.date(i,j,1).strftime('%b %Y')  for i in range(2013,2019) for j in [1,7]])
ax1.plot(bitcoin_market_info['Date'].astype(datetime.datetime),bitcoin_market_info['bt_Open*'])
ax2.bar(bitcoin_market_info['Date'].astype(datetime.datetime).values, bitcoin_market_info['bt_Volume'].values)
fig.tight_layout()
fig.figimage(bitcoin_im, 100, 120, zorder=3,alpha=.5)
plt.show()