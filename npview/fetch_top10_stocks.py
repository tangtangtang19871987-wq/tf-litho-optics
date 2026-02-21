
import requests
import pandas as pd
from datetime import datetime

# 东方财富A股涨幅榜API
CSV_FILE = 'top10_stocks.csv'

def fetch_top10_stocks():
    # 东方财富A股涨幅榜API接口，排序字段为涨跌幅（f3），取前10
    url = (
        'https://push2.eastmoney.com/api/qt/clist/get'
        '?pn=1&pz=10&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:13,m:1+t:2,m:1+t:23'
        '&fields=f12,f14,f3'
    )
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.encoding = 'utf-8'
    js = resp.json()
    if not js or 'data' not in js or 'diff' not in js['data']:
        raise Exception('未获取到榜单数据')
    data = []
    for item in js['data']['diff']:
        code = item.get('f12', '')
        name = item.get('f14', '')
        percent = item.get('f3', '')
        data.append({'代码': code, '名称': name, '涨幅%': percent})
    return data



def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df['日期'] = datetime.now().strftime('%Y-%m-%d')
    df.to_csv(filename, index=False, encoding='utf-8-sig')



def main():
    data = fetch_top10_stocks()
    save_to_csv(data, CSV_FILE)
    print(f'已保存至 {CSV_FILE}')

if __name__ == '__main__':
    main()
