import http.server
import socketserver
import pandas as pd
import os

PORT = 8090
CSV_FILE = os.path.join(os.path.dirname(__file__), 'top10_stocks.csv')

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>今日A股涨幅前十</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        table {{ border-collapse: collapse; width: 60%; margin: auto; }}
        th, td {{ border: 1px solid #ccc; padding: 8px 12px; text-align: center; }}
        th {{ background: #f2f2f2; }}
    </style>
</head>
<body>
    <h2 style="text-align:center">今日A股涨幅前十</h2>
    {table}
</body>
</html>
'''

def get_table_html():
    if not os.path.exists(CSV_FILE):
        return '<p>暂无数据，请先运行数据抓取脚本。</p>'
    df = pd.read_csv(CSV_FILE)
    return df.to_html(index=False, border=0, justify='center')

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            table_html = get_table_html()
            html = HTML_TEMPLATE.format(table=table_html)
            self.send_response(200)
            self.send_header('Content-type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(html.encode('utf-8'))
        else:
            super().do_GET()

if __name__ == '__main__':
    with socketserver.TCPServer(('', PORT), Handler) as httpd:
        print(f'本地网页已启动: http://localhost:{PORT}')
        httpd.serve_forever()
