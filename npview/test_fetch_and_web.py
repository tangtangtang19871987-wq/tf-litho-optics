
import subprocess
import os
import time

# 虚拟环境python路径
VENV_PYTHON = os.path.abspath(os.path.join(os.path.dirname(__file__), '../.venv/bin/python'))

# 测试数据抓取脚本
print('测试数据抓取脚本...')
ret = subprocess.run([VENV_PYTHON, 'fetch_top10_stocks.py'], cwd=os.path.dirname(__file__))
assert ret.returncode == 0, '数据抓取脚本运行失败'
assert os.path.exists(os.path.join(os.path.dirname(__file__), 'top10_stocks.csv')), 'CSV文件未生成'
print('数据抓取脚本测试通过')

# 测试Web服务（仅测试能否启动）
print('测试Web服务脚本...')
proc = subprocess.Popen([VENV_PYTHON, 'web_server.py'], cwd=os.path.dirname(__file__))
time.sleep(2)
# 检查端口是否监听
import socket
s = socket.socket()
try:
    s.connect(('localhost', 8000))
    print('Web服务端口监听正常')
finally:
    s.close()
    proc.terminate()
print('Web服务测试通过')
