#!/bin/bash

# 检查名为monitor的进程数量，排除当前的grep命令
if [[ $(ps aux | grep 'monitor_DFA_textcnn.py' | grep -v grep | wc -l) -ne 1 ]]; then
    # 如果monitor.py进程不是1个，则使用conda run命令在tf2gpu环境中运行monitor.py
    conda run -n tf2gpu nohup python /root/DFA_Textcnn_Sensitive/monitor_DFA_textcnn.py &>/dev/null &
fi

# 输出111到/root/aaa.txt文件
echo 111 >> /root/DFA.txt
