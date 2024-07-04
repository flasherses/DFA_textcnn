#!/bin/bash
if [[ $(ps aux |grep monitor |grep -v grep |wc -l) != 1 ]];then /data/Miniconda3/bin/python -u /root/duxinbao/v3/monitor.py &>/dev/null & fi
echo 111 >> /root/DFA1.txt
