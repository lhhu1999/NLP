from test_ner import run_ner
from test_tsc import run_tsc

while True:
    print(">>>>>> 请输入内容：", end='')
    sss = input()
    print("############################### 情感分析 #################################")
    run_tsc(sss)
    print("############################### 实体抽取 #################################")
    run_ner(sss)
    print("#########################################################################")