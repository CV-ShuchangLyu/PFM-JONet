#!/bin/bash

T=`date +%m%d%H%M`
ROOT=../
export PYTHONPATH=$ROOT:$PYTHONPATH

#python $ROOT/demo/image_demo.py ./area8_256_768_768_1280.png SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet.py iter_11000_P2V_66.86.pth --device cuda:1 --out-file ./result.png
python $ROOT/demo/image_demo.py ./chicago78_1024_2048_1536_2560.png SAM_UDA_Sb5PromptSTAdv_bit-b16_upernet_P2C.py iter_35000_P2C_56.96.pth --device cuda:1 --out-file ./result.png