#!/usr/bin/env bash
python gen_result_file.py --name $1 --list /home/liuyudong/icdar-notb/data/icdar/test/img_list.txt --type icdar --poly --iter $3 --dev $2
python cvt_format.py $1_poly
/usr/bin/python /home/liuyudong/sren.pytorch/tools/script.py -g=gt.zip -s=submit/$1_poly.zip
