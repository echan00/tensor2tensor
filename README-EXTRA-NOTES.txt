
# split folder of files into multiple folders
i=0; 
for f in *; 
do 
    d=dir_$(printf %03d $((i/700+1))); 
    mkdir -p $d; 
    mv "$f" $d; 
    let i++; 
done



# AWS S3 SYNC EXAMPLE
aws s3 sync /root/T2T_Model/4b_zh-tokenized-sample-en s3://nda-ai/final-dec-14-2018/4b_zh-tokenized-sample-en



# delete 0 byte files in unix
#find . -type f -size 0b -print
find /root/T2T_Model/4b_zh-tokenized-sample-en -type f -size 0b -delete



# keep translation server alive  - https://www.digitalocean.com/community/tutorials/how-to-install-and-configure-monit
1) sudo apt-get install monit
2) run 'monit'
3) add to /etc/monit/monitrc
check process tensorflow_mode
        matching "tensorflow_mode"
        start program = "/script/start.sh"
        stop program = "/usr/bin/killall t2t-query-serve"

4) add to /script/start.sh
#!/bin/bash
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=6 -p 8560:8500 \
  --mount type=bind,source=/home/eee/T2T_Model/t2t_train/translate_enzh_wmt32k/transformer-transformer_base/export/,target=/models/my_model \
  --mount type=bind,source=/home/eee/T2T_Model/batching.conf,target=/models/batching.conf \
  -e MODEL_NAME=my_model -t tensorflow/serving:latest-gpu --batching_parameters_file=/models/batching.conf --enable_batching

5) 'monit reload'