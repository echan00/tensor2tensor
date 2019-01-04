

ansible-playbook --private-key '/Users/erikchan/Downloads/digitalocean' 1_playbook.yml


ssh -i ./digitalocean root@104.248.77.64

cd T2T_Model/

tmux new -s session1

docker run -p 8500:8500 \
  --mount type=bind,source=/root/T2T_Model/t2t_train/translate_enzh_wmt32k/transformer-transformer_base/export/,target=/models/my_model \
  --mount type=bind,source=/root/T2T_Model/batching.conf,target=/models/batching.conf \
  -e MODEL_NAME=my_model -t test/tensorflow-serving --batching_parameters_file=/models/batching.conf --enable_batching

tmux new -s session2

t2t-query-server \
  --server=0.0.0.0:8500 \
  --servable_name=my_model \
  --problem=translate_enzh_wmt32k \
  --data_dir=/root/T2T_Model/t2t_data \
  --timeout_secs=30 \
  --TFX=1 \
  --subdir='dir_001' \
  --inputs_once=你好 \
  --bleualign_upload=1



