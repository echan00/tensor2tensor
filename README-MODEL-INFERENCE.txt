
# TO EXPORT MODEL
#t2t-exporter \
#  --model=transformer \
#  --hparams_set=transformer_base \
#  --decode_hparams='beam_size=4,alpha=0.6' \
#  --problem=translate_enzh_wmt32k_rev \
#  --data_dir=/home/ubuntu//T2T_Model/t2t_data \
#  --output_dir=/home/ubuntu/T2T_Model/t2t_train/translate_enzh_wmt32k/transformer-transformer_base

t2t-exporter \
  --model=transformer \
  --hparams_set=transformer_base \
  --decode_hparams='beam_size=4,alpha=0.6' \
  --problem=translate_enzh_wmt32k_rev \
  --data_dir=/root/T2T_Model/t2t_data \
  --output_dir=/root/T2T_Model/t2t_train/translate_enzh_wmt32k/transformer-transformer_base


# TO SERVE MODEL
tensorflow_model_server \
  --port=9000 \
  --model_name=transformer \
  --model_base_path=/root/T2T_Model/t2t_train/translate_enzh_wmt32k/transformer-transformer_base/export \
  --enable_batching --batching_parameters_file=batching.conf
#  --rest_api_port=9001

docker run -p 9001:9001 \
  --mount type=bind,source=/root/T2T_Model/t2t_train/translate_enzh_wmt32k/transformer-transformer_base/export/,target=/models/my_model \
  --mount type=bind,source=/root/T2T_Model/batching.conf,target=/models/batching.conf \
  -e MODEL_NAME=my_model -t tensorflow-serving --batching_parameters_file=/models/batching.conf --enable_batching


# BUILD YOUR OWN TF SERVING
# https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/docker.md#serving-with-docker-using-your-gpu

docker build -t test/tensorflow-serving-devel -f Dockerfile.devel https://github.com/tensorflow/serving.git#:tensorflow_serving/tools/docker

docker build -t test/tensorflow-serving --build-arg TF_SERVING_BUILD_IMAGE=test/tensorflow-serving-devel https://github.com/tensorflow/serving.git#:tensorflow_serving/tools/docker

 
# RUN TF SERVERING
docker run -p 8500:8500 \
  --mount type=bind,source=/root/T2T_Model/t2t_train/translate_enzh_wmt32k/transformer-transformer_base/export/,target=/models/my_model \
  --mount type=bind,source=/root/T2T_Model/batching.conf,target=/models/batching.conf \
  -e MODEL_NAME=my_model -t test/tensorflow-serving --batching_parameters_file=/models/batching.conf --enable_batching

# RUN TF SERVERING (edmunds server)
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 -p 8500:8500 \
  --mount type=bind,source=/home/eee/T2T_Model/t2t_train/translate_enzh_wmt32k/transformer-transformer_base/export/,target=/models/my_model \
  --mount type=bind,source=/home/eee/T2T_Model/batching.conf,target=/models/batching.conf \
  -e MODEL_NAME=my_model -t tensorflow/serving:latest-gpu --batching_parameters_file=/models/batching.conf --enable_batching

# 8510 is the port to connect to and 8500 is the port inside docker container
docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 -p 8510:8500 \
  --mount type=bind,source=/home/eee/T2T_Model/t2t_train/translate_enzh_wmt32k/transformer-transformer_base/export/,target=/models/my_model \
  --mount type=bind,source=/home/eee/T2T_Model/batching.conf,target=/models/batching.conf \
  -e MODEL_NAME=my_model -t tensorflow/serving:latest-gpu --batching_parameters_file=/models/batching.conf --enable_batching

## batching.conf parameters
max_batch_size { value: 1000 }
batch_timeout_micros { value: 0 }
num_batch_threads { value: 32 }
max_enqueued_batches { value: 1000 }



# TO QUERY MODEL
t2t-query-server \
  --server=0.0.0.0:8510 \
  --servable_name=my_model \
  --problem=translate_enzh_wmt32k \
  --data_dir=/home/eee/T2T_Model/t2t_data \
  --timeout_secs=60 \
  --TFX=1 \
  --subdir='dir_002' \
  --inputs_once=你好 \
  --bleualign_upload=0


