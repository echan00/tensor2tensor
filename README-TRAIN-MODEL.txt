
BUILDING TENSOR2TENSOR CHINESE->ENGLISH TRANSLATION MODEL 

pip install tensor2tensor

# See what problems, models, and hyperparameter sets are available.
# You can easily swap between them (and add new ones).
t2t-trainer --registry_help

PROBLEM=translate_enzh_wmt32k
MODEL=transformer
HPARAMS=transformer_base #transformer_big

DATA_DIR=$HOME/t2t_data2
TMP_DIR=/tmp/t2t_datagen
TRAIN_DIR=$HOME/t2t_train2/$PROBLEM/$MODEL-$HPARAMS

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR

# Generate data
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
PROBLEM=translate_enzh_wmt32k_rev # for chinese to english

# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR
  --worker_gpu=8


  hparams.max_length = 256
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate_schedule = ("constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
  hparams.learning_rate_constant = 2.0
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  hparams.batch_size = 2048
  hparams.num_heads = 16
  hparams.optimizer_adam_beta2 = 0.997


# Decode
source test/bin/activate

DATA_DIR=$HOME/T2T_Model/t2t_data
PROBLEM=translate_enzh_wmt32k_rev
MODEL=transformer
HPARAMS=transformer_base
TRAIN_DIR=$HOME/T2T_Model/t2t_train/translate_enzh_wmt32k/$MODEL-$HPARAMS

t2t-decoder \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --decode_hparams="beam_size=4,alpha=0.6" \
  --decode_from_file=4a_zh-tokenized-converted/LTN20120402854.txt \
  --decode_to_file=4b_zh-tokenized-sample-en/LTN20120402854.txt

# Evaluate the BLEU score
# Note: Report this BLEU score in papers, not the internal approx_bleu metric.
# t2t-bleu --translation=4a_zh-tokenized-converted/LTN20120402854.txt --reference=3_en-tokenized/LTN20120402854.txt



t2t-query-server \
  --server=0.0.0.0:9000 \
  --servable_name=transformer \
  --problem=translate_enzh_wmt32k_rev \
  --data_dir=/root/T2T_Model/t2t_data \
  --inputs_once='你好'

t2t-decoder \
  --data_dir=/root/T2T_Model/t2t_data \
  --problem=translate_enzh_wmt32k_rev \
  --model=transformer \
  --hparams_set=transformer_base \
  --output_dir=/root/T2T_Model/t2t_train/translate_enzh_wmt32k/transformer-transformer_base \
  --decode_hparams='beam_size=4,alpha=0.6' \
  --decode_interactive

  --decode_from_file=/root/T2T_Model/4a_zh-tokenized-converted/1000.txt \
  --decode_to_file=/root/T2T_Model/4b_zh-tokenized-sample-en/1000.txt 

  
  --worker_gpu=0 --locally_shard_to_cpu





HOW TO USE TMUX
1) tmux new -s session1
2) Ctrl + b and then d -> to leave
3) tmux a -t session1
4) tmux kill-session -t session1



