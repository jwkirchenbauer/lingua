setup 


```
python setup/download_prepare_hf_data.py fineweb_edu_10bt 64 --data_dir ./data --seed 42 --nchunks 8

python setup/download_tokenizer.py llama3 ./tokenizers --api_key $HF_HUB_TOKEN

srun -N1 -n2 --ntasks-per-node=2 --mem=128G --cpus-per-task=2 --unbuffered \
python -m apps.main.train \
    config=apps/main/configs/llama_1B_fw_10bt_nexus.yaml


srun -N1 -n1 --ntasks-per-node=1 --gpus-per-task=1 --mem=128G --cpus-per-task=2 --unbuffered \
python -m apps.main.train \
    config=apps/main/configs/llama_7B_fw_10bt_nexus.yaml

srun -N1 -n2 --ntasks-per-node=2 --mem=128G --cpus-per-task=2 --unbuffered \
python -m apps.main.train \
    config=apps/main/configs/llama_7B_fw_10bt_nexus.yaml

srun -N1 -n4 --ntasks-per-node=4 --mem=128G --cpus-per-task=2 --unbuffered \
python -m apps.main.train \
    config=apps/main/configs/llama_7B_fw_10bt_nexus.yaml \
    data.batch_size=2 \
    data.seq_len=4096 \

9.5k tps/gpu reported

checks out via the reported other numbers
8192 toks / 0.87 s/step = 9.4k tps/gpu

now, with gqa set as 8 kv heads which is closer to the litgpt Llama-3-8B ...
we get a faster 11.1k tps/gpu at 86% mem
```

note that some cache related env vars are expected to be managed as you run out of space fast
```
# torch compile
export TRITON_CACHE_DIR="/cmlscratch/jkirchen/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="/cmlscratch/jkirchen/.cache/inductor"

# general temp
export TMPDIR="/cmlscratch/jkirchen/.cache/tmp"
export TEMP=$TMPDIR
export TMP=$TMPDIR
```