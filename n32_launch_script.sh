
ml rocm/6.4.0
plugin_loc=/collab/usr/global/tools/rccl/$SYS_TYPE/rocm-6.3.1/install/lib
LD_LIBRARY_PATH=${plugin_loc}:$LD_LIBRARY_PATH  # point to directory with plugin library
#aws ofi plugin set up in my env already
#use one of the 2 following
#ml rccl/working-env
ml rccl/fast-env-slows-mpi
#fix hangs, per nikoli
FI_MR_CACHE_MONITOR=userfaultfd
# optimizations, https://github.com/LBANN/HPC-launcher/blob/main/hpc_launcher/systems/lc/el_capitan_family.py#L133
FI_CXI_RDZV_PROTO=alt_read
FI_CXI_RDZV_THRESHOLD=0
FI_CXI_RDZV_GET_MIN=0
FI_CXI_RDZV_EAGER_SIZE=0

head_node_ip=$(hostname --ip-address)
echo Head Node IP: $head_node_ip
flux run -N 32 bash -c "echo \$(hostname)"

torchrun-hpc \
        -N 32 \
        -n 4 \
        --gpus-per-proc 1 \
        apps/main/train.py \
            config=apps/main/configs/comma_7B_brb.yaml \
            data.batch_size=4
# :'
# flux run -N 32 -o fastload=on \
#     torchrun \
#         --nnodes 32 \
#         --nproc_per_node 4 \
#         --rdzv_id $RANDOM \
#         --rdzv_backend c10d \
#         --rdzv_endpoint $head_node_ip:29501 \
#         -m apps.main.train \
#             config=apps/main/configs/comma_7B_brb_32N.yaml
# '
echo Done
