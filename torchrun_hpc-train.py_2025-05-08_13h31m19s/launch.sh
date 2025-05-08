#!/bin/sh
export NCCL_NET_GDR_LEVEL=3
export NCCL_MIN_NCHANNELS=24
export MIOPEN_DEBUG_DISABLE_FIND_DB=0
export MIOPEN_DISABLE_CACHE=0
export MIOPEN_USER_DB_PATH=/var/tmp/bartolds/MIOpen_user_db
export MIOPEN_CUSTOM_CACHE_DIR=/var/tmp/bartolds/MIOpen_custom_cache
export LD_LIBRARY_PATH=/opt/cray/pe/mpich/8.1.32/ofi/crayclang/18.0/lib:/opt/cray/pe/mpich/8.1.32/gtl/lib:/opt/cray/pe/libsci/25.03.0/CRAYCLANG/18.0/x86_64/lib:/opt/cray/pe/cce/18.0.1/cce-clang/x86_64/lib:/opt/cray/pe/cce/18.0.1/cce/x86_64/lib:/opt/cray/pe/perftools/25.03.0/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=/opt/rocm-6.4.0/llvm/lib:${LD_LIBRARY_PATH}

# Performance tuning for RCCL + HPE Slingshot Cassini NIC (Audited on 3/31/25)
export FI_CXI_RDZV_PROTO=alt_read		# Performance tuning for RCCL + HPE Slingshot Cassini NIC (Audited on 3/31/25)
export FI_CXI_RDZV_THRESHOLD=0		# Performance tuning for RCCL + HPE Slingshot Cassini NIC (Audited on 3/31/25)
export FI_CXI_RDZV_GET_MIN=0		# Performance tuning for RCCL + HPE Slingshot Cassini NIC (Audited on 3/31/25)
export FI_CXI_RDZV_EAGER_SIZE=0		# Performance tuning for RCCL + HPE Slingshot Cassini NIC (Audited on 3/31/25)

# Known issue with memhooks and RCCL hangs (Audited on 3/31/25)
# https://support.hpe.com/hpesc/public/docDisplay?docId=dp00004854en_us&docLocale=en_US
export FI_MR_CACHE_MONITOR=userfaultfd		# Known issue with memhooks and RCCL hang (Audited on 3/31/25)
export FI_CXI_DEFAULT_TX_SIZE=1024		# Performance tuning for HPE Slingshot Cassini NIC (Audited on 3/31/25)
export FI_CXI_DISABLE_HOST_REGISTER=1		# Performance tuning for HPE Slingshot Cassini NIC (Audited on 3/31/25)
export FI_CXI_DEFAULT_CQ_SIZE=131072		# Performance tuning for HPE Slingshot Cassini NIC (Audited on 3/31/25)
export FI_CXI_RX_MATCH_MODE=hybrid		# Performance tuning for HPE Slingshot Cassini NIC (Audited on 3/31/25)

# General tuning knobs (Audited on 3/31/25)
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsi0
export TORCHRUN_HPC_SCHEDULER=FluxScheduler
export RANK=${FLUX_TASK_RANK}
export TORCHRUN_HPC_MASTER_ADDR=`flux hostlist local | /bin/hostlist -n 1`
export TORCHRUN_HPC_MASTER_PORT=23456
export TORCHRUN_HPC_RDV_PROTOCOL="tcp://${TORCHRUN_HPC_MASTER_ADDR}:${TORCHRUN_HPC_MASTER_PORT}"
export TORCHRUN_HPC_MAX_GPU_MEM=0.8

export PYTHONPATH=/usr/WS2/bartolds/lingua:${PYTHONPATH}
/usr/workspace/bartolds/x86_miniconda3/envs/lingua_rocm64/bin/python -u /usr/WS2/bartolds/lingua/torchrun_hpc-train.py_2025-05-08_13h31m19s/torchrun_hpc_trampoline.py /usr/WS2/bartolds/lingua/apps/main/train.py config=apps/main/configs/comma_7B_brb.yaml

# Launch command: flux run -N32 -n128 -u -onosetpgrp --exclusive -ofastload=on --setattr=rdzv_get_en=0 -ompibind=omp_proc_bind,omp_places --gpus-per-task=1 /usr/WS2/bartolds/lingua/torchrun_hpc-train.py_2025-05-08_13h31m19s/launch.sh
# User command invoked: /usr/workspace/bartolds/x86_miniconda3/envs/lingua_rocm64/bin/torchrun-hpc -N 32 -n 4 --gpus-per-proc 1 apps/main/train.py config=apps/main/configs/comma_7B_brb.yaml
