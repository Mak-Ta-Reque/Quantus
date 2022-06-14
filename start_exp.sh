srun \
  --container-image=/netscratch/duynguyen/Docker-Image/vissl_A100.sqsh -v\
  --cpus-per-task=8\
  --mem-per-cpu=16G\
  --container-mounts=/netscratch/software:/netscratch/software:ro,/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" \
  --time=04:00:00 \
  --task-prolog=install.sh \
  python run_exp.py