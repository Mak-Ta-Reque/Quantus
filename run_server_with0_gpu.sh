srun \
  --container-image=/netscratch/duynguyen/Docker-Image/vissl_A100.sqsh -v\
  --mem-per-cpu=16G\
  --container-mounts=/netscratch/software:/netscratch/software:ro,/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
  --container-workdir="`pwd`" \
  --time=04:00:00 \
  --task-prolog=install.sh \
  start_code_server.sh
