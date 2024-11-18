OUT_PATH=./ckpts/blrp-dinov2-vitl14-reg-lora0-fw-5e-06-mt09996-bs3-800/checkpoints/
[ ! -d $OUT_PATH ] && echo "Downloading ckpt" && mkdir -p $OUT_PATH &&  wget https://huggingface.co/AmazonScience/MILE/resolve/main/blrp-dinov2-vitl14-reg-lora0-fw-5e-06-mt09996-bs3-800_ckpt0390.pth $OUT_PATH

OUT_PATH=./ckpts/mile-vitl14-reg-LRx8-g48x8-100M-mt0996-bs3-s250000-ep800-1/checkpoints/
[ ! -d $OUT_PATH ] && echo "Downloading ckpt" && mkdir -p $OUT_PATH &&  wget https://huggingface.co/AmazonScience/MILE/resolve/main/mile-vitl14-reg-LRx8-g48x8-100M-mt0996-bs3-s250000-ep800-1_ckpt0361.pth $OUT_PATH

cd ..

CKPT_ROOT=./test/ckpts/

DATA_PATH=/home/digrigor/ebs/MILE/abo_test/
python inference_MILE.py --local_run  --ckpt_root $CKPT_ROOT --data_path  $DATA_PATH --dataset_name Abo_retrieval_test --samples_per_class 4   --test_source test.query  --target_source test.target  --patch_size 14  --view multi-view  --output_type latent --k_max 10   --peft on   --arch dinov2 --model_name blrp-dinov2-vitl14-reg-lora0-fw-5e-06-mt09996-bs3-800 --ckpt_name blrp-dinov2-vitl14-reg-lora0-fw-5e-06-mt09996-bs3-800_ckpt0390.pth
  
DATA_PATH=/large_shared/Benchmark_top10_deduped_min4pics/recall_setup_1000
python inference_MILE.py --local_run  --ckpt_root $CKPT_ROOT  --arch dinov2  --samples_per_class 4  --peft on  --patch_size 14  --view multi-view  --output_type latent  --k_max 10  --dataset_name Benchmark_top10_deduped_min4pics_1000  --data_path $DATA_PATH  --test_source query  --target_source gallery  --model_name mile-vitl14-reg-LRx8-g48x8-100M-mt0996-bs3-s250000-ep800-1 --ckpt_name mile-vitl14-reg-LRx8-g48x8-100M-mt0996-bs3-s250000-ep800-1_ckpt0361.pth
cd -
 