OUT_PATH=./ckpts/blrp-dinov2-vitl14-reg-lora0-fw-5e-06-mt09996-bs3-800/checkpoints/
[ ! -d $OUT_PATH ] && echo "Downloading ckpt" && mkdir -p $OUT_PATH &&  wget https://huggingface.co/AmazonScience/MILE/resolve/main/blrp-dinov2-vitl14-reg-lora0-fw-5e-06-mt09996-bs3-800_ckpt0390.pth $OUT_PATH

OUT_PATH=./ckpts/mile-vitl14-reg-LRx8-g48x8-100M-mt0996-bs3-s250000-ep800-1/checkpoints/
[ ! -d $OUT_PATH ] && echo "Downloading ckpt" && mkdir -p $OUT_PATH &&  wget https://huggingface.co/AmazonScience/MILE/resolve/main/mile-vitl14-reg-LRx8-g48x8-100M-mt0996-bs3-s250000-ep800-1_ckpt0361.pth $OUT_PATH


cd ..

CKPT_ROOT=./test/ckpts/

# iNaturalist
DATA_PATH=./data_preprocesing/inat/inat_raw_data/inference_by_class
python inference_MILE.py --local_run  --ckpt_root $CKPT_ROOT  --dataset_name iNaturalist_100  --test_source query  --target_source target  --patch_size 16  --view multi-view  --output_type latent  --k_max 10  --data_path $DATA_PATH  --subset_classes 100  --arch maws  --peft on  --samples_per_class 4  --base_model_name vit_h14_maws  --model_name blrp-vit-h14-maws-nat-lora0-fw-5e-06-mt09996-bs2-8000 --ckpt_name checkpoint2475.pth

# ABO
DATA_PATH=./data_preprocesing/abo/abo_raw_data/inference_by_class
python inference_MILE.py --local_run  --ckpt_root $CKPT_ROOT --data_path  $DATA_PATH --dataset_name Abo_retrieval_test --samples_per_class 4   --test_source test.query  --target_source test.target  --patch_size 14  --view multi-view  --output_type latent --k_max 10   --peft on   --arch dinov2 --model_name blrp-dinov2-vitl14-reg-lora0-fw-5e-06-mt09996-bs3-800 --ckpt_name checkpoint0390.pth

cd -
