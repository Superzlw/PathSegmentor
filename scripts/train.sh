export DETECTRON2_DATASETS=/home/chenzhixuan/Workspace/PathSegmentor/pathseg_datasets
export DATASET=/home/chenzhixuan/Workspace/PathSegmentor/pathseg_datasets
export DATASET2=/home/chenzhixuan/Workspace/PathSegmentor/pathseg_datasets
export VLDATASET=/home/chenzhixuan/Workspace/PathSegmentor/pathseg_datasets
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
#export WANDB_KEY=YOUR_WANDB_KEY # Provide your wandb key here

cuda_devices="1,2"
bs_per_gpu=1
nproc_per_node=$(echo $cuda_devices | tr ',' '\n' | wc -l)
bs_total=$(($bs_per_gpu * $nproc_per_node))
test_bs_total=$nproc_per_node

CUDA_VISIBLE_DEVICES=$cuda_devices mpirun -n $nproc_per_node python entry.py \
            --command train \
            --conf_files configs/pathsegmentor.yaml \
            --overrides \
            PORT '36877' \
            WANDB True \
            EVAL_AT_START False \
            VISUALIZE_FOR_EVAL False \
            FP16 True \
            RANDOM_SEED 2024 \
            PathSeg.INPUT.IMAGE_SIZE 1024 \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL $test_bs_total \
            TRAIN.BATCH_SIZE_TOTAL $bs_total \
            TRAIN.BATCH_SIZE_PER_GPU $bs_per_gpu \
            SOLVER.MAX_NUM_EPOCHS 20 \
            SOLVER.BASE_LR 0.00001 \
            SOLVER.FIX_PARAM.backbone False \
            SOLVER.FIX_PARAM.lang_encoder False \
            SOLVER.FIX_PARAM.pixel_decoder False \
            MODEL.DECODER.COST_SPATIAL.CLASS_WEIGHT 1.0 \
            MODEL.DECODER.COST_SPATIAL.MASK_WEIGHT 1.0 \
            MODEL.DECODER.COST_SPATIAL.DICE_WEIGHT 1.0 \
            MODEL.DECODER.TOP_SPATIAL_LAYERS 10 \
            MODEL.DECODER.SPATIAL.ENABLED True \
            MODEL.DECODER.GROUNDING.ENABLED True \
            LOADER.SAMPLE_PROB prop \
            FIND_UNUSED_PARAMETERS True \
            ATTENTION_ARCH.SPATIAL_MEMORIES 32 \
            MODEL.DECODER.SPATIAL.MAX_ITER 0 \
            ATTENTION_ARCH.QUERY_NUMBER 3 \
            STROKE_SAMPLER.MAX_CANDIDATE 10 \
            WEIGHT True \
            RESUME_FROM /jhcnas5/chenzhixuan/MyOpenSource/huggingface/PathSegmentor/model_state_dict.pt