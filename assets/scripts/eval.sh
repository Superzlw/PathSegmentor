export DETECTRON2_DATASETS=/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets
export DATASET=/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets
export DATASET2=/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets
export VLDATASET=/jhcnas5/chenzhixuan/data/PathSeg/biomedparse_datasets
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
#export WANDB_KEY=YOUR_WANDB_KEY # Provide your wandb key here

cuda_devices="0,1,3,5"
nproc_per_node=$(echo $cuda_devices | tr ',' '\n' | wc -l)
test_bs_total=$nproc_per_node

CUDA_VISIBLE_DEVICES=$cuda_devices mpirun -n $nproc_per_node python entry.py \
            --command evaluate \
            --conf_files configs/pathsegmentor.yaml \
            --overrides \
            VISUALIZE_FOR_EVAL False \
            PORT '36874' \
            MODEL.DECODER.HIDDEN_DIM 512 \
            MODEL.ENCODER.CONVS_DIM 512 \
            MODEL.ENCODER.MASK_DIM 512 \
            TEST.BATCH_SIZE_TOTAL $test_bs_total \
            FP16 True \
            WEIGHT True \
            STANDARD_TEXT_FOR_EVAL False \
            RESUME_FROM /jhcnas5/chenzhixuan/MyOpenSource/huggingface/PathSegmentor/model_state_dict.pt \
            