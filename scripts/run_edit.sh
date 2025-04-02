version=512
seed=123
name=UES

## DynamiCrafter w/ UES
ckpt='./checkpoints/UES_DC_512x320.ckpt'

## VideoCrafter2 w/ UES
# ckpt='./checkpoints/UES_VC2_512x320.ckpt'

config=configs/inference_v1.0.yaml

prompt_dir=prompts/edit_example/
res_dir="results/editing_output"


CUDA_VISIBLE_DEVICES=0 python3 scripts/evaluation/inference.py \
--seed ${seed} \
--ckpt_path $ckpt \
--config $config \
--savedir $res_dir/$name/delta \
--n_samples 1 \
--bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 7.5 \
--multiple_cond_cfg --cfg_img 7.5 \
--ddim_steps 50 \
--ddim_eta 1.0 \
--prompt_dir $prompt_dir \
--text_input \
--video_length 16 \
--frame_stride 6 \
--timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae
fi

## set "--cfg_img" to 0 when generation