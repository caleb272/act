python3 ./imitate_episodes.py --task_name sim_stack_block_scripted --ckpt_dir ./ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0


python3 ./imitate_episodes.py --task_name sim_stack_block_400_scripted --ckpt_dir ./ckpt/stack_block_400_len_chnk_size_25 --policy_class ACT --kl_weight 10 --chunk_size 25 --hidden_dim 512 --batch_size 6 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0







### TEMP RE_RUN_THIS_COMMAND
python3 ./imitate_episodes.py --task_name sim_stack_block_400_scripted --ckpt_dir /home/mlrig/Documents/act/ckpt/sim_stack_block_400_scripted_hidden_dim_512_bs_16_cs_16 --policy_class ACT --kl_weight 10 --chunk_size 16 --hidden_dim 512 --batch_size 16 --dim_feedforward 3200 --num_epochs 10000 --lr 1e-5 --seed 0
