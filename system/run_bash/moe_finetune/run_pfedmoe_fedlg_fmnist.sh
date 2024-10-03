

######################################### lock   
# 

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 2 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk2_moe_lr_0.1.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 4 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk4_moe_lr_0.1.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 8 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk8_moe_lr_0.1.log 2>&1


#  --moe_lr 0.05
python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 2 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.05 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk2_moe_lr_0.05.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 4 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.05 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk4_moe_lr_0.05.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 8 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.05 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk8_moe_lr_0.05.log 2>&1


#  --moe_lr 0.5
python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 2 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.5 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk2_moe_lr_0.5.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 4 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.5 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk4_moe_lr_0.5.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 8 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.5 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk8_moe_lr_0.5.log 2>&1



#  --moe_fine_tuning_epochs 100  
python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 2 --lock_experts 0  --moe_fine_tuning_epochs 100 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk2_moe_fine_tuning_epochs100.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 4 --lock_experts 0  --moe_fine_tuning_epochs 100 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk4_moe_fine_tuning_epochs100.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 8 --lock_experts 0  --moe_fine_tuning_epochs 100 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk8_moe_fine_tuning_epochs100.log 2>&1

#  --moe_fine_tuning_epochs 200
python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 2 --lock_experts 0  --moe_fine_tuning_epochs 200 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk2_moe_fine_tuning_epochs200.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 4 --lock_experts 0  --moe_fine_tuning_epochs 200 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk4_moe_fine_tuning_epochs200.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 8 --lock_experts 0  --moe_fine_tuning_epochs 200 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk8_moe_fine_tuning_epochs200.log 2>&1



#  --moe_fine_tuning_epochs 300
python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 2 --lock_experts 0  --moe_fine_tuning_epochs 300 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk2_moe_fine_tuning_epochs300.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 4 --lock_experts 0  --moe_fine_tuning_epochs 300 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk4_moe_fine_tuning_epochs300.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 8 --lock_experts 0  --moe_fine_tuning_epochs 300 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk8_moe_fine_tuning_epochs300.log 2>&1

###############################################

#  --moe_fine_tuning_epochs 400
python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 2 --lock_experts 0  --moe_fine_tuning_epochs 400 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk2_moe_fine_tuning_epochs400.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 4 --lock_experts 0  --moe_fine_tuning_epochs 400 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk4_moe_fine_tuning_epochs400.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 8 --lock_experts 0  --moe_fine_tuning_epochs 400 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk8_moe_fine_tuning_epochs400.log 2>&1






######################################### lock   
# 

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 16 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk16_moe_lr_0.1.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 20 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk20_moe_lr_0.1.log 2>&1


#  --moe_lr 0.05
python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 16 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.05 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk16_moe_lr_0.05.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 20 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.05 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk20_moe_lr_0.05.log 2>&1



#  --moe_lr 0.5
python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 16 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.5 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk16_moe_lr_0.5.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 20 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.5 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk20_moe_lr_0.5.log 2>&1




#  --moe_fine_tuning_epochs 100  
python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 16 --lock_experts 0  --moe_fine_tuning_epochs 100 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk16_moe_fine_tuning_epochs100.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 20 --lock_experts 0  --moe_fine_tuning_epochs 100 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk20_moe_fine_tuning_epochs100.log 2>&1


#  --moe_fine_tuning_epochs 200
python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 16 --lock_experts 0  --moe_fine_tuning_epochs 200 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk16_moe_fine_tuning_epochs200.log 2>&1

python moe_finetune.py -data FashionMNIST -m cnn -algo PMOE_LG_FedAvg -nb 10  --topk 20 --lock_experts 0  --moe_fine_tuning_epochs 200 --moe_lr 0.1 -did 0 >> logs/noniid_s/FashionMNIST_PMOE_LG_FedAvg_lock_experts_topk20_moe_fine_tuning_epochs200.log 2>&1


