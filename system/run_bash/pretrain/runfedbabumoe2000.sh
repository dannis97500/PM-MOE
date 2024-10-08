
python main.py -data MNIST -m cnn -nb 10 -algo PMOE_FedBABU -gr 2000 --fine_tuning_epochs 50 -did 0 >> logs/noniid_s/beforemoe/MNIST_PMOE_FedBABU_before_moe.log 2>&1

python main.py -data Cifar10 -m cnn -nb 10 -algo PMOE_FedBABU -gr 2000 --fine_tuning_epochs 50 -did 0 >> logs/noniid_s/beforemoe/cifar10_PMOE_FedBABU_before_moe.log 2>&1

python main.py -data Cifar100 -m cnn -nb 100 -algo PMOE_FedBABU -gr 2000 --fine_tuning_epochs 50  -did 0 >> logs/noniid_s/beforemoe/cifar100_PMOE_FedBABU_before_moe.log 2>&1

python main.py -data FashionMNIST -m cnn -nb 10 -algo PMOE_FedBABU -gr 2000 --fine_tuning_epochs 50 -did 0 >> logs/noniid_s/beforemoe/FashionMNIST_PMOE_FedBABU_before_moe.log 2>&1

python main.py -data TinyImagenet -m cnn -nb 200 -algo PMOE_FedBABU -gr 2000 --fine_tuning_epochs 50 -did 0 >> logs/noniid_s/beforemoe/TinyImagenet_PMOE_FedBABU_before_moe.log 2>&1

python main.py -data AGNews -m fastText -nb 4 -algo PMOE_FedBABU -gr 2000 --fine_tuning_epochs 50 -did 0 >> logs/noniid_s/beforemoe/AGNews_PMOE_FedBABU_before_moe.log 2>&1


