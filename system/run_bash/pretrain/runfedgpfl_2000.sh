# -lam 0.01 -mu 0.1 for 4-layer CNN and 3-layer MLP
# -lam 0.0001 -mu 0.0 for ResNet-18 and fastText
# -lam 0.01 -mu 1.0 for HAR-CNN

python main.py -data MNIST -m cnn -nb 10 -algo PMOE_GPFL -gr 2000 -lam 0.01 -mu 0.1 -did 0 >> logs/noniid_s/beforemoe/MNIST_PMOE_GPFL_before_moe.log 2>&1

python main.py -data Cifar10 -m cnn -nb 10 -algo PMOE_GPFL -gr 2000 -lam 0.01 -mu 0.1 -did 0 >> logs/noniid_s/beforemoe/cifar10_PMOE_GPFL_before_moe.log 2>&1

python main.py -data Cifar100 -m cnn -nb 100 -algo PMOE_GPFL -gr 2000 -lam 0.01 -mu 0.1  -did 0 >> logs/noniid_s/beforemoe/cifar100_PMOE_GPFL_before_moe.log 2>&1

python main.py -data FashionMNIST -m cnn -nb 10 -algo PMOE_GPFL -gr 2000 -lam 0.01 -mu 0.1 -did 0 >> logs/noniid_s/beforemoe/FashionMNIST_PMOE_GPFL_before_moe.log 2>&1

python main.py -data TinyImagenet -m cnn -nb 200 -algo PMOE_GPFL -gr 2000 -lam 0.01 -mu 0.1 -did 0 >> logs/noniid_s/beforemoe/TinyImagenet_PMOE_GPFL_before_moe.log 2>&1

python main.py -data AGNews -m fastText -nb 4 -algo PMOE_GPFL -gr 2000 -lam 0.0001 -mu 0.0 -did 0 >> logs/noniid_s/beforemoe/AGNews_PMOE_GPFL_before_moe.log 2>&1

