python main.py -data MNIST -m cnn -nb 10 -algo FedAvg -gr 2000 -did 0 >> logs/noniid_s/normalFL/MNIST_FedAvg.log 2>&1

python main.py -data Cifar10 -m cnn -nb 10 -algo FedAvg -gr 2000 -did 0 >> logs/noniid_s/normalFL/cifar10_FedAvg.log 2>&1

python main.py -data Cifar100 -m cnn -nb 100 -algo FedAvg -gr 2000 -did 0 >> logs/noniid_s/normalFL/cifar100_FedAvg.log 2>&1

python main.py -data FashionMNIST -m cnn -nb 10 -algo FedAvg -gr 2000 -did 0 >> logs/noniid_s/normalFL/FashionMNIST_FedAvg.log 2>&1

python main.py -data TinyImagenet -m cnn -nb 200 -algo FedAvg -gr 2000 -did 0 >> logs/noniid_s/normalFL/TinyImagenet_FedAvg.log 2>&1

python main.py -data AGNews -m fastText -nb 4 -algo FedAvg -gr 2000 -did 0 >> logs/noniid_s/normalFL/AGNews_FedAvg.log 2>&1



# FedGen
# MOON
# FedProx