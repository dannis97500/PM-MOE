python main.py -data MNIST -m cnn -nb 10 -algo MOON -gr 2000 -did 0 >> logs/noniid_s/normalFL/MNIST_MOON.log 2>&1

python main.py -data Cifar10 -m cnn -nb 10 -algo MOON -gr 2000 -did 0 >> logs/noniid_s/normalFL/cifar10_MOON.log 2>&1

python main.py -data Cifar100 -m cnn -nb 100 -algo MOON -gr 2000 -did 0 >> logs/noniid_s/normalFL/cifar100_MOON.log 2>&1

python main.py -data FashionMNIST -m cnn -nb 10 -algo MOON -gr 2000 -did 0 >> logs/noniid_s/normalFL/FashionMNIST_MOON.log 2>&1

python main.py -data TinyImagenet -m cnn -nb 200 -algo MOON -gr 2000 -did 0 >> logs/noniid_s/normalFL/TinyImagenet_MOON.log 2>&1

python main.py -data AGNews -m fastText -nb 4 -algo MOON -gr 2000 -did 0 >> logs/noniid_s/normalFL/AGNews_MOON.log 2>&1



# FedGen
# MOON
# FedProx
# SCAFFOLD