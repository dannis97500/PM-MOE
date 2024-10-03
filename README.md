# PM-MOE: Personalized Model Parameters with Mixture of Experts for Personalized Federated Learning

We continue to develop our algorithms on*** PFLlib***-- [Personalized Federated Learning Algorithm Library](https://arxiv.org/abs/2312.04992) 
***Special thanks to the [Jianqing Zhang](https://arxiv.org/search/cs?searchtype=author&query=Zhang,+J) who provided the personalized federated learning algorithm library***



To address the statistical heterogeneity of data in federated learning, research on personalized federated learning has made notable progress. To generate personalized models that better match the data domain, model-split-based personalized federated learning algorithms divide the model into a globally shared part and a locally private part. However, optimizing the local model while aggregating makes it challenging to effectively utilize the personalized knowledge from various clients. The locally private parameters after model convergence better represent the knowledge of the data domain. To overcome these limitations, we propose a personalized model parameter mixing expert (PM-MOE) method. Notably, this architecture features a two-phase training process, allowing each client to autonomously select the personalized model parameters converged by other clients. With only a few training iterations, PM-MOE can enhance a range of model-split-based personalized federated learning algorithms. Additionally, we conducted extensive experiments on six widely used datasets, demonstrating the superiority of our proposed method across two data splitting modes.



## Comparison of main algorithms 

> ### Traditional FL (tFL)

- **FedAvg** — [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html) *AISTATS 2017*

  ***Update-correction-based tFL***

- **SCAFFOLD** - [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a.html) *ICML 2020*

  ***Regularization-based tFL***

- **FedProx** — [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127) *MLsys 2020*

- **MOON** — [Model-Contrastive Federated Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Model-Contrastive_Federated_Learning_CVPR_2021_paper.html) *CVPR 2021*

  ***Knowledge-distillation-based tFL***

- **FedGen** — [Data-Free Knowledge Distillation for Heterogeneous Federated Learning](http://proceedings.mlr.press/v139/zhu21b.html) *ICML 2021*

> ### Personalized FL (pFL)

***Model-splitting-based pFL***

- **FedPer** — [Federated Learning with Personalization Layers](https://arxiv.org/abs/1912.00818) *2019*
- **LG-FedAvg** — [Think Locally, Act Globally: Federated Learning with Local and Global Representations](https://arxiv.org/abs/2001.01523) *2020*
- **FedRep** — [Exploiting Shared Representations for Personalized Federated Learning](http://proceedings.mlr.press/v139/collins21a.html) *ICML 2021*
- **FedRoD** — [On Bridging Generic and Personalized Federated Learning for Image Classification](https://openreview.net/forum?id=I1hQbx10Kxn) *ICLR 2022*
- **FedBABU** — [Fedbabu: Towards enhanced representation for federated image classification](https://openreview.net/forum?id=HuaYQfggn5u) *ICLR 2022*
- **FedCP** — [FedCP: Separating Feature Information for Personalized Federated Learning via Conditional Policy](https://arxiv.org/pdf/2307.01217v2.pdf) *KDD 2023*
- **GPFL** — [GPFL: Simultaneously Learning Generic and Personalized Feature Information for Personalized Federated Learning](https://arxiv.org/pdf/2308.10279v3.pdf) *ICCV 2023*
- **FedGH** — [FedGH: Heterogeneous Federated Learning with Generalized Global Header](https://dl.acm.org/doi/10.1145/3581783.3611781) *ACM MM 2023*
- **DBE** — [Eliminating Domain Bias for Federated Learning in Representation Space](https://openreview.net/forum?id=nO5i1XdUS0) *NeurIPS 2023*

> ### PM-MOE （pFL plus）
**PM-MOE is a plug-in integrated into the Model-splitting-based pFL algorithm. We adapted it to the following algorithm:**

- **FedPer+PM-MOE** 
- **LG-FedAvg+PM-MOE** 
- **FedRep+PM-MOE**
- **FedRoD+PM-MOE**
- **FedBABU+PM-MOE**
- **FedCP+PM-MOE**
- **GPFL+PM-MOE**
- **FedGH+PM-MOE**
- **DBE+PM-MOE** 


## Datasets and spilting method
For the ***label skew*** scenario, we used Dirichlet distribution with ***s=0 and s=20*** to split the data.
**MNIST**, **EMNIST**, **Fashion-MNIST**, **Cifar10**, **Cifar100**, **AG News**, they can be easy split into  **non-IID** version. 

In addition to the heterogeneous splitting method mentioned in PFLlib, we also added a categorical Dirichlet data splitting method with S sharing ratio.

- Dirichlet distribution with S=20: In the first setting, 20% of the data for each class is uniformly distributed among M clients, and the remaining data is assigned based on Dirichlet-distributed weights.
- Dirichlet distribution with S=0: In the second setting, no constraints are placed on class distribution across clients, with all data allocated based on Dirichlet-distributed weights.
Likewise,  we move these codes into `./dataset/utils/dataset_utils.py`. 

*If you need another data set, just write another code to download it and then use the utils.*

### Examples for **MNIST**
- MNIST
    ```
    cd ./dataset
    # python generate_MNIST.py noniid - dir # Dirichlet distribution with S=0
    # python generate_MNIST.py noniid - s_par # Dirichlet distribution with S=20
    ```

The output of `python generate_MNIST.py noniid - dir`
```
Number of classes: 10
Client 0         Size of data: 2630      Labels:  [0 1 4 5 7 8 9]
                 Samples of labels:  [(0, 140), (1, 890), (4, 1), (5, 319), (7, 29), (8, 1067), (9, 184)]
--------------------------------------------------
Client 1         Size of data: 499       Labels:  [0 2 5 6 8 9]
                 Samples of labels:  [(0, 5), (2, 27), (5, 19), (6, 335), (8, 6), (9, 107)]
--------------------------------------------------
Client 2         Size of data: 1630      Labels:  [0 3 6 9]
                 Samples of labels:  [(0, 3), (3, 143), (6, 1461), (9, 23)]
--------------------------------------------------
```
<details>
    <summary>Show more</summary>

    Client 3         Size of data: 2541      Labels:  [0 4 7 8]
                     Samples of labels:  [(0, 155), (4, 1), (7, 2381), (8, 4)]
    --------------------------------------------------
    Client 4         Size of data: 1917      Labels:  [0 1 3 5 6 8 9]
                     Samples of labels:  [(0, 71), (1, 13), (3, 207), (5, 1129), (6, 6), (8, 40), (9, 451)]
    --------------------------------------------------
    Client 5         Size of data: 6189      Labels:  [1 3 4 8 9]
                     Samples of labels:  [(1, 38), (3, 1), (4, 39), (8, 25), (9, 6086)]
    --------------------------------------------------
    Client 6         Size of data: 1256      Labels:  [1 2 3 6 8 9]
                     Samples of labels:  [(1, 873), (2, 176), (3, 46), (6, 42), (8, 13), (9, 106)]
    --------------------------------------------------
    Client 7         Size of data: 1269      Labels:  [1 2 3 5 7 8]
                     Samples of labels:  [(1, 21), (2, 5), (3, 11), (5, 787), (7, 4), (8, 441)]
    --------------------------------------------------
    Client 8         Size of data: 3600      Labels:  [0 1]
                     Samples of labels:  [(0, 1), (1, 3599)]
    --------------------------------------------------
    Client 9         Size of data: 4006      Labels:  [0 1 2 4 6]
                     Samples of labels:  [(0, 633), (1, 1997), (2, 89), (4, 519), (6, 768)]
    --------------------------------------------------
    Client 10        Size of data: 3116      Labels:  [0 1 2 3 4 5]
                     Samples of labels:  [(0, 920), (1, 2), (2, 1450), (3, 513), (4, 134), (5, 97)]
    --------------------------------------------------
    Client 11        Size of data: 3772      Labels:  [2 3 5]
                     Samples of labels:  [(2, 159), (3, 3055), (5, 558)]
    --------------------------------------------------
    Client 12        Size of data: 3613      Labels:  [0 1 2 5]
                     Samples of labels:  [(0, 8), (1, 180), (2, 3277), (5, 148)]
    --------------------------------------------------
    Client 13        Size of data: 2134      Labels:  [1 2 4 5 7]
                     Samples of labels:  [(1, 237), (2, 343), (4, 6), (5, 453), (7, 1095)]
    --------------------------------------------------
    Client 14        Size of data: 5730      Labels:  [5 7]
                     Samples of labels:  [(5, 2719), (7, 3011)]
    --------------------------------------------------
    Client 15        Size of data: 5448      Labels:  [0 3 5 6 7 8]
                     Samples of labels:  [(0, 31), (3, 1785), (5, 16), (6, 4), (7, 756), (8, 2856)]
    --------------------------------------------------
    Client 16        Size of data: 3628      Labels:  [0]
                     Samples of labels:  [(0, 3628)]
    --------------------------------------------------
    Client 17        Size of data: 5653      Labels:  [1 2 3 4 5 7 8]
                     Samples of labels:  [(1, 26), (2, 1463), (3, 1379), (4, 335), (5, 60), (7, 17), (8, 2373)]
    --------------------------------------------------
    Client 18        Size of data: 5266      Labels:  [0 5 6]
                     Samples of labels:  [(0, 998), (5, 8), (6, 4260)]
    --------------------------------------------------
    Client 19        Size of data: 6103      Labels:  [0 1 2 3 4 9]
                     Samples of labels:  [(0, 310), (1, 1), (2, 1), (3, 1), (4, 5789), (9, 1)]
    --------------------------------------------------
    Total number of samples: 70000
    The number of train samples: [1972, 374, 1222, 1905, 1437, 4641, 942, 951, 2700, 3004, 2337, 2829, 2709, 1600, 4297, 4086, 2721, 4239, 3949, 4577]
    The number of test samples: [658, 125, 408, 636, 480, 1548, 314, 318, 900, 1002, 779, 943, 904, 534, 1433, 1362, 907, 1414, 1317, 1526]
    
    Saving to disk.
    
    Finish generating dataset.
</details>

## Models
- for MNIST and Fashion-MNIST

    1. Mclr_Logistic(1\*28\*28)
    2. LeNet()
    3. DNN(1\*28\*28, 100) # non-convex

- for Cifar10, Cifar100 and Tiny-ImageNet

    1. Mclr_Logistic(3\*32\*32)
    2. FedAvgCNN()
    3. DNN(3\*32\*32, 100) # non-convex
    4. ResNet18, AlexNet, MobileNet, GoogleNet, etc.

- for AG_News and

    1. LSTM()
    2. fastText() in [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/) 
    3. TextCNN() in [Convolutional Neural Networks for Sentence Classification](https://aclanthology.org/D14-1181/)
    4. TransformerModel() in [Attention is all you need](https://proceedings.neurips.cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html)



## Environments
Install [CUDA](https://developer.nvidia.com/cuda-11-6-0-download-archive). 

Install [conda](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) and activate conda. 

```bash
conda env create -f env_cuda_latest.yaml # You may need to downgrade the torch using pip to match the CUDA version
```

## How to start simulating (examples for FedPer+PM-MOE)

- Create proper environments (see [Environments](#environments)).

- Download [this project](https://github.com/dannis97500/PM-MOE) to an appropriate location using [git](https://git-scm.com/).
    ```bash
    git clone https://github.com/dannis97500/PM-MOE.git
    ```

## Using this code requires two steps, the pre-training phase and the PM-MOE fine-tuning phase

- Run Step 1 pre-train: 
    ```bash
    cd ./system
    python main.py -data MNIST -m cnn -nb 10 -algo PMOE_FedPer -gr 2000 -pls 1 -did 0 >> logs/noniid_s/beforemoe/MNIST_PMOE_FedPer_before_moe.log 2>&1
    
    # using the MNIST dataset, the PMOE_FedPer algorithm, and the 4-layer CNN model
    ```
    **Note**: In serverper.py, you need to comment the code for loading the model. This serves the second stage.
```python
for i in range(len(self.clients)):   # --------please cancel annotate in pmoe finetune --------
    client = self.clients[i]                    # --------please cancel annotate in pmoe finetune --
    loaded_client = self.load_clients(client)   # --------please cancel annotate in pmoe finetune --
    self.clients[i] = loaded_client             # --------please cancel annotate in pmoe finetune --
    
self.load_model()                              # --------please cancel annotate in pmoe finetune --------
```



- Run Step 2 PM-MOE finetune: 
    ```bash
    cd ./system
    python moe_finetune.py -data MNIST -m cnn -algo PMOE_FedPer -nb 10  --topk 8 --lock_experts 0  --moe_fine_tuning_epochs 50 --moe_lr 0.5 -did 0 >> logs/noniid_s/MNIST_PMOE_FedPer_lock_experts_topk8_moe_lr_0.5.log 2>&1
    ```
    **Note**: In serverper.py, you need to uncomment the code that loads the model trained in the first stage



## Experimental results
We have prepared scripts for pre-training and PM-MOE fine-tuning to reproduce the experimental results. If you are interested, you can reproduce this experimental result.


