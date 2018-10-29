# mike2018

This repository contains the implementation of the experiments proposed in the paper [*Using stigmergy to incorporate the time into artificial neural networks*](http://www.iet.unipi.it/m.cimino/publications/cimino_pub62.pdf).  
If you are interested on the **actual implementation** of the **Stigmergic Neural Networks** please check out the  [**torchsnn repository**](https://github.com/galatolofederico/torchsnn)


## Installation

Clone this repository
```
git clone https://github.com/galatolofederico/mike2018 && cd mike2018
```
Create a python virtualenv and activate it, make sure to use **python3**
```
virtualenv --python=/usr/bin/python3 env && source ./env/bin/activate
```
Install the requirements
```
pip install -r requirements.txt
```
You are ready to go!


## Contents

### mnist/mnist.py

Python script to train and evaluate all the architectures described in the paper. It uses the [sacred](https://github.com/IDSIA/sacred/tree/master/sacred) framework to manage experiments configurations and results.  
It uses the sacred-style to set the configuration variables
```
python3 mnist.py with config1=val1 config2=val2
```
For example
```
python3 mnist.py with batch_size=20 use_mongo=True
```

You can set the following configuration variables

|Variable|Description|Default|
|---|---|---|
|arch|Architecture to use (possible values: 'stigmergic', 'feedforward', 'recurrent', 'lstm')|stigmergic|
|n_hidden|Number of hidden neurons|10|
|n_layers|Number of hidden layers (valid only for feedforward and lstm)|1 for feedforward and 3 for lstm|
|avg_window|Moving average window size for logging|100|
|use_mongo|Use MongoDB Observer to log the experiments|False|


### mnist/networks/*.py

Python implementation of all the architectures described in the paper

### xor.py

Train and evaluation of the xor problem using only **one stigmergic perceptron**

## Citing

If you want to cite us please use this BibTeX

```
@article{galatolo_snn
,	author	= {Galatolo, Federico A and Cimino, Mario GCA and Vaglini, Gigliola}
,	title	= {Using stigmergy to incorporate the time into artificial neural networks}
,	journal	= {MIKE 2018}
,	year	= {2018}
}
```

## Contributing

This code is released under GNU/GPLv3 so feel free to fork it and submit your changes, every PR helps.  
If you need help using it or for any question please reach me at [galatolo.federico@gmail.com](mailto:galatolo.federico@gmail.com) or on Telegram  [@galatolo](https://t.me/galatolo)