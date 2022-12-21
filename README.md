<div align="center">    
 
# Deep Traffic Generation Paper
<!--
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  

ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)

![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)
-->

<!--  
Conference   
-->   
</div>

## Desciption
Package that provides neural networks (mostly autoencoders) to embed and generate traffic trajectories. This project relies on [traffic](https://traffic-viz.github.io/) and [Pytorch-Lightning](https://www.pytorchlightning.ai/) libraries. This repositroy provides the plots in the paper [Deep Generative Modelling of Aircraft Trajectories in Terminal Maneuvering Areas](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4254106)

## Installation
```bash
# create new python environment for traffic
conda create -n traffic -c conda-forge python=3.10
pip install traffic
conda activate traffic

# clone project   
git clone https://github.com/kruuZHAW/deep-traffic-generation-paper

# install package
cd code
pip install .
```

## How to run 
 Navigate to `code` folder and run `run.sh`. To choose to train the model rather than display the plots, comment and uncomment the corresponding lines in `run.sh`.   
 
 ```bash
cd code
bash run.sh
```

You can use Tensorboard to visualize training logs.

```bash
cd code/deep-traffic-generation
tensorboard --logdir lightning_logs --bind_all
```

## Publication
