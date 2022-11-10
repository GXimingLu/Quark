# QUARK

This is the official repo for the paper ["Quark: Controllable Text Generation with Reinforced Unlearning"](https://arxiv.org/abs/2205.13636) (NeurIPS 2022)

## Requirement
We suggest using conda to setup environment. You need to first replace ``prefix`` in [environment.yml](environment.yml) with your home path. With conda installed, create an environment called `quark` with:
```
conda env create -f environment.yml
```

## Instuction
The ``main`` branch contains **toxicity** unlearning task. We put the other two tasks, sentiment steering and repetition reduction in ``sentiment`` branch and ``repetition`` branch separately. 

We use the [PerspectiveAPI](https://github.com/conversationai/perspectiveapi) to score toxicity in reward computing, which requires API key for access.
Please refer to their website for API key application. 

### Training

Please first replace `PERSPECTIVE_API_KEY` in [constants.py](utils/constants.py) with your own API key.
For training quark for toxicity reduction with default hyperparameters,
```
python main.py
```
You can change hyperparameters in [arguments.py](arguments.py) via argparse.

### Evaluation

To evaluate the toxicity of unlearned model, please use [sample.py](sample.py). You need to first replace ``save_path`` and ``checkpoint_path`` with your output directory and model checkpoint path, then
```
python sample.py
```
It will save the evaluation result to your output directory.

To evaluate perplexity of the generations, please use [perplexity.py](perplexity.py). You need to first replace ``save_path`` with the same output directory specified above, then
```
python perplexity.py
```
It will save the perplexity result to the same output directory.


### Model Checkpoint
We release our model checkpoints for all three tasks: [toxicity unlearn](https://storage.googleapis.com/ai2-jack-public/quark/toxicity/ckp_11000.pth), sentiment steering ([positive](https://storage.googleapis.com/ai2-jack-public/quark/positive_sentiment/ckp_6000.pth), [negative](https://storage.googleapis.com/ai2-jack-public/quark/negative_sentiment/ckp_20000.pth)) and [repetition reduction](https://storage.googleapis.com/ai2-jack-public/quark/wiki/ckp_80000.pth).


## Citation
If you use this codebase in your work, please consider citing our paper:
```
@article{Lu2022QuarkCT,
  title={Quark: Controllable Text Generation with Reinforced Unlearning},
  author={Ximing Lu and Sean Welleck and Liwei Jiang and Jack Hessel and Lianhui Qin and Peter West and Prithviraj Ammanabrolu and Yejin Choi},
  journal={ArXiv},
  year={2022},
  volume={abs/2205.13636}
}
```



