# Robustly overfitting latents for flexible neural image compression
This is the PyTorch code for the paper: [Robustly overfitting latents for flexible neural image compression](https://arxiv.org/abs/2401.17789). In this repository various method implementations from the paper can be found.

The mean-scale hyperprior from [(Minnen et al., 2018)](https://arxiv.org/abs/1809.02736) is trained in PyTorch and uses the package from [compressAI](https://interdigitalinc.github.io/CompressAI/models.html).

This work builts upon [Improving Inference for Neural Image Compression (Yang et al., 2020)](https://arxiv.org/abs/2006.04240), and is partly re-implementated from TensorFlow to PyTorch from [github](https://github.com/mandt-lab/improving-inference-for-neural-image-compression).

A BibTeX entry for LaTeX users:
```
@article{perugachi2024robustly,
  title={Robustly overfitting latents for flexible neural image compression},
  author={Perugachi-Diaz, Yura and Gansekoele, Arwin and Bhulai, Sandjai},
  journal={arXiv preprint arXiv:2401.17789},
  year={2024}
}
```

## Abstract
Neural image compression has made a great deal of progress. State-of-the-art models are based on variational autoencoders and are outperforming classical models. Neural compression models learn to encode an image into a quantized latent representation that can be efficiently sent to the decoder, which decodes the quantized latent into a reconstructed image. While these models have proven successful in practice, they lead to sub-optimal results due to imperfect optimization and limitations in the encoder and decoder capacity. Recent work shows how to use stochastic Gumbel annealing (SGA) to refine the latents of pre-trained neural image compression models. We extend this idea by introducing SGA+, which contains three different methods that build upon SGA. Further, we give a detailed analysis of our proposed methods, show how they improve performance, and show that they are less sensitive to hyperparameter choices. Besides, we show how each method can be extended to three- instead of two-class rounding. Finally, we show how refinement of the latents with our best-performing method improves the compression performance on the Tecnick dataset and how it can be deployed to partly move along the rate-distortion curve.

## Requirements:
- Python (tested with 3.11)
- PyTorch (tested with 2.2.0)
- CompressAI (tested with 1.2.4)

To install requirements run:
```
pip install -r requirements.txt
```

## Training
The models are pre-trained from scratch with λ ∈ {0.001, 0.0025, 0.005, 0.01, 0.02, 0.04, 0.08}. Additional default settings can be found in the example file: `example_files/pretrain.json`.

### Downloaded training datasets
The training dataset consist of a combination of the full size ImageNet2012 and CLIC2020 dataset and are assumed to be in .tfrecord format. ImageNet needs to be transformed into .tfrecords, this is not supported in this codebase. For evaluation the Kodak dataset is used.
- [ImageNet2012](https://www.image-net.org/challenges/LSVRC/2012/)
- [CLIC2020](https://www.tensorflow.org/datasets/catalog/clic) is automatically downloaded
- [Kodak](https://r0k.us/graphics/kodak/)

### Running script
To train a mean-scale hyperprior from scratch, as in the paper, run the following command:
```
python train.py --load_json true --config example_files/pretrain.json
```

## Inference
After pre-training a model the (hyper)latents are optimized for 2000 iterations to improve the overall compression performance.

### Downloaded inference datasets
Optimizing latents can be done on one the following datasets:
- [Kodak](https://r0k.us/graphics/kodak/)
- [Tecnick](https://sourceforge.net/projects/testimages/files/OLD/OLD_SAMPLING)

### Running Script
To improve the inference, as in the paper, run the following command:
```
python train_flexible_inference.py --load_json true --config example_files/flexible_inference.json
```

## Inference methods: optimizing latents
The SGA+ methods for two- or three-class rounding can be specified under `dev_options` in the .json file (see: `example_files/flexible_inference.json`):
```
"dev_options": {
  "improving_methods": {
                "sga_logits": false,
                "sga_logits_3c": true
  }
}
```
The best settings for the proposed SGA+ methods are the default settings in `experimental_inference_methods.py`.

### Two-class rounding
The main SGA+ methods for two-class rounding to improve the compression performance, can be specified in `dev_options` by setting the following flag to true: 
```
"sga_logits": true
```
Next, in the following flag the SGA+ method can be specified:
```
"sga_logits_params": {
            "logits": "ssl",
            "ub": 1.0
        }
```
`"ub"` indicates the upper bound of the annealed temperature, default is 1. An SGA+ method can be specified in `"logits"` to one of the following commands:
- "logx" (linear)
- "cosx" (cosine)
- "ssl" (sigmoid scaled logit)

When using `"ssl"` do not forget to set its scale factor, resembling a function of your choice. The default is linear:
```
"scale_factor": 1
```

### Three-class rounding
For three-class rounding with SGA+, in `dev_options`, set the following flag to true:
```
"sga_logits_3c": true
```
Similarly as for the two-class rounding, the SGA+ method can be specified in:
```
"sga_logits_params": {
            "logits": "log_ssl_a_factor_pow",
            "ub": 1.0
        }
```
Specify one of the following SGA+ methods for flag `"logits"`:
- "log_absx_factor_pow" (extended linear version)
- "log_cosx_factor_pow" (extended cosine version)
- "log_ssl_a_factor_pow" (extended ssl version)

Do not forget to set a scaling factor (for ssl), power and factor:
```
"scale_factor": 1.4,
"3c_power": 1.4,
"3c_factor": 0.98
```

### Inference lambda
To train with another inference lambda than a pre-trained model is trained on, initialize and set the following flag in `dev_options`:
```
"inference_lambda": 0.01
```