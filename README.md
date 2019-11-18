# COMS4995_SBA-GAN

## Task
Fei Zheng: main model
- Main model, return CLASS model which can be called by Runtime Function.

Chirong Zhang: runtime functions
- Given model as parameter, custom runtime fuction to train and test model
- train.py (epoch, mini_batch, print_info, dir_creat, image_generate....)

Xiaoxi Zhao: dataloader, metrics and util functions
- dataloader.py(batch, dir....)
- metrics.py(inception score, R-precision)
- util.py...to be decided
#############finished############

## TODO
1. Code Integration
2. Progressive training
3. WGAN loss
4. Try different architectures

## Results:

Generated images: https://drive.google.com/open?id=11J_XfP8IE53kUUfCL9ZGYbF5ZRvPyejj

## Timeline
Proposal: Oct. 15th 11:30-12:00am

Milestone: Nov. 7th 11:30-12:00am

Final report time: Dec. 3rd 12:00-12:30pm


## Reference
### Paper:

BERT: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)

pgGAN: [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)

styleGAN: [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/abs/1812.04948)

cGAN: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004)

text2image: [Generative Adversarial Text to Image Synthesis](https://arxiv.org/abs/1605.05396)  

### Github:

styleGAN: [https://github.com/NVlabs/stylegan](https://github.com/NVlabs/stylegan)

text2image: [https://github.com/nashory/text2image-benchmark](https://github.com/nashory/text2image-benchmark)

StyleGAN-tf2: [https://github.com/ialhashim/StyleGAN-Tensorflow2](https://github.com/ialhashim/StyleGAN-Tensorflow2)

