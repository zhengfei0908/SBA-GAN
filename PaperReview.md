### GAN-CLS-INT

用DCGAN实现，提出了

- CLS：D部分给real-mismatched text pair的数据（除了原本的real-matched text和fake-matched text pair）
- INT：对已有文本的embedding进行差值得到mismatched text，增强数据流型的smooth


### StackGAN（StackGAN++）

- 在word-embedding部分，提出了Conditional Augamentation，即通过full-connect layer 把embedded representation映射成 均值向量 and 标准差向量，增加了生成图片的robustness和variation。（通过KL loss增加映射空间的smooth）
- 通过stack2个GAN网络（用了相同的embedding和不同的Augamentationt提取了不同级别的特征）
- StackGAN++（还没仔细看）

### AttnGAN

- 提取了句子的sentence级别和word级别信息。
- 网络结构类似于stackGAN++，在不同stack之间加入了word级别信息的attention模型的结果
- 还需要具体看看
### StyleGAN
### LoGAN
- 就是刚发的Conditional StyleGAN那篇
- 个人感觉比较low，就是加了个one-hot的label向量（虽然有一点randomness）

### 几点总结
- 数据集方面，大多利用了CUB（鸟数据集）和COCO（各种图片），也有用到OXFORD（花数据集），为了能够比较结果，我们应该也需要用这两个
- metrics方面，大多数T2I论文用了inception score，然后也用了R-precision去衡量text和image的match程度（这部分还需要具体设计一下）
- styleGAN用自己的方式做了Conditional Augamentation，然后G网络部分的pgGAN也是stackGAN更高级的（个人感觉），所以对于之前的T2I的几点improvement，styleGAN似乎都有更高的应用形式
- 可能做的方向：通过attention把sentence映射成不同级别的embedding，加入z后嵌入G网络的block之间从而更细致地影响图片的生成；style-content分离，因为text更多影响content，z更多影响style（背景），希望能够有某种方式更好地解耦