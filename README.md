# **sNeuron-TST**

Code implements for paper **[Style-Specific Neurons for Steering LLMs in Text Style Transfer](http://arxiv.org/abs/2410.00593)** published in EMNLP 2024 main conference.
The neuron identification processes are based on public code : [Language-Specific-Neurons](https://github.com/RUCAIBox/Language-Specific-Neurons)

---

**Requirements**

1. Transformers (>=4.37.0)
2. Pytorch (>=1.9.0)

---

This repository contains the **official implementation** of the following paper:

> **Style-Specific Neurons for Steering LLMs in Text Style Transfer** http://arxiv.org/abs/2410.00593
>
> **Abstract:** _Text style transfer (TST) aims to modify the style of a text without altering its original meaning. Large language models (LLMs) demonstrate superior performance across multiple tasks, including TST. However, in zero-shot setups, they tend to directly copy a significant portion of the input text to the output without effectively changing its style. To enhance the stylistic variety and fluency of the text, we present sNeuron-TST, a novel approach for steering LLMs using style-specific neurons in TST. Specifically, we identify neurons associated with the source and target styles and deactivate source-style-only neurons to give target-style words a higher probability, aiming to enhance the stylistic diversity of the generated text. However, we find that this deactivation negatively impacts the fluency of the generated text, which we address by proposing an improved contrastive decoding method that accounts for rapid token probability shifts across layers caused by deactivated source-style neurons. Empirical experiments demonstrate the effectiveness of the proposed method on six benchmarks, encompassing formality, toxicity, politics, politeness, authorship, and sentiment._

---
**Pipeline**

+ Data Preparation
   + Download the dataset for six benchmarks we evaluated:
        - Formality: [GYAFC Dataset](https://github.com/raosudha89/GYAFC-corpus) (Liciences required, please email the original author.)
        - Toxicity: [ParaDetox](https://github.com/s-nlp/paradetox)
        - Politics: [RtGender](https://nlp.stanford.edu/robvoigt/rtgender/)
        - Politeness: [Politness](https://github.com/tag-and-generate/politeness-dataset)
        - Authorship: [Shakespeare](https://github.com/harsh19/Shakespearizing-Modern-English)
        - Sentiment: [Yelp](https://www.yelp.com/dataset)
    + Train/Test spliting using the code from ```data_pre```

+ Activation Storage
```
python activation.py -m $MODEL -s $STYLE -d $DATA
```
+ Identify Neurons
```
python identify.py
```
+ Remove intersection for each neurons
```
python Analysis/select_neurons.py
```

+ Generate with Dola
```
python Our/run_gen_dola.py
```
+ Evaluation
    - code in ```Evaluation```, including style cls, fluency, bleurt, similarity (labse and wieting)

****
If you find our paper useful, please kindly cite our paper. Thanks!
```bibtex
@article{lai2024style,
  title={Style-Specific Neurons for Steering LLMs in Text Style Transfer},
  author={Lai, Wen and Hangya, Viktor and Fraser, Alexander},
  journal={arXiv preprint arXiv:2410.00593},
  year={2024}
}
```
   
### Contact
If you have any questions about our paper, please feel convenient to let me know through email: [wen.lai@tum.de](mailto:wen.lai@tum.de) 

   

