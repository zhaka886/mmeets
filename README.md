### I Mean I Am a Mouse:Mmeets for Bilingual Multimodal Meme Sarcasm Classification from Large Language Models

This is the **official repository** of the "I Mean I Am a Mouse:Mmeets for Bilingual Multimodal Meme Sarcasm Classification from Large Language Models" (**Mmeets**).

## Overview

### Abstract

Multimodal image-text memes are widely used on social networks. Given their deep cultural and regional influences, understanding and analyzing these memes presents a range of challenges for high precision sentiment analysis, social network analysis, and understanding different user communities. However, existing analyses of multimodal memes have primarily focused on English-speaking communities and preliminary sentiment tasks such as harmful memes detection. In this paper, we focus on high-precision sentiment analysis in various contexts — the classification of sarcasm. Firstly, a new dataset for sarcasm classification of multimodal memes in both Chinese and English is introduced. Secondly, a framework named Mmeets is proposed, which utilizes Large Language Models (LLMs) for causal reasoning to achieve multimodal fusion and lightweight fine-tuning.Mmeets leverages a pre-trained AltCLIP vision-language model and the Abductive Reasoning with LLMs to effectively capture the multimodal semantic content of the memes.Our findings suggest that our Mmeets method outperforms state-of-the-art approaches in the task of sarcasm classification.



xx

![](asset/mmeets_architecture.png "Architecture of the method")



<details>
<summary><h2>Getting Started</h2></summary>
We recommend using the [**Anaconda**](https://www.anaconda.com/) package manager to avoid dependency/reproducibility
problems.
For Linux systems, you can find a conda installation
guide [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html).

### Installation

1. Clone the repository

```sh
git clone https://github.com/zhaka886/mmeets.git
```

2. Install Python dependencies

Navigate to the root folder of the repository and use the command:
```sh
conda config --add channels conda-forge
conda create -n issues -y python=3.9.16
conda activate issues
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install --file requirements.txt
```

3. Log in to your WandB account
```sh
wandb login
```

## Datasets
We created the first categorical dataset of mockery and self-mockery in both Chinese and English.

- $BSMM_{Chinese}$ -Contains **1977** memes
- $BSMM_{English}$- Contains **2306** memes


### Data Preparation
Download the files in the [release](https://github.com/miccunifi/ISSUES/releases/tag/latest) and place the `resources` folder in the root folder:

<pre>
project_base_path
└─── <b>resources</b>
  ...
└─── src
  | main.py
  | datasets.py
  | engine.py
  ...

...
</pre>

Ensure the $BSMM_{Chinese}$ and $BSMM_{Chinese}$ datasets match the following structure:

<pre>
project_base_path
└─── resources
  └─── datasets
    └─── zh


      └─── <b>img
          | deprecating image1.png
          | deprecating image2.png
          | deprecating image3.png
          ....</b>
    
      └─── labels
          | zh_finally.csv
    
    └─── en
      └─── <b>img
          | deprecating image1.png
          | deprecating image1.png
          | deprecating image1.png
          ....</b>
        
      └─── labels
          | en_finally.csv
  ...

└─── src
  | main.py
  | datasets.py
  | engine.py
  ...

...
</pre>

</details>
