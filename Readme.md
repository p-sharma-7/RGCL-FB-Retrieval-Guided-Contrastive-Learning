# RGCL: Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning
This is the official repo for the paper: Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning. 

- The link to the paper is [here](https://aclanthology.org/2024.acl-long.291.pdf).
- The link to the project page is [here](https://rgclmm.github.io/).


## Updates
- [29/10/2024] 🔥🔥Initial Release of the code base.
- [10/08/2024] 🔥Our paper appears at ACL2024 Main.

Useage
--------------------
## Create Env
```shell
conda create -n RGCL python=3.10 -y
conda activate RGCL
```

Install pytorch
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

Install FAISS
```
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021 blas=1.0=mkl -y
```

```
pip install -r requirements.txt
```


Dataset Preparation 
--------------------
#### Image data
Dump image into `./data/image/dataset_name/All` folder.
For example: `./data/image/FB/All/12345.png`, `./data/image/HarMeme/All`, `./data/image/Propaganda/All`, etc..
#### Annotation data
Dump `jsonl` annotation file into `./data/gt/dataset_name` folder.

#### Generate CLIP Embedding
We generate CLIP embedding prior to training to avoid repeated generation during training.

```shell
python3 src/utils/generate_CLIP_embedding_HF.py --dataset "FB"
python3 src/utils/generate_CLIP_embedding_HF.py --dataset "HarMeme"

```

#### Generate ALIGN Embedding
```shell
python3 src/utils/generate_ALIGN_embedding_HF.py --dataset "FB"
python3 src/utils/generate_ALIGN_embedding_HF.py --dataset "HarMeme"

```

#### Generate Sparse Retrieval Index
##### Generate VinVL BoundingBox Prediction (Optional)
ToDo


Training and Evalution 
--------------------
```
bash scripts\experiments.sh
```

## Common Issues
If you experience stuck in training, it might be due to the `faiss` installation. 

## Citation
If our work helped your research, please kindly cite our paper
```
@inproceedings{RGCL2024Mei,
    title = "Improving Hateful Meme Detection through Retrieval-Guided Contrastive Learning",
    author = "Mei, Jingbiao  and
      Chen, Jinghong  and
      Lin, Weizhe  and
      Byrne, Bill  and
      Tomalin, Marcus",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.291",
    doi = "10.18653/v1/2024.acl-long.291",
    pages = "5333--5347"
}
```
