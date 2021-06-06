# CFER

Paper: Coarse-to-Fine Entity Representations for Document-level Relation Extraction ([URL](https://arxiv.org/pdf/2012.02507.pdf))

## Requirements

* Pytorch (1.6.0)
* numpy (1.19.4)
* tqdm (4.50.2)
* transformers (4.6.1)
* spacy (2.3.2)

## Downloading Data

You can download the required data from [here](https://drive.google.com/file/d/1kXin2BIiuE5BDrB3UXTdMQYHOZgctyp8/view?usp=sharing). 
After you download `data.zip`, unzip it and put it to the root directory of this project. 
Besides the datasets, we also provide some preprocessing results (see `data/adj/` and `data/path`) for saving time.

## Specifying Configuration

Before training, you can edit `code/config.py` to specify the configurations, including filepath information, and hyper-parameters. 
If you want to reproduce our results reported in the paper, you can use the reported hyper-parameters, and keep other hyper-parameters unchanged. 

## Training and Evaluating

1. Change the working directory to the root directory of this project. 
2. Run `python3 code/main.py`.

## Citation

If you use this code for your research, please kindly cite our paper: 

```
@article{dai2020cfer,
        author    = {Damai Dai and
                    Jing Ren and
                    Shuang Zeng and
                    Baobao Chang and
                    Zhifang Sui},
        title     = {Coarse-to-Fine Entity Representations for Document-level Relation Extraction},
        journal   = {CoRR},
        volume    = {abs/2012.02507},
        year      = {2020},
        url       = {https://arxiv.org/abs/2012.02507}
}
```

## Contact

This project is supported by Jing Ren. 
If you have any problems, please contact us via the following e-mail addresses. 

Jing Ren: rjj@pku.edu.cn

Damai Dai: daidamai@pku.edu.cn
