
# Sematic Role Labeling with syntactic K-pruning algorithm

This repository contains code for training and using the SRL model described in:
[Syntax for Semantic Role Labeling, To Be, Or Not To Be](http://aclweb.org/anthology/P18-1192)

If you use our code, please cite our paper as follows:  
```
@inproceedings{he2018syntax,
        title={Syntax for Semantic Role Labeling, To Be, Or Not To Be},
        author={He, Shexia and Li, Zuchao and Zhao, Hai and Bai Hongxiao},
        booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (ACL 2018)},
        year={2018}
}
```

Semantic role labeling (SRL) is dedicated to recognizing the predicate-argument structure of a sentence. Previous studies have shown syntactic information has a remarkable contribution to SRL performance. However, such perception was challenged by a few recent neural SRL models which give impressive performance without a syntactic backbone. This paper intends to quantify the importance of syntactic information to dependency SRL in deep learning framework. We propose an enhanced argument labeling model companying with an extended korder argument pruning algorithm for effectively exploiting syntactic information. Our model achieves state-of-the-art results on the CoNLL-2008, 2009 benchmarks for both English and Chinese, showing the quantitative significance of syntax to neural SRL together with a thorough empirical survey over existing models. 

The framework of the proposed model:
<center style="padding: 40px"><img width="70%" src="http://gitlab.leisure-x.com/lzc/COLING2018-Code-Seq2seq4dp/raw/dev/screenshots/seq2seq_model.png" /></center>
 
## Requirements

```bash
pip install -r requirements.txt
```
this project is tested on pytorch 0.3.1, the other version may need some modification.

## Quickstart

### Step 1: Convert the dependency parsing dataset

```bash
python data/scripts/make_dp_dataset.py
```


### Step 2: Preprocessing the data

```bash
python preprocess.py -train_src data/input/dp/src_ptb_sd_train.input -train_tgt data/input/dp/tgt_ptb_sd_train.input -valid_src data/input/dp/src_ptb_sd_dev.input -valid_tgt data/input/dp/tgt_ptb_sd_dev.input -save_data data/temp/dp/dp
```
We will be working with some example data in `data/` folder.

The data consists of parallel source (`src`) and target (`tgt`) data containing one sentence per line with tokens separated by a space:

* `src-train.txt`
* `tgt-train.txt`
* `src-val.txt`
* `tgt-val.txt`

Validation files are required and used to evaluate the convergence of the training. It usually contains no more than 5000 sentences.


After running the preprocessing, the following files are generated:

* `dp.train.pt`: serialized PyTorch file containing training data
* `dp.valid.pt`: serialized PyTorch file containing validation data
* `dp.vocab.pt`: serialized PyTorch file containing vocabulary data


Internally the system never touches the words themselves, but uses these indices.

### Step 2: Make the pretrain embedding

```bash
python tools/embeddings_to_torch.py -emb_file_enc data/pretrain/glove.6B.100d.txt -dict_file data/temp/dp/dp.vocab.pt -output_file data/temp/dp/en_embeddings -type GloVe
```


### Step 3: Train the model

```bash
python train.py -save_model data/model/dp/dp -batch_size 64 -enc_layers 4 -dec_layers 2 -rnn_size 800 -word_vec_size 100 -feat_vec_size 100 -pre_word_vecs_enc data/temp/dp/en_embeddings.enc.pt -data data/temp/dp/dp -encoder_type brnn -gpuid 0 -position_encoding -bridge -global_attention mlp -optim adam -learning_rate 0.001 -tensorboard -tensorboard_log_dir logs -elmo -elmo_size 500 -elmo_options data/pretrain/elmo_2x4096_512_2048cnn_2xhighway_options.json -elmo_weight data/pretrain/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5 -subword_elmo -subword_elmo_size 500 -subword_elmo_options data/pretrain/subword_elmo_options.json -subword_weight data/pretrain/en.wiki.bpe.op10000.d50.w2v.txt -subword_spm_model data/pretrain/en.wiki.bpe.op10000.model
```

### Step 3: Translate

```bash
python translate.py -model data/model/dp/xxx.pt -src data/input/dp/src_ptb_sd_test.input -tgt data/input/dp/tgt_ptb_sd_test.input -output data/results/dp/tgt_ptb_sd_test.pred -replace_unk -verbose -gpu 0 -beam_size 64 -constraint_length 8 -alpha_c 0.8 -alpha_p 0.8
```

Now you have a model which you can use to predict on new data. We do this by running beam search where `constraint_length`, `alpha_c`, `alpha_p` are parameters used in tree constraints.

# Notes
You can refer to our paper for more details. Thank you!



