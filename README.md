# ConversationQueryRewriter

This repo contains data and code for SIGIR 2020 paper ["Few-Shot Generative Conversational Query Rewriting"](https://arxiv.org/abs/2006.05009).

- [ConversationQueryRewriter](#conversationqueryrewriter)
  * [Dependencies](#dependencies)
  * [Data](#data)
    + [TREC CAsT 2019 Data](#trec-cast-2019-data)
    + [MS MARCO Conversatioanl Search Corpus](#ms-marco-conversatioanl-search-corpus)
    + [Preprocess TREC CAsT 2019 Data](#preprocess-trec-cast-2019-data)
  * [Generate Weak Supervision Data](#generate-weak-supervision-data)
    + [Filter MS MARCO Conversatioanl Search Corpus](#filter-ms-marco-conversatioanl-search-corpus)
    + [Rule-based Method](#rule-based-method)
    + [Self-learn Method](#self-learn-method)
  * [Train](#train)
    + [Cross-validation on TREC CAsT 2019](#cross-validation-on-trec-cast-2019)
    + [Rule-based](#rule-based)
    + [Self-learn](#self-learn)
    + [Rule-based + CV](#rule-based---cv)
    + [Self-learn + CV](#self-learn---cv)
  * [Download Trained Models](#download-trained-models)
  * [Inference](#inference)
    + [Cross-validation](#cross-validation)
    + [Rule-based](#rule-based-1)
    + [Self-learn](#self-learn-1)
    + [Rule-based + CV](#rule-based---cv-1)
    + [Self-learn + CV](#self-learn---cv-1)
  * [Results](#results)
  * [Contact](#contact)

## Dependencies

We require python >= 3.6, pytorch, transformers 2.3.0, and a handful of other supporting libraries. To install dependencies use

```
pip install -r requirements.txt
```

The spaCy model for English is needed and can be fetched with:

```
python -m spacy download en_core_web_sm
```

The easiest way to run this code is to use:

```
export PYTHONPATH=${PYTHONPATH}:`pwd`
```

## Data

By default, we expect source and preprocessed data to be stored in "./data".

### TREC CAsT 2019 Data

TREC CAsT 2019 data can be obtained from [here](https://github.com/daltonj/treccastweb).

Use the following commands to download the TREC CAsT 2019 data to the `data` folder:

```
cd data
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv
```

### MS MARCO Conversatioanl Search Corpus

MS MARCO Conversational Search corpus is used to genearte weak supervison data and it can be obtained from [here](https://github.com/microsoft/MSMARCO-Conversational-Search).

Use the following commands to get and unpack the Dev sessions:

```
mkdir data/ms_marco
cd data/ms_marco
wget https://msmarco.blob.core.windows.net/conversationalsearch/ann_session_dev.tar.gz
tar xvzf ann_session_dev.tar.gz
```

### Preprocess TREC CAsT 2019 Data

Run `cqr/preprocess.py` to convert and split folds for TREC CAsT 2019 data:

```
python cqr/preprocess.py
```

This will generate `eval_topics.jsonl` in `data` along with 5 folds `eval_topics.jsonl.x(x=0,1,2,3,4)` for cross-validation.


## Generate Weak Supervision Data

This section describes the steps to generate weak supervision data. You can directly use our generated files located in `data/weak_supervision_data`.

### Filter MS MARCO Conversatioanl Search Corpus

First, we need to filter the MS MARCO Conversatioanl Search corpus to keep only question-like queries. This can be done as follows:

```
python cqr/weak_supervision/filter.py --input_file data/ms_marco/marco_ann_session.dev.all.tsv --output_file data/ms_marco/marco_ann_session.dev.all.filtered.tsv
```

After filtering, you can choose either of the two methods (rule-based or self-learn) to generate weak supervision data for the GPT-2 query rewriter.

### Rule-based Method

The rule-based weak supervision data is available at `data/weak_supervision_data/rule-based.jsonl`.

To generate on your own, use the following commands to apply rules on the filtered MS MARCO dataset:

```
python cqr/weak_supervision/rule_based/apply_rules.py --input_file data/ms_marco/marco_ann_session.dev.all.filtered.tsv --output_file data/weak_supervision_data/rule-based.jsonl --use_coreference --use_omission
```

### Self-learn Method

The self-learned weak supervision data is available at `data/weak_supervision_data/self-learn.jsonl.x(x=0,1,2,3,4)`. 

To generate on your own, you first need to train a query simplifier using the TREC CAsT 2019 data. Convert the TREC CAsT data into training data for query simplifier using the following commands:

```
python cqr/weak_supervision/self_learn/generate_training_data.py
```

Then train query simplifying models:

```
nohup python -u cqr/run_training.py --output_dir=models/query-simplifier-bs2-e4 --train_file data/training_data_for_query_simplifier.jsonl --cross_validate --model_name_or_path=gpt2-medium  --per_gpu_train_batch_size=2 --per_gpu_eval_batch_size=2 --num_train_epochs=4 --save_steps=-1 &> run_train_query_simplifier.log &
```

Since we use the evaluation data of TREC CAsT 2019 to train our query simplifier, we do it in a way of k-fold cross validation as we mentioned in the paper. Therefore, this command results in 5 models from different training folds.

Then apply the models on the filtered MS MARCO Conversatioanl Search data and generate weak supervision data for query rewriting model. Please note that this could be slow. For example:

```
python weak_supervision/self_learn/generate_weak_supervision_data.py --model_path models/query-simplifier-bs2-e4 --input_file data/ms_marco/marco_ann_session.dev.all.filtered.tsv --output_file data/weak_supervision_data/self-learn.jsonl
```

This would generate 5 different version of weak supervision data (self-learn.json.0, self-learn.json.1, ..., self-learn.json.4), each coming from one model.

## Train

Our models can be trained by:

```
python cqr/run_training.py --model_name_or_path <pretrained_model_path> --train_file <input_json_file> --output_dir <output_model_path>
```

### Cross-validation on TREC CAsT 2019

For example:

```
nohup python -u cqr/run_training.py --output_dir=models/query-rewriter-cv-bs2-e4 --train_file data/eval_topics.jsonl --cross_validate --model_name_or_path=gpt2-medium --per_gpu_train_batch_size=2 --num_train_epochs=4 --save_steps=-1 &> run_train_query_rewriter_cv.log &
```

You would get 5 models (e.g. models/model-medium-cv-s2-e4-\<i\> where i = 0..4) using the default setting (NUM\_FOLD=5).

### Rule-based

For example:

```
nohup python -u cqr/run_training.py --output_dir=models/query-rewriter-rule-based-bs2-e1 --train_file data/weak_supervision_data/rule-based.jsonl --model_name_or_path=gpt2-medium --per_gpu_train_batch_size=2 --save_steps=-1 &> run_train_query_rewriter_rule_based.log &
```

### Self-learn

Recall that we have 5 sets of data generated by 5 different simplifiers trained with different training data. Query rewriting models could be trained each using data from one query simplifier. For example:

```
nohup python -u cqr/run_training.py --output_dir=models/query-rewriter-self-learn-bs2-e1-<i> --train_file data/weak_supervision_data/self-learn.jsonl.<i> --model_name_or_path=gpt2-medium --per_gpu_train_batch_size=2 --save_steps=-1 &> run_train_query_rewriter_self_learn_<i>.log &
```

where i = 0, 1, ..., 4.

### Rule-based + CV

Just change the parameter 'model_name_or_path' in the cross-validation example from a pretrained GPT-2 to the directory of the trained rule-based model. For example:

```
nohup python -u cqr/run_training.py --output_dir=models/query-rewriter-rule-based-bs2-e1-cv-e4 --train_file data/eval_topics.jsonl --cross_validate --model_name_or_path=models/query-rewriter-rule-based-bs2-e1 --per_gpu_train_batch_size=2 --save_steps=-1 &> run_train_query_rewriter_rule_based_plus_cv.log &
```

### Self-learn + CV

Don't forget to use '--init_from_multiple_models' in this setting to start from 5 models trained on 5 different sets of weak supervision data. For example:

```
nohup python -u cqr/run_training.py --output_dir=models/query-rewriter-self-learn-bs2-e1-cv-e4 --train_file data/eval_topics.jsonl --cross_validate --init_from_multiple_models --model_name_or_path=models/query-rewriter-self-learn-bs2-e1 --per_gpu_train_batch_size=2 --save_steps=-1 &> run_train_query_rewriter_self_learn_plus_cv.log &
```

## Download Trained Models

Two trained models can be downloaded with the following link: [Self-learn+CV-0](https://thunlp.s3-us-west-1.amazonaws.com/Self-Learn%2BCV-0.zip) and [Rule-based+CV-1](https://thunlp.s3-us-west-1.amazonaws.com/Rule-based%2BCV-1.zip). We made minor changes to the code, so the result may be slightly different from the paper.


## Inference

You can use the following command to do inference:

```
python cqr/run_prediction.py --model_path <model_path> --input_file <input_json_file> --output_file <output_json_file>
```

### Cross-validation

For example:

```
python cqr/run_prediction.py --model_path=models/query-rewriter-cv-bs2-e4 --cross_validate --input_file=data/eval_topics.jsonl --output_file=cv-predictions.jsonl
```

### Rule-based

For example:
```
python cqr/run_prediction.py --model_path=models/query-rewriter-rule-based-bs2-e1 --input_file=data/eval_topics.jsonl --output_file=rule-based-predictions.jsonl
```

### Self-learn

Recall that we have 5 models trained on 5 different sets of generated data. Thus we need the `--cross_validate` option to do the inference of their unseen parts:

```
python cqr/run_prediction.py --model_path=models/query-rewriter-model-based-bs2-e1 --cross_validate --input_file=data/eval_topics.jsonl --output_file=model-based-predictions.jsonl
```

### Rule-based + CV

For example:

```
python cqr/run_prediction.py --model_path=models/query-rewriter-rule-based-bs2-e1-cv-e4 --cross_validate --input_file=data/eval_topics.jsonl --output_file=rule-based-plus-cv-predictions.jsonl
```

### Self-learn + CV

For example:
```
python cqr/run_prediction.py --model_path=models/query-rewriter-model-based-bs2-e1-cv-e4 --cross_validate --input_file=data/eval_topics.jsonl --output_file=model-based-plus-cv-predictions.jsonl
```
## Results

Our BERT runs and GPT-2 rewrites are placed in the `results` folder.

|Methods|NDCG@3|Rewrites|BERT runs|
|-------|------|--------|---------|
|Original|0.304|N/A|`bert_base_run_raw.trec`|
|Oracle|0.544|N/A|`bert_base_run_oracle.trec`|
|Cross-validation|0.467|`query_rewriter_output_cv.jsonlines`|`bert_base_run_cv.trec`|
|Rule-based|0.437|`query_rewriter_output_rule_based.jsonlines`|`bert_base_run_rule_based.trec`|
|Self-learn|0.435|`query_rewriter_output_self_learn.jsonlines`|`bert_base_run_self_learn.trec`|
|Rule-based + CV|0.492|`query_rewriter_output_rule_based_cv.jsonlines`|`bert_base_run_rule_based_cv.trec`|
|Self-learn + CV|0.491|`query_rewriter_output_self_learn_cv.jsonlines`|`bert_base_run_self_learn_cv.trec`|

## Contact

If you have any question or suggestion, please send email to alphaf52@gmail.com or yus17@mails.tsinghua.edu.cn.

