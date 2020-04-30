# ConversationQueryRewriter

## Dependencies
We require python >= 3.5, pytorch, transformer 2.2.0, and a handful of other supporting libraries. To install dependencies use
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

You could simply use:
```
mkdir data
cd data
wget https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json
https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv
```

### MS MARCO Conversatioanl Search Corpus
MS MARCO Conversational Search corpus is used to genearte weak supervison data and it can be obtained from [here](https://github.com/microsoft/MSMARCO-Conversational-Search).

You could simply use:
```
mkdir data/ms_marco
cd data/ms_marco
wget https://msmarco.blob.core.windows.net/conversationalsearch/ann_session_dev.tar.gz
tar xvzf ann_session_dev.tar.gz
```

### Preprocess TREC CAsT 2019 Data
Convert format and split fold for TREC CAsT 2019 data:
```
python cqr/preprocess.py
```


## Generate Weak Supervision Data
### Filter MS MARCO Conversatioanl Search Corpus
First, we need to filter the MS MARCO Conversatioanl Search corpus. This can be done as follows:
```
python cqr/weak_supervision/filter.py --input_file data/ms_marco/marco_ann_session.dev.all.tsv --output_file data/ms_marco/marco_ann_session.dev.all.filtered.tsv
```

### Rule-based Method
You could use "cqr/weak\_supervision/rule\_based/apply\_rules.py" to generate weak supervision using Rule-based Method. For example:
```
mkdir data/weak_supervision_data
python cqr/weak_supervision/rule_based/apply_rules.py --input_file data/ms_marco/marco_ann_session.dev.all.filtered.tsv --output_file data/weak_supervision_data/rule-based.jsonl --use_coreference --use_omission
```

### Model-based Method
Convert TREC CAsT data into training data for query simplify model:
```
python cqr/weak_supervision/model_based/generate_training_data.py
```

Then train query simplify models:
```
nohup python -u cqr/run_training.py --output_dir=models/query-simplifier-bs2-e4 --train_file data/training_data_for_query_simplifier.jsonl --cross_validate --model_name_or_path=gpt2  --per_gpu_train_batch_size=2 --per_gpu_eval_batch_size=2 --num_train_epochs=4 --save_steps=-1 &> run_train_query_simplifier.log &
```
Since we use the evaluation data of TREC CAsT 2019 to train our query simplifier, we do it in a way of k-fold cross validation as we mentioned in the paper. Therefore, this command results in 5 models from different training folds.

Apply query simplify model on MS MARCO Conversatioanl Search data and generate weak supervision data for query rewriting model. Please note that this could be slow. For example:
```
python weak_supervision/model_based/generate_weak_supervision_data.py --model_path models/query-simplifier-bs2-e4 --input_file data/ms_marco/marco_ann_session.dev.all.filtered.tsv --output_file data/weak_supervision_data/model-based.jsonl
```
This would generate 5 different version of weak supervision data (model-based.json.0, model-based.json.1, ..., model-based.json.4), each coming from one model.


## Train
Our models can be trained by:
```
python cqr/run_training.py --model_name_or_path <pretrained_model_path> --train_file <input_json_file> --output_dir <output_model_path>
```

### Cross Validation
For example:
```
nohup python -u cqr/run_training.py --output_dir=models/query-rewriter-cv-bs2-e4 --train_file data/eval_topics.jsonl --cross_validate --model_name_or_path=gpt2 --per_gpu_train_batch_size=2 --num_train_epochs=4 --save_steps=-1 &> run_train_query_rewriter_cv.log &
```
You would get 5 models (e.g. models/model-medium-cv-s2-e4-\<i\> where i = 0..4) using the default setting (NUM\_FOLD=5).

### Rule-based
For example:
```
nohup python -u cqr/run_training.py --output_dir=models/query-rewriter-rule-based-bs2-e1 --train_file data/weak_supervision_data/rule-based.jsonl --model_name_or_path=gpt2 --per_gpu_train_batch_size=2 --save_steps=-1 &> run_train_query_rewriter_rule_based.log &
```

### Model-based
Five query rewriting models could be trained each using data from one query simplifier. For example:
```
nohup python -u cqr/run_training.py --output_dir=models/query-rewriter-model-based-bs2-e1-<i> --train_file data/weak_supervision_data/model-based.jsonl.<i> --model_name_or_path=gpt2 --per_gpu_train_batch_size=2 --save_steps=-1 &> run_train_query_rewriter_model_based_<i>.log &
```
where i = 0, 1, ..., 4.

### Rule-based + CV
You could change the parameter 'model_name_or_path' in the Cross Validation example to the directory of the trained rule-based model. For example:
```
nohup python -u cqr/run_training.py --output_dir=models/query-rewriter-rule-based-bs2-e1-cv-e4 --train_file data/eval_topics.jsonl --cross_validate --model_name_or_path=models/query-rewriter-rule-based-bs2-e1 --per_gpu_train_batch_size=2 --save_steps=-1 &> run_train_query_rewriter_rule_based_plus_cv.log &
```

### Model-based + CV
Don't forget to use '--init_from_multiple_models' in this setting. For example:
```
nohup python -u cqr/run_training.py --output_dir=models/query-rewriter-model-based-bs2-e1-cv-e4 --train_file data/eval_topics.jsonl --cross_validate --init_from_multiple_models --model_name_or_path=models/query-rewriter-model-based-bs2-e1 --per_gpu_train_batch_size=2 --save_steps=-1 &> run_train_query_rewriter_model_based_plus_cv.log &
```


## Download Trained Models
Two trained models can be downloaded with the following link: [Model-based+CV-0](https://thunlp.s3-us-west-1.amazonaws.com/Model-based%2BCV-0.zip) and [Rule-based+CV-1](https://thunlp.s3-us-west-1.amazonaws.com/Rule-based%2BCV-1.zip).


## Inference
You could use the following command to do inference:
```
python cqr/run_prediction.py --model_path <model_path> --input_file <input_json_file> --output_file <output_json_file>
```

### Cross Validation
For example:
```
python cqr/run_prediction.py --model_path=models/query-rewriter-cv-bs2-e4 --cross_validate --input_file=data/eval_topics.jsonl --output_file=cv-predictions.jsonl
```

### Rule-based
For example:
```
python cqr/run_prediction.py --model_path=models/query-rewriter-rule-based-bs2-e1 --input_file=data/eval_topics.jsonl --output_file=rule-based-predictions.jsonl
```

### Model-based
For example:
```
python cqr/run_prediction.py --model_path=models/query-rewriter-model-based-bs2-e1 --cross_validate --input_file=data/eval_topics.jsonl --output_file=model-based-predictions.jsonl
```

### Rule-based + CV
For example:
```
python cqr/run_prediction.py --model_path=models/query-rewriter-rule-based-bs2-e1-cv-e4 --cross_validate --input_file=data/eval_topics.jsonl --output_file=rule-based-plus-cv-predictions.jsonl
```

### Model-based + CV
For example:
```
python cqr/run_prediction.py --model_path=models/query-rewriter-model-based-bs2-e1-cv-e4 --cross_validate --input_file=data/eval_topics.jsonl --output_file=model-based-plus-cv-predictions.jsonl
```


## Contact
If you have any question or suggestion, please send email to alphaf52@gmail.com.

