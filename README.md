## ConversationQueryRewriter

### Dependencies
To install the dependencies for ConversationQueryRewriter, use
```
pip install -r requirements.txt
```

### Download Trained Models
Two trained models can be downloaded with the following link: [Model-based+CV-0](https://thunlp.s3-us-west-1.amazonaws.com/Model-based%2BCV-0.zip) and [Rule-based+CV-1](https://thunlp.s3-us-west-1.amazonaws.com/Rule-based%2BCV-1.zip).

### Inference
You could use the following command to do inference:
```
python run_prediction.py --model_path <model_path> --input_file data/eval_topics.jsonl --output_file <output_json_file>
```

### Contact
If you have any question or suggestion, please send email to alphaf52@gmail.com.
