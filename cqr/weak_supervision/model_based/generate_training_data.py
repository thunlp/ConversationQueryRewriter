
import json
import copy

from cqr.utils import NUM_FOLD

with open('data/evaluation_topics_v1.0.json', 'r') as fin:
    raw_data = json.load(fin)

with open('data/evaluation_topics_annotated_resolved_v1.0.tsv', 'r') as fin:
    annonated_lines = fin.readlines()

all_annonated = {}
for line in annonated_lines:
    splitted = line.split('\t')
    topic_query = splitted[0]
    query = splitted[1].strip()
    topic_id = topic_query.split('_')[0]
    query_id = topic_query.split('_')[1]
    if topic_id not in all_annonated:
        all_annonated[topic_id] = {}
    all_annonated[topic_id][query_id] = query

topic_number_dict = {}
data = []
for group in raw_data:
    topic_number, description, turn, title = str(group['number']), group.get('description', ''), group['turn'], group.get('title', '')
    query_rewrites = []
    original_queries = []
    for query in turn:
        query_number, original_query = str(query['number']), query['raw_utterance']
        query_rewrites.append(all_annonated[topic_number][query_number])
        original_queries.append(original_query)
        if query_number == '1':
          continue
        record = {}
        record['topic_number'] = topic_number
        record['query_number'] = query_number
        record['description'] = description
        record['title'] = title
        record['input'] = original_queries[:-1] + [query_rewrites[-1]]
        record['target'] = original_query
        if not topic_number in topic_number_dict:
            topic_number_dict[topic_number] = len(topic_number_dict)
        data.append(record)

with open('data/training_data_for_query_simplifier.jsonl', 'w') as fout:
    for item in data:
        json_str = json.dumps(item, ensure_ascii=False)
        fout.write(json_str + '\n')

# Split eval data into K-fold
topic_per_fold = len(topic_number_dict) // NUM_FOLD
for i in range(NUM_FOLD):
    with open('data/training_data_for_query_simplifier.jsonl.%d' % i, 'w') as fout:
        for item in data:
            idx = topic_number_dict[item['topic_number']]
            if idx // topic_per_fold == i:
                json_str = json.dumps(item, ensure_ascii=False)
                fout.write(json_str + '\n')

