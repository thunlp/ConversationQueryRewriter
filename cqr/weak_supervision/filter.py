
import argparse
import json

from cqr.utils import QUESTION_WORD_LIST, OTHER_WORD_LIST

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input tsv file")
    parser.add_argument("--output_file", type=str, required=True, help="Output file")
    args = parser.parse_args()

    with open(args.input_file, 'r') as fin, open(args.output_file, 'w') as fout:
        ln = 0
        wn = 0
        for line in fin:
            splitted = line[:-1].split('\t')
            sid = splitted[0]
            queries = splitted[1:]
            last = 0
            modified_queries = []
            for i, query in enumerate(queries):
                last = i
                is_question = False
                is_others = False
                if any([query.lower().startswith(word) for word in QUESTION_WORD_LIST]):
                    is_question = True
                if any([query.lower().startswith(word) for word in OTHER_WORD_LIST]):
                    is_others = True
                if (not is_question) and (not is_others):
                    break
                modified_queries.append(query[0].upper() + query[1:] + ("?" if is_question == True else "."))
            if last > 1:
                fout.write(sid + "\t" + "\t".join(modified_queries) + "\n")
                wn += 1
            ln += 1
        
    print("total: %d, after filtering: %d" % (ln, wn))

