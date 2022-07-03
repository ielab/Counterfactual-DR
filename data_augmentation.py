import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from argparse import ArgumentParser
import collections
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_collection(collection_path):
    collection = {}
    with open(collection_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="loading collection...."):
            docid, text = line.strip().split("\t")
            collection[int(docid)] = text
    return collection


def load_query_file(path):
    queries = []
    qids = []
    for line in open(path):
        [qid, query] = line.strip().split('\t')
        queries.append(query)
        qids.append(qid)
    return qids, queries


def load_qrels(path: str):
    qrels = collections.defaultdict(dict)
    for line in open(path):
        line = ' '.join(line.split())
        query_id, _, doc_id, relevance = line.split(' ')
        qrels[query_id][int(doc_id)] = int(relevance)
    return qrels


def main(args):
    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco', cache_dir="cache")
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco', cache_dir="cache")
    model.to(device)

    qrels = load_qrels(args.qrel_file)
    collection = load_collection(args.collection_file)
    qids, queries = load_query_file(args.query_file)

    new_queries = []
    new_qrels = {}
    query_set = set([])

    # add original data
    for qid, query in zip(qids, queries):
        new_queries.append((qid, query.lower()))
        query_set.add(query)
        new_qrels[qid] = qrels[qid]

    # new data
    for qid in tqdm(qrels.keys()):
        i = 0
        for docid in qrels[qid].keys():
            new_qid = f"{qid}-{i}"
            rel = qrels[qid][docid]
            if rel >= 2:
                doc_text = collection[docid]
                input_ids = tokenizer.encode(doc_text, return_tensors='pt').to(device)
                while True:  # make sure no repeated query
                    outputs = model.generate(
                        input_ids=input_ids,
                        max_length=64,
                        do_sample=True,
                        top_k=10,
                        num_return_sequences=1)
                    new_query = tokenizer.decode(outputs[0], skip_special_tokens=True).lower()
                    if new_query in query_set:
                        continue
                    else:
                        query_set.add(new_query)
                        new_queries.append((new_qid, new_query))
                        new_qrels[new_qid] = qrels[qid]
                        i += 1
                        break
    with open(args.query_out, 'w') as qf:
        for qid, query in new_queries:
            qf.write(f"{qid}\t{query}\n")

    with open(args.qrel_out, 'w') as rf:
        for qid in new_qrels.keys():
            for docid, rel in new_qrels[qid].items():
                rf.write(f"{qid} Q0 {docid} {rel}\n")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--collection_file", required=True)
    parser.add_argument("--qrel_file", required=True)
    parser.add_argument("--query_file", required=True)
    parser.add_argument("--query_out", required=True)
    parser.add_argument("--qrel_out", required=True)
    args = parser.parse_args()

    main(args)

