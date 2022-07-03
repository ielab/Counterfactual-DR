# Counterfactual-DR
The official repository for [Implicit Feedback for Dense Passage Retrieval: A Counterfactual Approach](https://arxiv.org/pdf/2204.00718.pdf), Shengyao Zhuang, Hang Li and Guido Zuccon, SIGIR2022


#### Toy example
To make it easier for readers to understand our approach, we make a toy example to show how counterfactual DR works. Simply run: `python toy_exmample.py`

## Download DR indexes
> Note, this repo is tested with `torch==1.8.1` `faiss-cpu==1.7.2` `transformers==4.20.1`

We use [pyserini](https://github.com/castorini/pyserini) toolkit to download DR models, indexes and evaluation.
```
pip install pyserini==0.17.0
```
To download DR indexes, run the following pyserini command:
```
# Download TCT-ColBERTv2 faiss index
python -c "from pyserini.search.lucene import LuceneSearcher; LuceneSearcher.from_prebuilt_index('msmarco-passage-tct_colbert-v2-hnp-bf')"

# Download ANCE faiss index
python -c "from pyserini.search.lucene import LuceneSearcher; LuceneSearcher.from_prebuilt_index('msmarco-passage-ance-bf')"
```
The index will be downloaded in `~/.cache/pyserini/indexes/`


## Run Experiments

Set the model and dataset variables in command line:
```
YEAR=2019
MODEL_NAME=TCTv2
MODEL_PATH=castorini/tct_colbert-v2-hnp-msmarco
INDEX_PATH=~/.cache/pyserini/indexes/dindex-msmarco-passage-tct_colbert-v2-hnp-bf-20210608-5f341b.c3c3fc3a288bcdf61708d4bba4bc79ff/index
```
We use TCT-colbert v2, DL2019 dataset as an example for the experiments. You can easily run other experiments by changing `MODEL_NAME=ANCE`, `YEAR=2020` and corresponding index path.

### Logged query experiments
#### Original DR results
The following commands will run the original TCT-ColBERTv2 method (single stage retrieval without any feedback).
```
python3 run_seen_queries.py \
--model_name ${MODEL_NAME} \
--output_path runs/${MODEL_NAME}/${MODEL_NAME}_dl${YEAR}.txt \
--query_path data/${YEAR}queries-pass.tsv \
--qrel_path data/${YEAR}qrels-pass.txt \
--model_path ${MODEL_PATH} \
--index_path ${INDEX_PATH}

```

Evaluate the original TCT-ColBERTv2 run file:
```
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 -m recall.1000 -m map data/2019qrels-pass.txt \
runs/TCTv2/TCTv2_dl2019.txt

Results:
map     all     0.4469
recall_1000     all     0.8261
ndcg_cut_10     all     0.7204
```
---
#### DR with Pseudo-relevance feedback (TCT-ColBERTv2 + VPRF) 
The following commands will run the TCT-ColBERTv2 with vector PRF method.
```
python3 run_seen_queries.py \
--model_name ${MODEL_NAME} \
--output_path runs/${MODEL_NAME}/${MODEL_NAME}_dl${YEAR}_vprf.txt \
--query_path data/${YEAR}queries-pass.tsv \
--qrel_path data/${YEAR}qrels-pass.txt \
--model_path ${MODEL_PATH} \
--index_path ${INDEX_PATH} \
--feedback
```

Evaluate the TCT-ColBERTv2 + VPRF run file:

```
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 -m recall.1000 -m map data/2019qrels-pass.txt \
runs/TCTv2/TCTv2_dl2019_vprf.txt

Results:
map     all     0.4797
recall_1000     all     0.8633
ndcg_cut_10     all     0.6982
```
---
#### DR with Implicit feedback
The experiments in this section are conducted with synthetic user click models. 

The following commands will run the TCT-ColBERTv2 with Rocchio algorithme under perfect and unbiased click setting, i.e., 
users will exam all the documents in the rankings and click every relevant documents. 
Same experiment for noise click setting can be done by setting `--click_model noise`.
```
python3 run_seen_queries.py \
--model_name ${MODEL_NAME} \
--output_path runs/${MODEL_NAME}/${MODEL_NAME}_dl${YEAR}_rocchio_perfect_unbiased.txt \
--query_path data/${YEAR}queries-pass.tsv \
--qrel_path data/${YEAR}qrels-pass.txt \
--model_path ${MODEL_PATH} \
--index_path ${INDEX_PATH} \
--feedback \
--propensity 0 \
--click_model perfect
```

Evaluate the TCT-ColBERTv2 + Rocchio run file:

```
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 -m recall.1000 -m map data/2019qrels-pass.txt \
runs/TCTv2/TCTv2_dl2019_rocchio_perfect_unbiased.txt

Results:
map     all     0.5778
recall_1000     all     0.9001
ndcg_cut_10     all     0.7963
```

---
The following commands will run the TCT-ColBERTv2 with Rocchio algorithme under perfect and biased click setting, i.e.,
higher ranked documents have higher probability to be examined by users and users will click every relevant documents they examined.
Same experiment for noise click setting can be done by setting `--click_model noise`.
```
python3 run_seen_queries.py \
--model_name ${MODEL_NAME} \
--output_path runs/${MODEL_NAME}/${MODEL_NAME}_dl${YEAR}_rocchio_perfect_biased.txt \
--query_path data/${YEAR}queries-pass.tsv \
--qrel_path data/${YEAR}qrels-pass.txt \
--model_path ${MODEL_PATH} \
--index_path ${INDEX_PATH} \
--feedback \
--propensity 1 \
--click_model perfect
```

Evaluate the TCT-ColBERTv2 + Rocchio with biased user run file:

```
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 -m recall.1000 -m map data/2019qrels-pass.txt \
runs/TCTv2/TCTv2_dl2019_rocchio_perfect_biased.txt

Results:
map     all     0.5592
recall_1000     all     0.8948
ndcg_cut_10     all     0.7883
```
We observed effectiveness decreased across all the metrics due to the position bias of user behaviour.

---
The following commands will run the TCT-ColBERTv2 with CoRocchio algorithme under perfect and biased click setting, i.e.,
higher ranked documents have higher probability to be examined by users and users will click every relevant documents they examined.
Same experiment for noise click setting can be done by setting `--click_model noise`.
```
python3 run_seen_queries.py \
--model_name ${MODEL_NAME} \
--output_path runs/${MODEL_NAME}/${MODEL_NAME}_dl${YEAR}_corocchio_perfect_biased.txt \
--query_path data/${YEAR}queries-pass.tsv \
--qrel_path data/${YEAR}qrels-pass.txt \
--model_path ${MODEL_PATH} \
--index_path ${INDEX_PATH} \
--feedback \
--propensity 1 \
--click_model perfect \
--debias
```

Evaluate the TCT-ColBERTv2 + CoRocchio with biased user run file:

```
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 -m recall.1000 -m map data/2019qrels-pass.txt \
runs/TCTv2/TCTv2_dl2019_corocchio_perfect_biased.txt

Results:
map     all     0.5775
recall_1000     all     0.9004
ndcg_cut_10     all     0.7963
```
The scores are very close to the perfect and unbiased setting, meaning that CoRocchio can debias users' position bias.

---

### Unseen query experiments
#### Dataset Augmentation with Synthetic Query Generation
TREC DL datasets have only a small set of judged queries, and they are unrelated to each other, 
withholding a subset of TREC DL queries is as unseen queries is not possible. 
Thus we adapt the docTquery-T5 method to augment the current TREC DL
datasets with unseen, but related queries with associated relevance judgements.

To do that, first download the MS MARCO passage collection file `collection.tsv` from this [link](https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz) and put it in the `./data` folder.

Then run the following command to generate a new query and qrel file. We take an example of augmenting DL2019 dataset.
```
python3 data_augmentation.py \
--collection_file ./data/collection.tsv \
--qrel_file ./data/2019qrels-pass.txt \
--query_file ./data/2019queries-pass.tsv \
--query_out ./data/2019queries-pass-new.tsv \
--qrel_out ./data/2019qerl-pass-new.txt 
```
These new queries are relevance are generated based on the original queries' highly relevant passages (rel > 2). 
Then you just need to split the `2019queries-pass-new.tsv` into a train and a test subsets. We provided our generated 
new query and qrel files in the data folder, and randomly split it into a train (80%) and test (20%) file. Note in our experiments
we treat train queries are logged queries and test quereis as unseen queries.


---
#### Run unseen query experiments
The commands of running unseen query experiments are very similar to the logged query experiments except you just need to set 
`--train_query_path` and `--test_query_path` to the train and test query files. 

We take the original TCT-ColBERTv2 experiments as an example, the other experiments are same as [logged query experiments](###logged-query-experiments)
The following commands will run the original TCT-ColBERTv2 method (single stage retrieval without any feedback).
```
python3 run_unseen_queries.py \
--model_name ${MODEL_NAME} \
--output_path runs/${MODEL_NAME}_unseen/${MODEL_NAME}_dl${YEAR}.txt \
--train_query_path data/${YEAR}queries-pass-new-train.tsv \
--test_query_path data/${YEAR}queries-pass-new-test.tsv \
--qrel_path data/${YEAR}qrels-pass-new.txt \
--model_path ${MODEL_PATH} \
--index_path ${INDEX_PATH}
```

Evaluate the original TCT-ColBERTv2 run file:
```
python -m pyserini.eval.trec_eval -c -l 2 -m ndcg_cut.10 -m recall.1000 -m map data/2019qrels-pass-new.txt \
runs/TCTv2_unseen/TCTv2_dl2019.txt

Results:
map     all     0.2417
recall_1000     all     0.5211
ndcg_cut_10     all     0.4203
```
