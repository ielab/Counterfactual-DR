from modeling import AnceQueryEncoder, PositionBasedClickModel, TctColBertQueryEncoder
import faiss
from faiss import read_index
import numpy as np
from tqdm import tqdm
import os
import collections
from argparse import ArgumentParser


def write_run_file(qid, scores, docids, output_path, type='trec'):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))

    num_docs = len(docids)
    ms_lines = []
    trec_lines = []
    top_lines = str(qid)
    for k in range(num_docs):
        score = scores[k]
        docid = docids[k]
        ms_lines.append(str(qid) + "\t" + str(docid) + "\t" + str(k + 1) + "\n")
        trec_lines.append(str(qid) + " " + "Q0" + " " + str(docid) + " " + str(k + 1) + " " + str(score) + " " + "ielab" + "\n")
        top_lines += " " + str(docid)
    top_lines += "\n"

    if type == 'msmarco':
        with open(output_path, "a+") as f:
            f.writelines(ms_lines)
    if type == 'trec':
        with open(output_path, "a+") as f:
            f.writelines(trec_lines)


def read_query_file(path):
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


def get_batch_rels(qids, batch_docids, qrel):
    batch_rels = []
    for i, qid in enumerate(qids):
        rels = []
        for docid in batch_docids[i]:
            if docid not in qrel[qid]:
                rels.append(0)
            else:
                rels.append(qrel[qid][docid])
        batch_rels.append(rels)
    return np.array(batch_rels)


def rocchio_get_batch_prf_q_emb(emb_qs: np.ndarray = None,
                                batch_candidates: np.ndarray = None,
                                batch_clicks: np.ndarray = None,
                                propensity: np.ndarray = None,
                                beta=0.6,
                                alpha=0.4):
    new_emb_qs = []
    for index in range(len(emb_qs)):
        prf_candidates = batch_candidates[index]
        if propensity is not None:
            prf_candidates = prf_candidates / propensity[:, np.newaxis]

        if batch_clicks is not None:
            prf_candidates = prf_candidates[np.nonzero(batch_clicks[index])[0]]
            if len(prf_candidates) == 0:
                new_emb_qs.append(emb_qs[index])
                continue

        weighted_sum_doc_embs = beta * np.sum(prf_candidates, axis=0)
        weighted_query_embs = alpha * emb_qs[index]
        new_emb_q = np.sum(np.vstack((weighted_query_embs, weighted_sum_doc_embs)), axis=0)
        new_emb_qs.append(new_emb_q)
    new_emb_qs = np.array(new_emb_qs).astype('float32')
    return new_emb_qs


def rocchio_get_batch_prf_doc_emb(batch_candidates: np.ndarray = None,
                                  batch_clicks: np.ndarray = None,
                                  propensity: np.ndarray = None,
                                  ):
    new_doc_embs = []
    for index in range(len(batch_candidates)):
        prf_candidates = batch_candidates[index]
        if propensity is not None:
            prf_candidates = prf_candidates / propensity[:, np.newaxis]

        if batch_clicks is not None:
            prf_candidates = prf_candidates[np.nonzero(batch_clicks[index])[0]]
            if len(prf_candidates) == 0:
                new_doc_embs.append(np.zeros(768))  # hard code for now
                continue
        weighted_sum_doc_embs = np.sum(prf_candidates, axis=0)
        new_doc_embs.append(weighted_sum_doc_embs)

    new_doc_embs = np.array(new_doc_embs).astype('float32')
    return new_doc_embs


def main(args):
    qids, queries = read_query_file(args.query_path)
    qrel = load_qrels(args.qrel_path)

    if args.model_name == 'ANCE':
        encoder = AnceQueryEncoder(encoder_dir=args.model_path)
    elif args.model_name == 'TCTv2':
        encoder = TctColBertQueryEncoder(encoder_dir=args.model_path)
    else:
        raise NotImplementedError("query encoder is not implemented")

    print('Loading index...')
    faiss_index = read_index(args.index_path)
    # GPU has problem with reconstruct
    # res = faiss.StandardGpuResources()
    # faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
    faiss.omp_set_num_threads(10)


    print("Encoding...")
    q_embs = np.array([encoder.encode(q) for q in tqdm(queries)])
    print("Searching...")

    # First round of search
    if not args.feedback:
        batch_scores, batch_docids, batch_vectors = faiss_index.search_and_reconstruct(q_embs, 1000)
        for qid, scores, docids, vectors in zip(qids, batch_scores, batch_docids, batch_vectors):
            write_run_file(qid, scores, docids, args.output_path)
        return

    _, batch_docids, batch_vectors = faiss_index.search_and_reconstruct(q_embs, 10)
    batch_rels = get_batch_rels(qids, batch_docids, qrel)

    candidates = batch_vectors
    # second round of search
    if args.click_model is None:
        new_doc_embs = rocchio_get_batch_prf_doc_emb(candidates[:, :10])
        new_q_embs = 0.4 * q_embs + 0.6 * new_doc_embs
    else:
        # initial click model
        eta = args.propensity
        if args.click_model == 'perfect':
            pc = [0.0, 0.0, 1.0, 1.0]
        elif args.click_model == 'noise':
            pc = [0.2, 0.4, 0.8, 0.9]
        else:
            raise NotImplementedError(f"Click model {args.click_model} not implemented")
        click_model = PositionBasedClickModel(pc=pc, eta=eta)

        new_doc_embs = np.zeros_like(q_embs)
        for _ in tqdm(range(args.num_rep)):
            if args.click_model is not None:
                batch_clicks = click_model.batch_simulate(batch_rels)
            if args.debias:
                new_doc_embs += rocchio_get_batch_prf_doc_emb(candidates, batch_clicks, click_model.propensities)
            else:
                new_doc_embs += rocchio_get_batch_prf_doc_emb(candidates, batch_clicks)
        new_q_embs = 0.4 * q_embs + 0.6 * (new_doc_embs/args.num_rep)

    new_batch_scores, new_batch_docids, new_batch_vectors = faiss_index.search_and_reconstruct(new_q_embs, 1000)
    for qid, scores, docids, vectors in zip(qids, new_batch_scores, new_batch_docids, new_batch_vectors):
        write_run_file(qid, scores, docids, args.output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--qrel_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--index_path", type=str,  required=True)
    parser.add_argument("--feedback", action='store_true', help='Run implicit feedback or not')
    parser.add_argument("--click_model", type=str,  default=None, help='perfect or noise, default to None (only prf)')
    parser.add_argument("--propensity", type=float, default=0, help='eta, 0 means no propensity, default to 0')
    parser.add_argument("--num_rep", type=int, default=1000, help='Number of repeated queries')
    parser.add_argument("--debias", action='store_true')
    args = parser.parse_args()
    np.random.seed(313)
    if os.path.isfile(args.output_path):
        raise FileExistsError(f"run file {args.output_path} already exist")
    main(args)
