from modeling import AnceQueryEncoder, PositionBasedClickModel, TctColBertQueryEncoder
import faiss
from faiss import read_index
import numpy as np
from tqdm import tqdm
import os
from argparse import ArgumentParser
from run_seen_queries import write_run_file, read_query_file, load_qrels, get_batch_rels, rocchio_get_batch_prf_doc_emb


def main(args):
    train_qids, train_queries = read_query_file(args.train_query_path)
    test_qids, test_queries = read_query_file(args.test_query_path)
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

    train_q_embs = np.array([encoder.encode(q) for q in tqdm(train_queries, desc="Encoding train queries")])
    test_q_embs = np.array([encoder.encode(q) for q in tqdm(test_queries, desc="Encoding test queries")])

    # find close queires
    q_distances = np.matmul(test_q_embs, train_q_embs.T)
    topk_idx = []
    k = 3
    for row in q_distances:
        topk_idx.append(np.argpartition(row, -k)[-k:])
    topk_idx = np.array(topk_idx)

    ####### print topk nearst queries #######
    # for i, query in enumerate(test_queries):
    #     print(f"original query: {test_qids[i]} {query}")
    #     for n, idx in enumerate(topk_idx[i]):
    #         print(f"NN query {n}: {train_qids[idx]} {train_queries[idx]}")
    #     print()
    #########################################

    # First round of search
    if not args.feedback:
        batch_scores, batch_docids, batch_vectors = faiss_index.search_and_reconstruct(test_q_embs, 1000)
        for qid, scores, docids, vectors in zip(test_qids, batch_scores, batch_docids, batch_vectors):
            write_run_file(qid, scores, docids, args.output_path)
        return

    _, batch_docids, batch_vectors = faiss_index.search_and_reconstruct(test_q_embs, 10)

    candidates = batch_vectors
    # second round of search
    if args.click_model is None:
        new_doc_embs = rocchio_get_batch_prf_doc_emb(candidates[:, :10])
        new_q_embs = 0.4 * test_q_embs + 0.6 * new_doc_embs
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

        print("Search for train queries")
        _, train_batch_docids, train_batch_vectors = faiss_index.search_and_reconstruct(train_q_embs, 10)
        batch_rels = get_batch_rels(train_qids, train_batch_docids, qrel)

        candidates = train_batch_vectors

        train_doc_embs = np.zeros_like(train_q_embs)
        for _ in tqdm(range(args.num_rep)):
            if args.click_model is not None:
                batch_clicks = click_model.batch_simulate(batch_rels)

            if args.debias:
                train_doc_embs += rocchio_get_batch_prf_doc_emb(candidates, batch_clicks, click_model.propensities)
            else:
                train_doc_embs += rocchio_get_batch_prf_doc_emb(candidates, batch_clicks)

        train_doc_embs = train_doc_embs/args.num_rep
        new_q_embs = 0.4 * test_q_embs + 0.6 * np.sum(train_doc_embs[topk_idx], axis=1)/k

    new_batch_scores, new_batch_docids, new_batch_vectors = faiss_index.search_and_reconstruct(new_q_embs, 1000)
    for qid, scores, docids, vectors in zip(test_qids, new_batch_scores, new_batch_docids, new_batch_vectors):
        write_run_file(qid, scores, docids, args.output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--train_query_path", type=str, required=True)
    parser.add_argument("--test_query_path", type=str, required=True)
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
