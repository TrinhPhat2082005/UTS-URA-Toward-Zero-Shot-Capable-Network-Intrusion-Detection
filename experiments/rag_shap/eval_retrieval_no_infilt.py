import argparse
import numpy as np


def cosine_sim_matrix(query, pool):
    q_norm = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    p_norm = pool / (np.linalg.norm(pool, axis=1, keepdims=True) + 1e-12)
    return q_norm @ p_norm.T


def eval_method(eval_sets, build_pools, method):
    pool_labels = list(build_pools.keys())
    if method == "centroid":
        centroids = {lbl: vecs.mean(axis=0, keepdims=True) for lbl, vecs in build_pools.items()}

    results = []
    for true_label, eval_vecs in eval_sets.items():
        n_eval = len(eval_vecs)
        best_sim_per_label = np.zeros((n_eval, len(pool_labels)), dtype=np.float32)
        for j, cand_label in enumerate(pool_labels):
            if method == "centroid":
                sims = cosine_sim_matrix(eval_vecs, centroids[cand_label])
                best_sim_per_label[:, j] = sims[:, 0]
            else:  # knn
                sims = cosine_sim_matrix(eval_vecs, build_pools[cand_label])
                best_sim_per_label[:, j] = sims.max(axis=1)

        ranked_idx = np.argsort(-best_sim_per_label, axis=1)
        true_pool_idx = pool_labels.index(true_label)
        top1_hits = top3_hits = top5_hits = 0
        for row in ranked_idx:
            rank_of_true = np.where(row == true_pool_idx)[0][0]
            if rank_of_true == 0:
                top1_hits += 1
            if rank_of_true < 3:
                top3_hits += 1
            if rank_of_true < 5:
                top5_hits += 1

        results.append({
            "label": true_label,
            "n_build": len(build_pools[true_label]),
            "n_eval": n_eval,
            "top1_acc": top1_hits / n_eval,
            "top3_acc": top3_hits / n_eval,
            "top5_acc": top5_hits / n_eval,
        })
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--variant_name", required=True)
    parser.add_argument("--eval_frac", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    data = np.load(args.npz, allow_pickle=True)
    label_keys = [k for k in data.keys() if k.startswith("label__")]
    label_map = {k: k[len("label__"):].replace("_", " ") for k in label_keys}

    build_pools = {}
    eval_sets = {}
    for key in label_keys:
        label = label_map[key]
        if label == "Infilteration":
            continue  # excluded per user direction: zero-shot, not a fair target
        vecs = data[key]
        n = len(vecs)
        idx = rng.permutation(n)
        n_eval = max(1, int(n * args.eval_frac))
        eval_idx = idx[:n_eval]
        build_idx = idx[n_eval:]
        build_pools[label] = vecs[build_idx]
        eval_sets[label] = vecs[eval_idx]

    print(f"\n{'='*70}\nVARIANT: {args.variant_name}  (Infilteration excluded, {len(build_pools)} labels)\n{'='*70}")

    for method in ["centroid", "knn"]:
        results = eval_method(eval_sets, build_pools, method)
        avg_top1 = np.mean([r["top1_acc"] for r in results])
        avg_top3 = np.mean([r["top3_acc"] for r in results])
        avg_top5 = np.mean([r["top5_acc"] for r in results])
        print(f"\n--- method: {method} ---")
        print(f"{'label':28s} {'n_build':>8s} {'n_eval':>7s} {'top1':>6s} {'top3':>6s} {'top5':>6s}")
        for r in results:
            print(f"{r['label']:28s} {r['n_build']:8d} {r['n_eval']:7d} "
                  f"{r['top1_acc']:6.3f} {r['top3_acc']:6.3f} {r['top5_acc']:6.3f}")
        print(f"{'MACRO AVERAGE':28s} {'':8s} {'':7s} {avg_top1:6.3f} {avg_top3:6.3f} {avg_top5:6.3f}")


if __name__ == "__main__":
    main()
