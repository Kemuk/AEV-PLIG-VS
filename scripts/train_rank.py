"""CLI: aev-plig-rank-train — train a LambdaMART ranking model on ligand fingerprints."""
import argparse
from pathlib import Path

from aev_plig.rank import RankConfig, overall_enrichment, per_target_enrichment, train_rank

_CHOICES = ["pdbbind", "bindingnet", "bindingdb"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources",      nargs="+", default=["pdbbind", "bindingnet"], choices=_CHOICES)
    p.add_argument("--pool-sources", nargs="+", default=None, choices=_CHOICES,
                   help="Datasets for negatives (default: same as --sources)")
    p.add_argument("--data-dir",     default="data")
    p.add_argument("--n-negatives",  type=int,   default=RankConfig.N_NEGATIVES)
    p.add_argument("--seed",         type=int,   default=RankConfig.NEGATIVE_SEED)
    p.add_argument("--n-estimators", type=int,   default=RankConfig.N_ESTIMATORS)
    p.add_argument("--lr",           type=float, default=RankConfig.LEARNING_RATE)
    p.add_argument("--num-leaves",   type=int,   default=RankConfig.NUM_LEAVES)
    p.add_argument("--n-jobs",       type=int,   default=1,
                   help="Parallel workers for featurisation (serial default: 1)")
    p.add_argument("--chunk-size",   type=int,   default=5000,
                   help="Number of records processed per serial chunk during prep")
    p.add_argument("--cache-dir",    default=None,
                   help="Directory to cache fingerprints (skip if absent)")
    p.add_argument("--output-dir",   default=RankConfig.RANK_MODELS_DIR)
    return p.parse_args()


def main():
    args = parse_args()

    cfg = RankConfig()
    cfg.N_NEGATIVES = args.n_negatives
    cfg.NEGATIVE_SEED = args.seed
    cfg.N_ESTIMATORS = args.n_estimators
    cfg.LEARNING_RATE = args.lr
    cfg.NUM_LEAVES = args.num_leaves

    print(f"Sources:      {args.sources}")
    print(f"Pool sources: {args.pool_sources or '(same as sources)'}")
    print(
        f"Negatives:    {cfg.N_NEGATIVES}  |  Seed: {cfg.NEGATIVE_SEED}  |  "
        f"Jobs: {args.n_jobs}  |  Chunk size: {args.chunk_size}"
    )

    model, dataset = train_rank(
        sources=args.sources,
        pool_sources=args.pool_sources,
        data_dir=args.data_dir,
        config=cfg,
        output_dir=args.output_dir,
        n_jobs=args.n_jobs,
        cache_dir=args.cache_dir,
        chunk_size=args.chunk_size,
    )

    print("\n=== Validation set enrichment ===")
    try:
        summary = overall_enrichment(model, dataset, split="valid")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}")
        df = per_target_enrichment(model, dataset, split="valid")
        out = Path(args.output_dir) / "validation_enrichment.csv"
        df.write_csv(out)
        print(f"\nPer-target results saved to {out}")
    except KeyError:
        print("  No validation split found — skipping evaluation.")

if __name__ == "__main__":
    main()
