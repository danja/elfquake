"""Self-supervised model CLI commands."""

from __future__ import annotations

from argparse import Namespace, _SubParsersAction
from pathlib import Path

from elfquake.models.torch_self_supervised import compare_sequence_embedding_domains, pretrain_sequence_autoencoder


def register_self_supervised_commands(subparsers: _SubParsersAction) -> None:
    sequence_autoencoder = subparsers.add_parser("pretrain-sequence-autoencoder")
    sequence_autoencoder.add_argument("--sequence-manifest", type=Path, required=True)
    sequence_autoencoder.add_argument("--out", type=Path, required=True)
    sequence_autoencoder.add_argument("--modality", default="")
    sequence_autoencoder.add_argument("--lookback-steps", type=int, default=24)
    sequence_autoencoder.add_argument("--stride", type=int, default=1)
    sequence_autoencoder.add_argument("--train-fraction", type=float, default=0.8)
    sequence_autoencoder.add_argument("--mask-probability", type=float, default=0.15)
    sequence_autoencoder.add_argument("--clean-loss-weight", type=float, default=0.0)
    sequence_autoencoder.add_argument("--epochs", type=int, default=30)
    sequence_autoencoder.add_argument("--learning-rate", type=float, default=0.0003)
    sequence_autoencoder.add_argument("--hidden-units", type=int, default=32)
    sequence_autoencoder.add_argument("--embedding-units", type=int, default=8)
    sequence_autoencoder.add_argument("--batch-size", type=int, default=32)
    sequence_autoencoder.add_argument("--seed", type=int, default=42)
    sequence_autoencoder.add_argument("--no-missing-masks", action="store_true")
    sequence_autoencoder.add_argument("--checkpoint-out", type=Path)
    sequence_autoencoder.add_argument("--embeddings-out", type=Path)
    sequence_autoencoder.set_defaults(func=_pretrain_sequence_autoencoder)

    domain_compare = subparsers.add_parser("compare-sequence-embedding-domains")
    domain_compare.add_argument("--real-sequence-manifest", type=Path, required=True)
    domain_compare.add_argument("--synthetic-sequence-manifest", type=Path, action="append", required=True)
    domain_compare.add_argument("--out", type=Path, required=True)
    domain_compare.add_argument("--real-modality", default="real_vlf_image")
    domain_compare.add_argument("--synthetic-modality", default="synthetic_piezo_vlf")
    domain_compare.add_argument("--descriptor-profile", default="shape", choices=["shape", "full"])
    domain_compare.add_argument("--lookback-steps", type=int, default=24)
    domain_compare.add_argument("--stride", type=int, default=1)
    domain_compare.add_argument("--train-fraction", type=float, default=0.8)
    domain_compare.add_argument("--mask-probability", type=float, default=0.15)
    domain_compare.add_argument("--clean-loss-weight", type=float, default=0.0)
    domain_compare.add_argument("--inlier-fraction", type=float, default=0.25)
    domain_compare.add_argument("--epochs", type=int, default=30)
    domain_compare.add_argument("--learning-rate", type=float, default=0.0003)
    domain_compare.add_argument("--hidden-units", type=int, default=32)
    domain_compare.add_argument("--embedding-units", type=int, default=8)
    domain_compare.add_argument("--batch-size", type=int, default=32)
    domain_compare.add_argument("--seed", type=int, default=42)
    domain_compare.add_argument("--no-missing-masks", action="store_true")
    domain_compare.add_argument("--embeddings-out", type=Path)
    domain_compare.set_defaults(func=_compare_sequence_embedding_domains)


def _pretrain_sequence_autoencoder(args: Namespace) -> int:
    report = pretrain_sequence_autoencoder(
        sequence_manifest_path=args.sequence_manifest,
        out_path=args.out,
        modality=args.modality,
        lookback_steps=args.lookback_steps,
        stride=args.stride,
        train_fraction=args.train_fraction,
        mask_probability=args.mask_probability,
        clean_loss_weight=args.clean_loss_weight,
        inlier_fraction=args.inlier_fraction,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        embedding_units=args.embedding_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        checkpoint_out=args.checkpoint_out,
        embeddings_out=args.embeddings_out,
    )
    print(f"status: {report['status']}")
    print(f"rows: {report['row_count']}")
    print(f"windows: {report['window_count']}")
    if "train_window_count" in report:
        print(f"train windows: {report['train_window_count']}")
        print(f"test windows: {report['test_window_count']}")
    print(f"output: {args.out}")
    return 0


def _compare_sequence_embedding_domains(args: Namespace) -> int:
    report = compare_sequence_embedding_domains(
        real_sequence_manifest_path=args.real_sequence_manifest,
        synthetic_sequence_manifest_paths=args.synthetic_sequence_manifest,
        out_path=args.out,
        real_modality=args.real_modality,
        synthetic_modality=args.synthetic_modality,
        descriptor_profile=args.descriptor_profile,
        lookback_steps=args.lookback_steps,
        stride=args.stride,
        train_fraction=args.train_fraction,
        mask_probability=args.mask_probability,
        clean_loss_weight=args.clean_loss_weight,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        embedding_units=args.embedding_units,
        batch_size=args.batch_size,
        seed=args.seed,
        include_missing_masks=not args.no_missing_masks,
        embeddings_out=args.embeddings_out,
    )
    print(f"status: {report['status']}")
    print(f"real windows: {report['real_window_count']}")
    print(f"synthetic windows: {report['synthetic_window_count']}")
    if report.get("embedding_comparison"):
        comparison = report["embedding_comparison"]
        print(f"centroid distance: {comparison.get('centroid_euclidean_distance', '')}")
    print(f"output: {args.out}")
    return 0
