"""Build one EEG preprocessing variant from the THINGS-EEG2 raw recordings.

    python scripts/09_preprocess_eeg_raw.py --config configs/EEG/preproc/baseline.yaml
    python scripts/09_preprocess_eeg_raw.py --config configs/EEG/preproc/channels_17.yaml --subjects sub-08
    python scripts/09_preprocess_eeg_raw.py --config configs/EEG/preproc/baseline.yaml --force

Writes ``data/processed/eeg_preproc/<variant>/<subject>/`` with
``preprocessed_eeg_{training,test}.npy`` (same contract as the official
derivatives), ``metadata.json`` and QC figures. Train the experiments on it with
``--set dataset.source=raw --set dataset.preproc_variant=<variant>``.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.preprocessing import build_variant, preproc_config_hash  # noqa: E402
from src.utils import ExtendOverrides, get_logger, load_config, save_config  # noqa: E402

logger = get_logger("preprocess_eeg")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", nargs="*", action=ExtendOverrides, default=None,
                    help="config overrides, e.g. dataset.subject_selection=sub-08")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="subject ids to build (default: dataset.subject_selection)")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if the variant cache already exists")
    ap.add_argument("--no-qc", action="store_true", help="skip QC figures")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    variant = cfg.get("dataset.preproc_variant", "baseline")
    logger.info("Variant '%s' (config hash %s) from %s", variant,
                preproc_config_hash(cfg), cfg.get("dataset.root_dir"))

    subjects = args.subjects or None
    summary = build_variant(cfg, subjects=subjects, force=args.force,
                            qc=not args.no_qc)

    out_dir = Path(summary["out_dir"])
    save_config(cfg, out_dir / "config.yaml")
    logger.info("Done -> %s", out_dir)
    for subj, info in summary["subjects"].items():
        if info.get("status") == "cached":
            logger.info("  %s: cached (use --force to rebuild)", subj)
        else:
            logger.info("  %s: train %s | test %s | %.1fs", subj,
                        info.get("training", {}).get("shape"),
                        info.get("test", {}).get("shape"), info.get("seconds", 0))


if __name__ == "__main__":
    main()
