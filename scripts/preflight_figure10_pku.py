#!/usr/bin/env python3
"""Read-only environment and data preflight for PKU Figure 10 training."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.requirements import Requirement


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WAN22 = PROJECT_ROOT.parent.parent / "weights/shared/Wan2.2-TI2V-5B"
CONFIGS = {
    "robot-only": "figure10_pku_robot_only",
    "ego2robot-ego": "figure10_pku_ego2robot_ego",
    "ego2robot-robot": "figure10_pku_ego2robot_robot",
    "ego-robot": "figure10_pku_ego_robot",
}


def report(ok: bool, label: str, detail: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def check_requirements() -> bool:
    if sys.version_info < (3, 10):
        report(False, "python", sys.version.split()[0] + " (requires >=3.10)")
        return False
    report(True, "python", f"{sys.executable} ({sys.version.split()[0]})")

    dependencies = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    failures = []
    for raw in dependencies:
        requirement = Requirement(raw)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = version(requirement.name)
        except PackageNotFoundError:
            failures.append(f"{requirement.name}=missing (need {requirement.specifier or 'installed'})")
            continue
        if requirement.specifier and installed not in requirement.specifier:
            failures.append(f"{requirement.name}={installed} (need {requirement.specifier})")
    return report(not failures, "dependencies", "all constraints satisfied" if not failures else "; ".join(failures))


def check_cuda(skip_gpu: bool) -> bool:
    try:
        import torch
    except Exception as exc:
        return report(False, "cuda", f"torch import failed: {exc}")
    available = torch.cuda.is_available()
    count = torch.cuda.device_count() if available else 0
    detail = f"available={available}, devices={count}, torch_cuda={torch.version.cuda}"
    if skip_gpu:
        report(True, "cuda", detail + " (not required by --skip-gpu)")
        return True
    return report(available and count > 0, "cuda", detail)


def check_weights() -> bool:
    root = Path(os.environ.get("OPENWAM_WAN22_PATH", DEFAULT_WAN22)).expanduser().resolve()
    expected = [
        "config.json",
        "Wan2.2_VAE.pth",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "diffusion_pytorch_model.safetensors.index.json",
    ]
    missing = [name for name in expected if not (root / name).is_file()]
    shards = sorted(root.glob("diffusion_pytorch_model-*-of-*.safetensors")) if root.is_dir() else []
    ok = not missing and len(shards) == 3
    detail = str(root) if ok else f"root={root}, missing={missing}, shards={len(shards)}/3"
    return report(ok, "Wan2.2 weights", detail)


def _native_datasets(dataset):
    if hasattr(dataset, "records"):
        yield dataset
        return
    for child in getattr(dataset, "datasets", ()):
        yield from _native_datasets(child)


def _sample_records_by_source(dataset, count: int):
    groups: dict[tuple[str, str], list[dict]] = {}
    for native in _native_datasets(dataset):
        for record in native.records:
            key = (str(record.get("_source_domain", "unknown")), str(record.get("source_name", "pku")))
            groups.setdefault(key, []).append(record)

    for key, records in sorted(groups.items()):
        sample_count = min(count, len(records))
        if sample_count == 1:
            positions = [0]
        else:
            positions = [i * (len(records) - 1) // (sample_count - 1) for i in range(sample_count)]
        for position in dict.fromkeys(positions):
            yield key, records[position]


def check_payload_paths(dataset, samples_per_source: int) -> bool:
    from openwam.dataloader.pku_native30 import pku_record_payload_paths

    missing: list[str] = []
    checked_records = 0
    checked_paths = 0
    sources: set[tuple[str, str]] = set()
    for source, record in _sample_records_by_source(dataset, samples_per_source):
        checked_records += 1
        sources.add(source)
        for path in pku_record_payload_paths(record):
            checked_paths += 1
            if not path.is_file():
                missing.append(str(path))

    detail = f"sources={len(sources)}, records={checked_records}, files={checked_paths}"
    if missing:
        preview = "; ".join(missing[:8])
        if len(missing) > 8:
            preview += f"; ... ({len(missing) - 8} more)"
        detail += f", missing={len(missing)}: {preview}"
    return report(not missing, "dataset payload paths", detail)


def check_dataset(mode: str, sample: bool, paths_per_source: int) -> bool:
    try:
        import hydra
        import torch
        from hydra import compose
        from openwam.dataloader.registry import build_dataset

        with hydra.initialize_config_dir(version_base=None, config_dir=str(PROJECT_ROOT / "configs")):
            cfg = compose(config_name=CONFIGS[mode])
        dataset = build_dataset(cfg.dataloader, split="train")
        detail = f"{type(dataset).__name__}, windows={len(dataset):,}"
        paths_ok = True
        if paths_per_source > 0:
            paths_ok = check_payload_paths(dataset, paths_per_source)
        if sample:
            item = next(
                iter(
                    torch.utils.data.DataLoader(
                        dataset,
                        batch_size=1,
                        num_workers=0,
                        shuffle=False,
                        collate_fn=list,
                        pin_memory=False,
                    )
                )
            )[0]
            detail += (
                f", sample={item.get('_dataset_name')}, frames={len(item['video'])}, "
                f"action={tuple(item['action'].shape)}, active_mask={int(item['action_mask'].sum())}"
            )
        return report(paths_ok, "dataset", detail)
    except Exception as exc:
        return report(False, "dataset", f"{type(exc).__name__}: {exc}")


def check_stage_checkpoint(mode: str) -> bool:
    if mode != "ego2robot-robot":
        return True
    raw = os.environ.get("OPENWAM_EGO_STAGE_CKPT")
    if not raw:
        return report(False, "ego stage checkpoint", "OPENWAM_EGO_STAGE_CKPT is not set")
    path = Path(raw).expanduser().resolve()
    weights = list(path.glob("checkpoint_step_*.safetensors")) if path.is_dir() else []
    return report(bool(weights), "ego stage checkpoint", f"{path}, checkpoints={len(weights)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=CONFIGS, required=True)
    parser.add_argument("--sample", action="store_true", help="decode one real sample with num_workers=0")
    parser.add_argument(
        "--check-paths-per-source",
        type=int,
        default=0,
        help="check this many evenly spaced records per source before training",
    )
    parser.add_argument("--skip-gpu", action="store_true", help="report CUDA state without requiring a GPU")
    args = parser.parse_args()

    sys.path.insert(0, str(PROJECT_ROOT))
    checks = [
        check_requirements(),
        check_cuda(args.skip_gpu),
        check_weights(),
        check_stage_checkpoint(args.mode),
        check_dataset(args.mode, args.sample, max(0, args.check_paths_per_source)),
    ]
    ok = all(checks)
    print("PREFLIGHT=" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
