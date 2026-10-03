"""Causal pre-validation replay shared by training and deployed inference."""
from dataclasses import replace

import numpy as np
import torch

from .decision_history import causal_c0_bridge_batch
from .pipeline import load_backbone


@torch.inference_mode()
def with_validation_burnin(train, validation, panel, config, device, days):
    """Prepend X-only history; run the actor ONCE, then discard these rows.

    This warms GRU, holdings and position ages together. Labels are not read.
    The missing label-boundary date is inferred from causal predictor inputs.
    """
    if days < 0:
        raise ValueError("validation burnin must be nonnegative")
    if days == 0:
        return validation
    for key in ("backbone_sha256", "panel_manifest_sha256"):
        if train.manifest.get(key) != validation.manifest.get(key):
            raise ValueError(f"train/validation cache mismatch: {key}")
    first = int(validation.date_indices[0])
    dates = np.arange(first - days, first, dtype=np.int64)
    if dates[0] < 0:
        raise ValueError("not enough pre-validation history")
    if validation.predicted_return is None or train.predicted_return is None:
        raise ValueError("validation burnin requires dual-head caches")
    names = ("hidden", "base_score", "eligible", "predicted_return")
    arrays = {name: np.empty((days, *getattr(validation, name).shape[1:]),
                            dtype=getattr(validation, name).dtype) for name in names}
    missing = []
    for position, date in enumerate(dates):
        if int(train.date_indices[0]) <= date <= int(train.date_indices[-1]):
            row = int(train.rows_for_dates(np.array([date]))[0])
            for name in names:
                arrays[name][position] = getattr(train, name)[row]
        else:
            missing.append((position, int(date)))
    if missing:
        # Constructing the frozen backbone consumes RNG for its temporary
        # initialization. Do not change actor initialization or rollout seeds
        # merely by enabling validation warmup on rank zero.
        rng_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=rng_devices):
            backbone, _ = load_backbone(config, panel, device)
        for position, date in missing:
            bridge = causal_c0_bridge_batch(panel, date, config)
            values, valid, eligible = [x.to(device) for x in bridge[:3]]
            hidden = backbone.encode_hidden(values, valid, eligible,
                long_memory=bridge[3].to(device) if len(bridge) == 4 else None)
            arrays["hidden"][position] = hidden[-1].float().cpu().numpy()
            arrays["base_score"][position] = backbone.score_hidden(hidden, eligible)[-1].float().cpu().numpy()
            arrays["predicted_return"][position] = backbone.predict_return_hidden(hidden, eligible)[-1].float().cpu().numpy()
            arrays["eligible"][position] = eligible[-1].cpu().numpy()
        del backbone, hidden, values, valid, eligible
        if device.type == "cuda":
            torch.cuda.empty_cache()
    combined = {name: np.concatenate((arrays[name], getattr(validation, name)), axis=0)
                for name in names}
    # Match the original training-cache tradability for non-official callers.
    prefix_tradable = ~np.asarray(panel.limit_flags[dates, :, 0], dtype=bool)
    for position, date in enumerate(dates):
        if int(train.date_indices[0]) <= date <= int(train.date_indices[-1]):
            row = int(train.rows_for_dates(np.array([date]))[0])
            prefix_tradable[position] = train.tradable[row]
    manifest = dict(validation.manifest,
                    hidden_shape=list(combined["hidden"].shape),
                    decision_burnin_days=days)
    return replace(validation, date_indices=np.concatenate((dates, validation.date_indices)),
                   tradable=np.concatenate((prefix_tradable, validation.tradable)),
                   manifest=manifest, **combined)
