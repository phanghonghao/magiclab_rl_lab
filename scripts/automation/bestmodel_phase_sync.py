"""Generate docs/tracking/bestmodel_phase.json from orchestrator state.

This keeps the phase-tracking JSON on the RTX server aligned with the latest
orchestrator progress so local tooling can simply sync the file back.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from automation.phase_manager import PhaseManager
from automation.state_store import OrchestratorState, StateStore

_ITER_RE = re.compile(r"model_(\d+)\.pt$")


def _extract_iteration(name: Optional[str]) -> Optional[int]:
    if not name:
        return None
    match = _ITER_RE.search(name)
    return int(match.group(1)) if match else None


def _run_dir_name(path_str: Optional[str]) -> Optional[str]:
    if not path_str:
        return None
    return Path(path_str).name


def _normalize_sub_phase_status(status: Optional[str]) -> str:
    mapping = {
        "overfitting": "COMPLETE",
        "rollback_exhausted": "COMPLETE",
        "complete": "COMPLETE",
        "running": "RUNNING",
        "pending": "PENDING",
        "failed": "FAILED",
    }
    return mapping.get((status or "").lower(), (status or "UNKNOWN").upper())


def _load_existing(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _existing_phase_map(existing: dict) -> dict[str, dict]:
    return {
        phase.get("phase_id"): phase
        for phase in existing.get("phases", [])
        if isinstance(phase, dict) and phase.get("phase_id")
    }


def _count_tracked_sub_phases(existing: dict) -> int:
    count = 0
    for phase in existing.get("phases", []):
        if not isinstance(phase, dict):
            continue
        for sub in phase.get("sub_phases", []):
            if isinstance(sub, dict) and sub.get("id"):
                count += 1
    return count


def _find_checkpoint_rel(project_root: Path, sub_phase_id: str, best_model: Optional[str]) -> Optional[str]:
    if not best_model:
        return None
    model_dir = project_root / "models" / "p" / sub_phase_id
    if not model_dir.is_dir():
        return None
    direct = model_dir / f"{sub_phase_id}_{best_model}"
    if direct.exists():
        return direct.relative_to(project_root).as_posix()
    for candidate in sorted(model_dir.glob("*.pt")):
        if candidate.name.endswith(best_model):
            return candidate.relative_to(project_root).as_posix()
    return None


def _find_policy_rel(project_root: Path, sub_phase_id: str, best_iteration: Optional[int]) -> Optional[str]:
    model_dir = project_root / "models" / "p" / sub_phase_id
    if not model_dir.is_dir():
        return None

    candidates = sorted(model_dir.glob("*policy*.pt"))
    if not candidates:
        return None

    if best_iteration is not None:
        tagged = [
            path for path in candidates
            if f"m{best_iteration}" in path.stem or f"_{best_iteration}" in path.stem
        ]
        if tagged:
            return tagged[0].relative_to(project_root).as_posix()

    return candidates[0].relative_to(project_root).as_posix()


def _find_video_rel(project_root: Path, sub_phase_id: str) -> Optional[str]:
    video_root = project_root / "videos" / "p" / sub_phase_id
    if not video_root.is_dir():
        return None

    child_dirs = sorted(
        [path for path in video_root.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if child_dirs:
        return child_dirs[0].relative_to(project_root).as_posix() + "/"
    return video_root.relative_to(project_root).as_posix() + "/"


def _build_sub_phase_entry(
    project_root: Path,
    sub_phase_id: str,
    entry: dict,
    *,
    current_status: Optional[str] = None,
) -> dict:
    checkpoint_path = entry.get("best_checkpoint_path")
    best_model = Path(checkpoint_path).name if checkpoint_path else None
    best_iteration = _extract_iteration(best_model)
    best_reward = entry.get("best_reward")
    run_dir = entry.get("training_run_dir")
    if not run_dir and checkpoint_path:
        run_dir = str(Path(checkpoint_path).parent)

    result = {
        "id": sub_phase_id,
        "run_dir": _run_dir_name(run_dir),
        "best_model": best_model,
        "best_iteration": best_iteration,
        "best_reward": best_reward,
        "peak_reward": best_reward,
        "status": _normalize_sub_phase_status(current_status or entry.get("status")),
    }

    completed_at = entry.get("completed_at")
    if completed_at:
        result["completed_at"] = completed_at

    checkpoint_rel = _find_checkpoint_rel(project_root, sub_phase_id, best_model)
    policy_rel = _find_policy_rel(project_root, sub_phase_id, best_iteration)
    if checkpoint_rel or policy_rel:
        result["local_models"] = {
            "checkpoint": checkpoint_rel,
            "jit_policy": policy_rel,
        }

    video_rel = _find_video_rel(project_root, sub_phase_id)
    if video_rel:
        result["video"] = video_rel

    return result


def _build_phase_best(
    sub_entries: list[dict],
    phase_history_entry: Optional[dict],
) -> tuple[Optional[str], Optional[float]]:
    if phase_history_entry:
        ckpt = phase_history_entry.get("best_checkpoint_path")
        reward = phase_history_entry.get("best_reward")
        if ckpt:
            ckpt_name = Path(ckpt).name
            for sub_entry in sub_entries:
                if sub_entry.get("best_model") == ckpt_name:
                    return f"{sub_entry['id']}/{ckpt_name}", reward

    ranked = [entry for entry in sub_entries if entry.get("best_reward") is not None and entry.get("best_model")]
    if not ranked:
        return None, None
    ranked.sort(key=lambda item: item["best_reward"], reverse=True)
    best = ranked[0]
    return f"{best['id']}/{best['best_model']}", best["best_reward"]


def build_bestmodel_phase_payload(
    project_root: str | Path,
    plan_path: str | Path,
    state_path: str | Path,
    output_path: str | Path,
) -> dict:
    project_root = Path(project_root).resolve()
    output_path = Path(output_path)
    phase_mgr = PhaseManager(plan_path)
    state = StateStore(project_root / state_path).load()
    existing = _load_existing(output_path)
    existing_phase_map = _existing_phase_map(existing)

    if state is None and _count_tracked_sub_phases(existing) == 0:
        raise RuntimeError(
            "Refusing to rebuild bestmodel_phase.json without orchestrator state "
            "or an existing populated tracking file."
        )

    stage_history = state.stage_history if state else []
    phase_history = state.phase_history if state else []

    latest_stage_by_id: dict[str, dict] = {}
    for item in stage_history:
        latest_stage_by_id[item.get("sub_phase_id")] = item

    phase_history_by_id = {item.get("phase_id"): item for item in phase_history}

    phases_payload: list[dict] = []
    for phase in phase_mgr.phases:
        existing_phase = existing_phase_map.get(phase.id, {})
        existing_sub_map = {
            sub.get("id"): sub
            for sub in existing_phase.get("sub_phases", [])
            if isinstance(sub, dict) and sub.get("id")
        }
        sub_entries: list[dict] = []

        for sub_phase in phase.sub_phases:
            record = latest_stage_by_id.get(sub_phase.id)
            if record is not None:
                sub_entries.append(_build_sub_phase_entry(project_root, sub_phase.id, record))
                continue

            if state and state.current_stage_id == sub_phase.id:
                current_entry = {
                    "status": state.current_stage_status,
                    "training_run_dir": state.training_run_dir,
                    "best_checkpoint_path": state.best_checkpoint_path,
                    "best_reward": state.best_reward,
                }
                sub_entries.append(
                    _build_sub_phase_entry(
                        project_root,
                        sub_phase.id,
                        current_entry,
                        current_status=state.current_stage_status,
                    )
                )
                continue

            if sub_phase.id in existing_sub_map:
                sub_entries.append(existing_sub_map[sub_phase.id])

        phase_entry = dict(existing_phase) if existing_phase else {}
        phase_entry["phase_id"] = phase.id
        phase_entry["phase_name"] = phase.name
        phase_entry["sub_phases"] = sub_entries

        phase_best, phase_best_reward = _build_phase_best(sub_entries, phase_history_by_id.get(phase.id))
        if phase_best is not None:
            phase_entry["phase_best"] = phase_best
            phase_entry["phase_best_reward"] = phase_best_reward
        elif "phase_best" in existing_phase:
            phase_entry["phase_best"] = existing_phase.get("phase_best")
            phase_entry["phase_best_reward"] = existing_phase.get("phase_best_reward")

        if phase.id in phase_history_by_id:
            phase_entry["status"] = "COMPLETE"
        elif state and state.current_phase_id == phase.id:
            if (state.current_stage_status or "").lower() == "failed":
                phase_entry["status"] = "FAILED"
            else:
                phase_entry["status"] = "IN_PROGRESS"
        elif sub_entries:
            existing_status = existing_phase.get("status")
            if existing_status:
                phase_entry["status"] = existing_status
            else:
                all_complete = all(
                    str(sub.get("status", "")).upper().startswith("COMPLETE")
                    for sub in sub_entries
                )
                phase_entry["status"] = "COMPLETE" if all_complete else "IN_PROGRESS"
        else:
            phase_entry["status"] = existing_phase.get("status", "PLANNED")

        phases_payload.append(phase_entry)

    return {
        "source": "5-phase-pipeline",
        "plan": Path(plan_path).name,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "phases": phases_payload,
        "archived_runs": existing.get("archived_runs", []),
    }


def sync_bestmodel_phase_json(
    project_root: str | Path,
    plan_path: str | Path,
    state_path: str | Path = "orchestrator_state.json",
    output_rel: str | Path = "docs/tracking/bestmodel_phase.json",
) -> Path:
    project_root = Path(project_root).resolve()
    output_path = project_root / output_rel
    payload = build_bestmodel_phase_payload(project_root, plan_path, state_path, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return output_path
