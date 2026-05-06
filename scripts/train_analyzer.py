#!/usr/bin/env python3
"""
Training Failure Analyzer for MagicBot Z1 RL Training.

Rule-based diagnostics engine that reads TensorBoard time-series data,
classifies failure modes, locates root causes, generates parameter tuning
recommendations, and compares against successful baseline runs.

Designed to be run on RTX6000 alongside train_monitor.py. Outputs structured
JSON for Claude to parse and present.

Usage:
    # Analyze all failed/overfitted runs
    python scripts/train_analyzer.py --all --terrain gentle

    # Analyze a single run
    python scripts/train_analyzer.py --run_dir logs/.../<RUN_DIR>

    # Specify baseline run
    python scripts/train_analyzer.py --run_dir logs/.../<RUN_DIR> --baseline s2_gentle

    # Only output JSON (for Claude parsing)
    python scripts/train_analyzer.py --run_dir logs/.../<RUN_DIR> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Reuse TensorBoard parsing from train_monitor
from train_monitor import (
    TensorBoardParser,
    RunState,
    TAG_REWARD,
    TAG_ACTION_RATE,
    TAG_VALUE_LOSS,
    TAG_ENTROPY,
    TAG_EP_LEN,
    TAG_TIME_OUT,
    TAG_BAD_ORI,
    TAG_VEL_ERR,
    MonitorConfig,
    OverfittingDetector,
    BestModelTracker,
    find_run_dirs,
)


# --------------------------------------------------------------------------- #
# 1. Trend Analysis                                                            #
# --------------------------------------------------------------------------- #


@dataclass
class TrendResult:
    """Trend metrics for a single scalar over time."""
    slope: float = 0.0           # linear regression slope (per iter)
    slope_pct: float = 0.0       # slope as % of peak absolute value
    inflection_iter: int = 0     # iter where slope sign reverses (peak/trough)
    best_iter: int = 0
    best_val: float = 0.0
    latest_val: float = 0.0
    latest_iter: int = 0
    n_points: int = 0


def compute_trend(data: list[tuple[int, float]]) -> TrendResult:
    """Compute trend metrics from (iter, value) time series.

    Uses simple linear regression for slope, and detects the inflection
    point where the running slope changes sign (peak/trough of the metric).
    """
    if len(data) < 3:
        if data:
            return TrendResult(
                best_iter=data[-1][0],
                best_val=data[-1][1],
                latest_val=data[-1][1],
                latest_iter=data[-1][0],
                n_points=len(data),
            )
        return TrendResult()

    iters = [d[0] for d in data]
    vals = [d[1] for d in data]
    n = len(data)

    # Linear regression: y = a + b*x
    mean_x = sum(iters) / n
    mean_y = sum(vals) / n
    ss_xx = sum((x - mean_x) ** 2 for x in iters)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in data)
    slope = ss_xy / ss_xx if ss_xx > 0 else 0.0

    # Peak absolute value for relative slope
    peak_abs = max(abs(v) for v in vals)
    slope_pct = (slope / peak_abs * 100) if peak_abs > 0 else 0.0

    # Best value (highest)
    best_idx = max(range(n), key=lambda i: vals[i])
    best_iter = iters[best_idx]
    best_val = vals[best_idx]

    # Inflection point: where slope changes sign using rolling window
    # Use a window of max(20, n//10) to smooth noise
    window = max(20, n // 10)
    inflection_iter = 0
    if n > window * 2:
        # Compute rolling slope in two halves
        prev_slope_sign = None
        for i in range(window, n - window, window // 2):
            seg_x = [iters[j] for j in range(i - window, i)]
            seg_y = [vals[j] for j in range(i - window, i)]
            seg_mx = sum(seg_x) / len(seg_x)
            seg_my = sum(seg_y) / len(seg_y)
            seg_ssxx = sum((x - seg_mx) ** 2 for x in seg_x)
            seg_ssxy = sum((x - seg_mx) * (y - seg_my) for x, y in zip(seg_x, seg_y))
            seg_slope = seg_ssxy / seg_ssxx if seg_ssxx > 0 else 0.0
            cur_sign = 1 if seg_slope > 0 else (-1 if seg_slope < 0 else 0)
            if prev_slope_sign is not None and cur_sign != prev_slope_sign and cur_sign != 0:
                inflection_iter = iters[i]
                break
            if cur_sign != 0:
                prev_slope_sign = cur_sign

    return TrendResult(
        slope=slope,
        slope_pct=slope_pct,
        inflection_iter=inflection_iter,
        best_iter=best_iter,
        best_val=best_val,
        latest_val=vals[-1],
        latest_iter=iters[-1],
        n_points=n,
    )


def compute_all_trends(state: RunState) -> dict[str, TrendResult]:
    """Compute trend metrics for all 8 tracked metrics."""
    metrics_map = {
        "reward": state.rewards,
        "action_rate": state.action_rates,
        "value_loss": state.value_losses,
        "entropy": state.entropies,
        "episode_length": state.episode_lengths,
        "time_out": state.time_outs,
        "bad_orientation": state.bad_orientations,
        "vel_error": state.vel_errors,
    }
    return {name: compute_trend(data) for name, data in metrics_map.items()}


# --------------------------------------------------------------------------- #
# 2. Failure Mode Classification                                               #
# --------------------------------------------------------------------------- #


@dataclass
class FailureMode:
    mode_id: str
    name: str
    severity: str  # CRITICAL, HIGH, MODERATE
    evidence: str  # human-readable explanation


def _median(data: list[float]) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    return s[len(s) // 2]


def _latest_or(data: list[tuple[int, float]], default: float = 0.0) -> float:
    return data[-1][1] if data else default


def _latest_median(data: list[tuple[int, float]], window: int = 10) -> float:
    if not data:
        return 0.0
    recent = [v for _, v in data[-window:]]
    return _median(recent)


def classify_failure(
    state: RunState,
    trends: dict[str, TrendResult],
    terrain: str,
) -> list[FailureMode]:
    """Classify failure modes based on metric patterns.

    Returns list of FailureMode sorted by severity (CRITICAL first).
    A run can match multiple failure modes.
    """
    failures: list[FailureMode] = []

    # Extract key values
    reward_trend = trends.get("reward", TrendResult())
    ar_trend = trends.get("action_rate", TrendResult())
    ent_trend = trends.get("entropy", TrendResult())
    vl_trend = trends.get("value_loss", TrendResult())
    ep_trend = trends.get("episode_length", TrendResult())
    to_trend = trends.get("time_out", TrendResult())
    bo_trend = trends.get("bad_orientation", TrendResult())
    ve_trend = trends.get("vel_error", TrendResult())

    time_out_pct = _latest_median(state.time_outs) * 100
    bad_ori_pct = _latest_median(state.bad_orientations) * 100
    vel_err = _latest_median(state.vel_errors)
    ep_len = _latest_median(state.episode_lengths)
    reward_latest = _latest_or(state.rewards)
    ar_latest = _latest_median(state.action_rates)
    ent_latest = _latest_or(state.entropies)
    ent_peak = state.peak_entropy if state.peak_entropy > 0 else 1.0
    vl_latest = _latest_median(state.value_losses, window=5)

    # --- POLICY_COLLAPSE (CRITICAL) ---
    # bad_ori > 40%, reward < 0, time_out < 70%
    if bad_ori_pct > 40 and reward_latest < 0 and time_out_pct < 70:
        failures.append(FailureMode(
            mode_id="POLICY_COLLAPSE",
            name="策略崩溃",
            severity="CRITICAL",
            evidence=f"bad_ori={bad_ori_pct:.0f}%, reward={reward_latest:.2f}, time_out={time_out_pct:.0f}%, ep_len={ep_len:.0f}",
        ))

    # --- ACTION_EXPLOSION (CRITICAL) ---
    # action_rate < -5.0, ep_len dropping fast
    if ar_latest < -5.0 and ep_trend.slope < -1.0:
        failures.append(FailureMode(
            mode_id="ACTION_EXPLOSION",
            name="动作爆炸",
            severity="CRITICAL",
            evidence=f"action_rate={ar_latest:.2f}, ep_len_slope={ep_trend.slope:.2f}/iter",
        ))

    # --- VALUE_DIVERGE (HIGH) ---
    # value_loss > 100 and still rising
    if vl_latest > 100 and vl_trend.slope > 0:
        failures.append(FailureMode(
            mode_id="VALUE_DIVERGE",
            name="价值函数发散",
            severity="HIGH",
            evidence=f"value_loss={vl_latest:.1f}, slope={vl_trend.slope:.4f}/iter",
        ))

    # --- ENTROPY_COLLAPSE (HIGH) ---
    # entropy dropped > 90% from peak, absolute < 0.05
    if ent_peak > 0:
        ent_decline_pct = (ent_peak - ent_latest) / ent_peak * 100
        if ent_decline_pct > 90 and ent_latest < 0.05:
            failures.append(FailureMode(
                mode_id="ENTROPY_COLLAPSE",
                name="熵坍缩",
                severity="HIGH",
                evidence=f"entropy {ent_peak:.2f} -> {ent_latest:.4f} ({ent_decline_pct:.0f}% decline)",
            ))

    # --- ROUGH_TERRAIN_FAIL (HIGH) ---
    # bad_ori > 60%, vel_err > 0.5, terrain != flat
    if terrain != "flat" and bad_ori_pct > 60 and vel_err > 0.5:
        failures.append(FailureMode(
            mode_id="ROUGH_TERRAIN_FAIL",
            name="粗糙地形失败",
            severity="HIGH",
            evidence=f"bad_ori={bad_ori_pct:.0f}%, vel_err={vel_err:.2f}, terrain={terrain}",
        ))

    # --- HIGH_FALL_RATE (CRITICAL) ---
    # bad_ori > 80%, reward >= 0, time_out < 50%
    # Robot keeps falling but hasn't fully collapsed (reward still positive)
    if bad_ori_pct > 80 and reward_latest >= 0 and time_out_pct < 50:
        failures.append(FailureMode(
            mode_id="HIGH_FALL_RATE",
            name="高频摔倒",
            severity="CRITICAL",
            evidence=f"bad_ori={bad_ori_pct:.0f}%, reward={reward_latest:.2f}, time_out={time_out_pct:.0f}%, ep_len={ep_len:.0f}",
        ))

    # --- POLICY_UNSTABLE (HIGH) ---
    # bad_ori 30-60% or time_out < 70%, indicating policy hasn't converged well
    if 20 < bad_ori_pct <= 60 or (time_out_pct < 70 and bad_ori_pct > 15):
        failures.append(FailureMode(
            mode_id="POLICY_UNSTABLE",
            name="策略不稳定",
            severity="HIGH",
            evidence=f"bad_ori={bad_ori_pct:.0f}%, time_out={time_out_pct:.0f}%, ep_len={ep_len:.0f}",
        ))

    # --- REWARD_DECLINE (MODERATE) ---
    # reward dropped > 15% from peak (use peak vs latest, not slope)
    if state.peak_reward > 0:
        decline_from_peak = (
            (state.peak_reward - reward_latest) / abs(state.peak_reward) * 100
            if state.peak_reward != 0 else 0
        )
        if decline_from_peak > 20 and bad_ori_pct < 30:
            failures.append(FailureMode(
                mode_id="REWARD_DECLINE",
                name="奖励渐退",
                severity="MODERATE",
                evidence=(
                    f"reward {state.peak_reward:.2f}@{state.peak_reward_iter} -> "
                    f"{reward_latest:.2f} ({decline_from_peak:.1f}% decline), "
                    f"time_out={time_out_pct:.0f}%"
                ),
            ))

    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2}
    failures.sort(key=lambda f: severity_order.get(f.severity, 3))

    return failures


# --------------------------------------------------------------------------- #
# 3. Root Cause Location                                                        #
# --------------------------------------------------------------------------- #


@dataclass
class RootCause:
    first_sign: str           # metric name that first showed trouble
    inflection_iter: int      # when it started going wrong
    timeline: str             # human-readable timeline string
    details: dict[str, int]   # metric -> inflection_iter


def locate_root_cause(
    state: RunState,
    trends: dict[str, TrendResult],
    failures: list[FailureMode],
) -> RootCause:
    """Find the first metric to go wrong by comparing inflection points.

    Examines reward, entropy, action_rate, and bad_orientation inflection
    points to construct a degradation timeline.
    """
    # Key metrics to check for early signs
    key_metrics = {
        "reward": trends.get("reward", TrendResult()),
        "entropy": trends.get("entropy", TrendResult()),
        "action_rate": trends.get("action_rate", TrendResult()),
        "bad_orientation": trends.get("bad_orientation", TrendResult()),
    }

    # Collect inflection points (where metric starts declining)
    inflections: dict[str, int] = {}
    for name, trend in key_metrics.items():
        if trend.inflection_iter > 0:
            inflections[name] = trend.inflection_iter
        elif name == "reward" and state.peak_reward_iter > 0:
            # If no inflection detected, use peak as starting point
            inflections[name] = state.peak_reward_iter

    # Find the earliest inflection
    if not inflections:
        return RootCause(
            first_sign="unknown",
            inflection_iter=0,
            timeline="Insufficient data for root cause analysis",
            details={},
        )

    # Sort by iteration
    sorted_inflections = sorted(inflections.items(), key=lambda x: x[1])
    first_sign = sorted_inflections[0][0]
    first_iter = sorted_inflections[0][1]

    # Build timeline
    events = []
    for metric_name, iter_num in sorted_inflections:
        if metric_name == "reward":
            events.append(f"reward peak @{iter_num}")
        elif metric_name == "entropy":
            events.append(f"entropy拐点 @{iter_num}")
        elif metric_name == "action_rate":
            events.append(f"action_rate恶化 @{iter_num}")
        elif metric_name == "bad_orientation":
            events.append(f"bad_ori上升 @{iter_num}")
        else:
            events.append(f"{metric_name}变化 @{iter_num}")

    # Add latest status
    if state.rewards:
        latest_iter = state.rewards[-1][0]
        latest_reward = state.rewards[-1][1]
        if state.peak_reward > 0:
            decline_pct = (state.peak_reward - latest_reward) / abs(state.peak_reward) * 100
            events.append(f"reward -{decline_pct:.0f}% @{latest_iter}")

    timeline = " -> ".join(events)

    return RootCause(
        first_sign=first_sign,
        inflection_iter=first_iter,
        timeline=timeline,
        details=inflections,
    )


# --------------------------------------------------------------------------- #
# 4. Parameter Recommendations                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class Recommendation:
    priority: int
    action: str            # short action name
    parameter: str         # parameter to change
    current: str           # current value (as string, may be "N/A")
    suggested: str         # suggested value (as string)
    reason: str            # why this change helps


def generate_recommendations(
    failures: list[FailureMode],
    state: RunState,
    trends: dict[str, TrendResult],
) -> list[Recommendation]:
    """Generate actionable parameter tuning suggestions based on failure modes."""
    recs: list[Recommendation] = []
    priority = 1
    failure_ids = {f.mode_id for f in failures}

    # --- Best model recommendation (always for overfitting) ---
    if state.overfitting_detected and state.best_model_iter > 0:
        recs.append(Recommendation(
            priority=priority,
            action="early_stop_use_best",
            parameter="best_model",
            current="N/A",
            suggested=f"model_{state.best_model_iter}.pt",
            reason=(
                f"Best model at iter {state.best_model_iter} "
                f"(reward: {state.best_model_reward:.2f}), "
                f"之后reward持续下降"
            ),
        ))
        priority += 1

    # --- Mode-specific recommendations ---
    if "REWARD_DECLINE" in failure_ids:
        recs.append(Recommendation(
            priority=priority,
            action="increase_entropy_coef",
            parameter="entropy_coef",
            current="0.01",
            suggested="0.02",
            reason="策略过度确定导致reward渐退，增加探索防止早熟",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="reduce_learning_rate",
            parameter="learning_rate",
            current="1e-3",
            suggested="5e-4",
            reason="reward下降速率较慢但持续，可能lr过大导致过拟合",
        ))
        priority += 1

    if "POLICY_COLLAPSE" in failure_ids:
        recs.append(Recommendation(
            priority=priority,
            action="increase_orientation_penalty",
            parameter="flat_orientation_weight",
            current="-5.0",
            suggested="-8.0",
            reason="姿态约束不足导致策略崩溃，需更严格的姿态惩罚",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="increase_height_penalty",
            parameter="base_height_weight",
            current="-10.0",
            suggested="-15.0",
            reason="高度约束不足，增大惩罚维持稳定站立",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="reduce_domain_rand",
            parameter="push_force_range",
            current="N/A",
            suggested="减小幅度",
            reason="域随机化力度可能过大导致训练不稳定",
        ))
        priority += 1

    if "ACTION_EXPLOSION" in failure_ids:
        recs.append(Recommendation(
            priority=priority,
            action="increase_action_rate_penalty",
            parameter="action_rate_weight",
            current="-0.05",
            suggested="-0.08 ~ -0.1",
            reason="动作速率惩罚不足导致高频抖动",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="increase_energy_penalty",
            parameter="energy_weight",
            current="-2e-5",
            suggested="-5e-5",
            reason="增大能量惩罚抑制过大的关节力矩",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="reduce_init_noise",
            parameter="init_noise_std",
            current="1.0",
            suggested="0.5",
            reason="初始噪声过大可能导致早期动作爆炸",
        ))
        priority += 1

    if "ENTROPY_COLLAPSE" in failure_ids:
        recs.append(Recommendation(
            priority=priority,
            action="increase_entropy_coef",
            parameter="entropy_coef",
            current="0.01",
            suggested="0.02-0.05",
            reason="熵下降过多，策略过早收敛到次优解",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="reduce_num_epochs",
            parameter="num_learning_epochs",
            current="5",
            suggested="3",
            reason="减少每轮更新次数，防止同一批数据过拟合",
        ))
        priority += 1

    if "VALUE_DIVERGE" in failure_ids:
        recs.append(Recommendation(
            priority=priority,
            action="reduce_lr_for_value",
            parameter="learning_rate",
            current="1e-3",
            suggested="1e-4",
            reason="价值函数发散，大幅降低学习率稳定训练",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="increase_max_grad_norm",
            parameter="max_grad_norm",
            current="1.0",
            suggested="2.0",
            reason="梯度裁剪过严可能导致价值函数不稳定",
        ))
        priority += 1

    if "POLICY_UNSTABLE" in failure_ids:
        recs.append(Recommendation(
            priority=priority,
            action="increase_orientation_penalty",
            parameter="flat_orientation_weight",
            current="-5.0",
            suggested="-8.0",
            reason="策略不稳定，姿态约束不足，需更严格的姿态惩罚",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="reduce_training_difficulty",
            parameter="general",
            current="N/A",
            suggested="减小地形难度或增大alive奖励",
            reason="策略尚未收敛到稳定行走，降低训练难度先建立基础能力",
        ))
        priority += 1

    if "ROUGH_TERRAIN_FAIL" in failure_ids:
        recs.append(Recommendation(
            priority=priority,
            action="reduce_terrain_difficulty",
            parameter="difficulty_range",
            current="(0, 1.0)",
            suggested="(0, 0.5)",
            reason="地形难度过高，减小难度范围让策略逐步适应",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="increase_flat_ratio",
            parameter="terrain_curriculum",
            current="N/A",
            suggested="增加平坦地形比例",
            reason="粗糙地形占比过高导致训练不稳定，应先在平地建立基础策略",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="increase_alive_bonus",
            parameter="alive_reward_weight",
            current="N/A",
            suggested="增大",
            reason="增大存活奖励鼓励机器人保持站立",
        ))
        priority += 1

    if "HIGH_FALL_RATE" in failure_ids:
        recs.append(Recommendation(
            priority=priority,
            action="increase_orientation_penalty",
            parameter="flat_orientation_weight",
            current="-5.0",
            suggested="-10.0 ~ -15.0",
            reason="机器人持续摔倒但奖励为正，需大幅增加姿态惩罚使站立成为优先目标",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="increase_height_penalty",
            parameter="base_height_weight",
            current="-10.0",
            suggested="-15.0 ~ -20.0",
            reason="增大高度惩罚防止机器人趴下，保持站立姿态",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="tighten_bad_ori_threshold",
            parameter="bad_orientation_limit_angle",
            current="0.8",
            suggested="0.5",
            reason="降低bad_orientation触发阈值，让惩罚更早介入防止摔倒",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="resume_from_good_checkpoint",
            parameter="checkpoint",
            current="current",
            suggested="从已知良好checkpoint恢复",
            reason="策略已偏离稳定区域，建议从之前表现好的checkpoint恢复训练",
        ))
        priority += 1

        recs.append(Recommendation(
            priority=priority,
            action="reduce_curriculum_speed",
            parameter="general",
            current="N/A",
            suggested="降低课程学习推进速度",
            reason="训练推进过快导致策略来不及适应，应放缓课程进度",
        ))
        priority += 1

    # If no specific failure mode matched but overfitting was detected
    if not failure_ids and state.overfitting_detected:
        recs.append(Recommendation(
            priority=priority,
            action="general_overfitting",
            parameter="general",
            current="N/A",
            suggested="降低学习率 + 增大entropy_coef",
            reason="过拟合但未匹配特定失败模式，尝试通用抗过拟合策略",
        ))

    return recs


# --------------------------------------------------------------------------- #
# 5. Baseline Comparison                                                       #
# --------------------------------------------------------------------------- #


@dataclass
class BaselineComparison:
    baseline_run: str
    reward_gap: float
    time_out_gap: str     # percentage points
    bad_ori_gap: str      # percentage points
    vel_err_gap: float
    action_rate_gap: float


def find_baseline_dir(log_root: str, baseline_name: str = "s2_gentle") -> Optional[str]:
    """Search log_root for a run directory containing the baseline name."""
    root = Path(log_root)
    if not root.is_dir():
        return None
    for d in root.iterdir():
        if d.is_dir() and baseline_name in d.name:
            return str(d)
    return None


def load_baseline_state(baseline_dir: str) -> Optional[RunState]:
    """Load RunState for the baseline run from TensorBoard data."""
    event_dir = TensorBoardParser.find_event_file(baseline_dir)
    if not event_dir:
        return None

    try:
        metrics = TensorBoardParser.read_metrics(event_dir)
        run_name = Path(baseline_dir).name
        state = RunState(run_name=run_name, run_dir=baseline_dir)
        state.rewards = metrics.get(TAG_REWARD, [])
        state.action_rates = metrics.get(TAG_ACTION_RATE, [])
        state.value_losses = metrics.get(TAG_VALUE_LOSS, [])
        state.entropies = metrics.get(TAG_ENTROPY, [])
        state.episode_lengths = metrics.get(TAG_EP_LEN, [])
        state.time_outs = metrics.get(TAG_TIME_OUT, [])
        state.bad_orientations = metrics.get(TAG_BAD_ORI, [])
        state.vel_errors = metrics.get(TAG_VEL_ERR, [])

        # Compute peak reward
        for step, reward in state.rewards:
            if reward > state.peak_reward:
                state.peak_reward = reward
                state.peak_reward_iter = step
        for _, ent in state.entropies:
            if ent > state.peak_entropy:
                state.peak_entropy = ent

        tracker = BestModelTracker(window=10)
        tracker.update(state)
        return state
    except Exception:
        return None


def compare_baseline(
    state: RunState,
    baseline_state: RunState,
) -> BaselineComparison:
    """Compare current run against baseline run."""
    cur_reward = _latest_or(state.rewards)
    bl_reward = _latest_or(baseline_state.rewards)
    reward_gap = round(cur_reward - bl_reward, 1)

    cur_to = _latest_median(state.time_outs) * 100
    bl_to = _latest_median(baseline_state.time_outs) * 100
    time_out_gap = f"{cur_to - bl_to:+.1f}%"

    cur_bo = _latest_median(state.bad_orientations) * 100
    bl_bo = _latest_median(baseline_state.bad_orientations) * 100
    bad_ori_gap = f"{cur_bo - bl_bo:+.1f}%"

    cur_ve = _latest_median(state.vel_errors)
    bl_ve = _latest_median(baseline_state.vel_errors)
    vel_err_gap = round(cur_ve - bl_ve, 2)

    cur_ar = _latest_median(state.action_rates)
    bl_ar = _latest_median(baseline_state.action_rates)
    action_rate_gap = round(cur_ar - bl_ar, 3)

    return BaselineComparison(
        baseline_run=baseline_state.run_name,
        reward_gap=reward_gap,
        time_out_gap=time_out_gap,
        bad_ori_gap=bad_ori_gap,
        vel_err_gap=vel_err_gap,
        action_rate_gap=action_rate_gap,
    )


# --------------------------------------------------------------------------- #
# 6. Main Analysis Pipeline                                                    #
# --------------------------------------------------------------------------- #


def analyze_run(
    run_dir: str,
    terrain: str,
    baseline_dir: Optional[str] = None,
    json_output: bool = False,
) -> dict:
    """Full analysis pipeline for a single training run.

    Returns a structured dict with all analysis results.
    """
    run_name = Path(run_dir).name

    # -- Load TensorBoard data -- #
    event_dir = TensorBoardParser.find_event_file(run_dir)
    state = RunState(run_name=run_name, run_dir=run_dir)

    if event_dir:
        metrics = TensorBoardParser.read_metrics(event_dir)
        state.rewards = metrics.get(TAG_REWARD, [])
        state.action_rates = metrics.get(TAG_ACTION_RATE, [])
        state.value_losses = metrics.get(TAG_VALUE_LOSS, [])
        state.entropies = metrics.get(TAG_ENTROPY, [])
        state.episode_lengths = metrics.get(TAG_EP_LEN, [])
        state.time_outs = metrics.get(TAG_TIME_OUT, [])
        state.bad_orientations = metrics.get(TAG_BAD_ORI, [])
        state.vel_errors = metrics.get(TAG_VEL_ERR, [])
    else:
        if not json_output:
            print(f"[WARN] No TensorBoard events found in {run_dir}")

    # -- Update tracking -- #
    for step, reward in state.rewards:
        if reward > state.peak_reward:
            state.peak_reward = reward
            state.peak_reward_iter = step
    for _, ent in state.entropies:
        if ent > state.peak_entropy:
            state.peak_entropy = ent

    tracker = BestModelTracker(window=10)
    tracker.update(state)

    # -- Overfitting detection (reuse from train_monitor) -- #
    cfg = MonitorConfig(terrain_type=terrain)
    detector = OverfittingDetector(cfg)
    reason = detector.check(state)
    if reason:
        state.overfitting_detected = True
        state.overfitting_reason = reason

    # -- Step 1: Trend analysis -- #
    trends = compute_all_trends(state)
    trend_summary = {}
    for name, trend in trends.items():
        trend_summary[name] = {
            "slope": round(trend.slope, 6),
            "slope_pct": round(trend.slope_pct, 4),
            "inflection_iter": trend.inflection_iter,
            "best_iter": trend.best_iter,
            "best_val": round(trend.best_val, 4),
            "latest_val": round(trend.latest_val, 4),
            "latest_iter": trend.latest_iter,
        }

    # -- Step 2: Failure classification -- #
    # Skip runs with no meaningful data
    if not state.rewards or len(state.rewards) < 3:
        if not json_output:
            print(f"[SKIP] Insufficient data in {run_name} ({len(state.rewards)} points)")
        return {"run_name": run_name, "overall_status": "SKIP", "skip_reason": "no_data"}
    failures = classify_failure(state, trends, terrain)

    # Determine overall severity
    if failures:
        severity = failures[0].severity
    else:
        severity = "HEALTHY"

    # -- Step 3: Root cause -- #
    root_cause = locate_root_cause(state, trends, failures)

    # -- Step 4: Recommendations -- #
    recommendations = generate_recommendations(failures, state, trends)

    # -- Step 5: Baseline comparison -- #
    baseline_comparison = None
    baseline_state = None
    if baseline_dir:
        baseline_state = load_baseline_state(baseline_dir)
    if baseline_state:
        comp = compare_baseline(state, baseline_state)
        baseline_comparison = {
            "baseline_run": comp.baseline_run,
            "reward_gap": comp.reward_gap,
            "time_out_gap": comp.time_out_gap,
            "bad_ori_gap": comp.bad_ori_gap,
            "vel_err_gap": comp.vel_err_gap,
            "action_rate_gap": comp.action_rate_gap,
        }

    # -- Build output -- #
    overall_status = "OVERFITTING" if state.overfitting_detected else "HEALTHY"
    if failures:
        overall_status = failures[0].severity + " " + overall_status

    result = {
        "run_name": run_name,
        "overall_status": overall_status,
        "failure_modes": [
            {
                "id": f.mode_id,
                "name": f.name,
                "severity": f.severity,
                "evidence": f.evidence,
            }
            for f in failures
        ],
        "severity": severity,
        "root_cause": {
            "first_sign": root_cause.first_sign,
            "inflection_iter": root_cause.inflection_iter,
            "timeline": root_cause.timeline,
        },
        "trend_analysis": trend_summary,
        "recommendations": [
            {
                "priority": r.priority,
                "action": r.action,
                "parameter": r.parameter,
                "current": r.current,
                "suggested": r.suggested,
                "reason": r.reason,
            }
            for r in recommendations
        ],
        "baseline_comparison": baseline_comparison,
        # Additional summary fields for display
        "summary": {
            "latest_iter": trends.get("reward", TrendResult()).latest_iter,
            "peak_reward": round(state.peak_reward, 2),
            "peak_reward_iter": state.peak_reward_iter,
            "best_model_iter": state.best_model_iter,
            "best_model_reward": round(state.best_model_reward, 2),
            "latest_reward": round(_latest_or(state.rewards), 2),
            "time_out_pct": round(_latest_median(state.time_outs) * 100, 1),
            "bad_ori_pct": round(_latest_median(state.bad_orientations) * 100, 1),
            "vel_err": round(_latest_median(state.vel_errors), 2),
            "action_rate": round(_latest_median(state.action_rates), 3),
            "entropy": round(_latest_or(state.entropies), 2),
            "value_loss": round(_latest_median(state.value_losses, 5), 2),
        },
    }

    return result


def print_human_report(result: dict) -> None:
    """Print a human-readable analysis report."""
    run_name = result["run_name"]
    summary = result["summary"]
    failures = result["failure_modes"]

    print()
    print(f"=== Z1 Training Analysis: {run_name} ===")
    print()
    print(f"  Status: {result['overall_status']}")
    if failures:
        mode_strs = [f"{f['id']} ({f['name']})" for f in failures]
        print(f"  Failure Modes: {', '.join(mode_strs)}")
    print(f"  Root Cause: {result['root_cause']['first_sign']} 先恶化 @iter {result['root_cause']['inflection_iter']}")
    print()
    print(f"  Timeline:")
    print(f"    {result['root_cause']['timeline']}")
    print()

    # Trend section
    trends = result["trend_analysis"]
    r_trend = trends.get("reward", {})
    e_trend = trends.get("entropy", {})
    to_trend = trends.get("time_out", {})
    bo_trend = trends.get("bad_orientation", {})

    print("  Trend (last phase):")
    print(f"    reward:      {r_trend.get('best_val', 0):.1f} -> {r_trend.get('latest_val', 0):.1f}  "
          f"(slope={r_trend.get('slope', 0):.4f}/iter)")
    print(f"    entropy:     {summary['entropy']:.1f}")
    print(f"    time_out:    {summary['time_out_pct']:.0f}%")
    print(f"    bad_ori:     {summary['bad_ori_pct']:.0f}%")
    print()

    # Baseline comparison
    bc = result.get("baseline_comparison")
    if bc:
        print(f"  vs Baseline ({bc['baseline_run']}):")
        print(f"    reward:     {summary['latest_reward']:.1f} vs baseline  (gap: {bc['reward_gap']:+.1f})")
        print(f"    time_out:   {summary['time_out_pct']:.0f}% vs baseline  (gap: {bc['time_out_gap']})")
        print(f"    bad_ori:    {summary['bad_ori_pct']:.0f}% vs baseline  (gap: {bc['bad_ori_gap']})")
        print()

    # Recommendations
    recs = result["recommendations"]
    if recs:
        print("  === Recommendations ===")
        print()
        for r in recs:
            print(f"  [{r['priority']}] {r['parameter']}: {r['current']} -> {r['suggested']}")
            print(f"      Reason: {r['reason']}")
            print()
    else:
        print("  No specific recommendations (run appears healthy)")
        print()


# --------------------------------------------------------------------------- #
# 7. CLI Entry Point                                                           #
# --------------------------------------------------------------------------- #


def parse_args():
    parser = argparse.ArgumentParser(
        description="Training Failure Analyzer for MagicBot Z1 RL Training",
    )

    # Mode
    parser.add_argument(
        "--all", action="store_true",
        help="Analyze all failed/overfitted runs under log_root",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output only JSON (no human-readable report)",
    )

    # Paths
    parser.add_argument(
        "--log_root", type=str,
        default="logs/rsl_rl/magiclab_z1_12dof_velocity",
        help="Root directory containing training runs",
    )
    parser.add_argument(
        "--run_dir", type=str, default=None,
        help="Single run directory to analyze",
    )

    # Terrain
    parser.add_argument(
        "--terrain", type=str, default="gentle",
        choices=["flat", "gentle", "rough"],
        help="Terrain type (adjusts thresholds)",
    )

    # Baseline
    parser.add_argument(
        "--baseline", type=str, default="s2_gentle",
        help="Baseline run name substring (default: s2_gentle)",
    )
    parser.add_argument(
        "--baseline_dir", type=str, default=None,
        help="Explicit baseline run directory path",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.run_dir:
        # Single run mode
        baseline_dir = args.baseline_dir
        if not baseline_dir and args.baseline:
            baseline_dir = find_baseline_dir(args.log_root, args.baseline)

        result = analyze_run(
            run_dir=args.run_dir,
            terrain=args.terrain,
            baseline_dir=baseline_dir,
            json_output=args.json,
        )
        results = [result]

    elif args.all:
        # All runs mode
        run_dirs = find_run_dirs(args.log_root)
        if not run_dirs:
            print("[ERROR] No run directories found", file=sys.stderr)
            sys.exit(1)

        baseline_dir = args.baseline_dir
        if not baseline_dir and args.baseline:
            baseline_dir = find_baseline_dir(args.log_root, args.baseline)

        results = []
        for rd in run_dirs:
            try:
                result = analyze_run(
                    run_dir=rd,
                    terrain=args.terrain,
                    baseline_dir=baseline_dir,
                    json_output=args.json,
                )
                results.append(result)
            except Exception as e:
                print(f"[WARN] Failed to analyze {rd}: {e}", file=sys.stderr)

        # Filter to only failed/overfitted runs (skip empty ones)
        failed = [r for r in results if r.get("overall_status") != "SKIP" and r["failure_modes"]]
        if not failed:
            if args.json:
                print(json.dumps({"status": "all_healthy", "total_runs": len(results)}, indent=2))
            else:
                print(f"[ANALYZER] All {len(results)} runs are healthy. No failures to analyze.")
            return
        results = failed

    else:
        print("[ERROR] Specify --run_dir <DIR> or --all", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2, ensure_ascii=False))
    else:
        for r in results:
            print_human_report(r)

        # Summary
        if len(results) > 1:
            print("=" * 60)
            print(f"  Summary: {len(results)} failed runs analyzed")
            # Count failure modes
            mode_counts: dict[str, int] = {}
            for r in results:
                for fm in r["failure_modes"]:
                    mode_counts[fm["id"]] = mode_counts.get(fm["id"], 0) + 1
            if mode_counts:
                print("  Failure mode distribution:")
                for mid, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
                    print(f"    {mid}: {count} runs")
            print()


if __name__ == "__main__":
    main()
