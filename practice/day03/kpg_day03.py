"""KPG-193 v2.0 부하 증감 AC 조류계산 실습.

기본 운전점의 양(+) 부하만 지정한 비율로 변경하고 AC-PF를 반복한다.
Pd와 Qd를 같은 비율로 조정해 각 Bus 부하의 역률은 유지하며,
음수 Pd로 모델링된 고정 발전은 변경하지 않는다. 기본 모드에서는
비-Slack 발전기의 남은 P 여유용량에 비례해 부하 변화분을 분담한다.

주의: 비용 기반 발전기 재급전이나 OPF는 수행하지 않는다. 발전기
여유용량이 부족한 잔여분과 계통 손실은 기준(REF/Slack) Bus 발전기가
맞추므로, 이 결과는 정적 민감도 분석이며 실제 운전 가능성을 보장하지
않는다.

예:
    python kpg_day03.py
    python kpg_day03.py --scales 90 100 105 110 120
    python kpg_day03.py --generation-mode slack
    python kpg_day03.py --output kpg_day03_results.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pypower.api import ppoption, runpf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from day01.kpg_day01 import (
    BUS_METADATA_FILE,
    CASE_FILE,
    add_fixed_hvdc_as_dummy_generators,
    load_matpower_matrix,
)


# MATPOWER/PYPOWER 열 번호를 Python 0-based index로 표현한다.
BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
VM = 7
VMAX = 11
VMIN = 12
REF = 3

GEN_BUS = 0
PG = 1
QG = 2
QMAX = 3
QMIN = 4
GEN_STATUS = 7
PMAX = 8
PMIN = 9

F_BUS = 0
T_BUS = 1
RATE_A = 5
BR_STATUS = 10
PF = 13
QF = 14
PT = 15
QT = 16

EPSILON = 1e-6
DEFAULT_SCALES = [100.0, 105.0, 110.0, 120.0, 125.0, 130.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KPG-193 부하 증감 AC 조류계산",
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
        help=(
            "기본 양(+) 부하에 적용할 백분율 목록 "
            "(기본값: 100 105 110 120 125 130)"
        ),
    )
    parser.add_argument(
        "--generation-mode",
        choices=("headroom", "slack"),
        default="headroom",
        help=(
            "headroom: 비-Slack 발전기 P 여유용량으로 부하 변화분 분담, "
            "slack: 발전기 설정을 유지하고 Slack이 전부 부담 "
            "(기본값: headroom)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "kpg_day03_load_scenarios.csv",
        help="시나리오별 결과 CSV 경로",
    )
    return parser.parse_args()


def load_bus_names(metadata_file: Path) -> dict[int, str]:
    """현재 kpg_day01.py와 같은 메타데이터에서 Bus 이름을 읽는다."""

    metadata = pd.read_csv(metadata_file, encoding="utf-8-sig")
    return {
        int(bus_id): str(name)
        for bus_id, name in zip(
            metadata["bus_id"],
            metadata["name_Korean"],
        )
    }


def make_case(
    bus: np.ndarray,
    gen: np.ndarray,
    branch: np.ndarray,
) -> dict[str, object]:
    return {
        "version": "2",
        "baseMVA": 100.0,
        "bus": bus.copy(),
        "gen": gen.copy(),
        "branch": branch.copy(),
    }


def run_ac_power_flow(
    bus: np.ndarray,
    gen: np.ndarray,
    branch: np.ndarray,
) -> tuple[dict[str, np.ndarray] | None, bool, str]:
    """AC-PF를 실행하고 경고·예외를 시나리오 결과로 돌려준다."""

    options = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        PF_ALG=1,
        ENFORCE_Q_LIMS=0,
    )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results, converged = runpf(make_case(bus, gen, branch), options)
        messages = [str(item.message) for item in caught]
        return results, bool(converged), " | ".join(dict.fromkeys(messages))
    except Exception as error:
        return None, False, f"{type(error).__name__}: {error}"


def scale_positive_loads(bus: np.ndarray, scale_pct: float) -> np.ndarray:
    """양(+) Pd와 그 Bus의 Qd만 같은 비율로 조정한다."""

    scaled = bus.copy()
    positive_load = bus[:, PD] > 0.0
    factor = scale_pct / 100.0
    scaled[positive_load, PD] = bus[positive_load, PD] * factor
    scaled[positive_load, QD] = bus[positive_load, QD] * factor
    return scaled


def schedule_generation(
    base_bus: np.ndarray,
    scenario_bus: np.ndarray,
    gen: np.ndarray,
    original_gen_count: int,
    generation_mode: str,
) -> tuple[np.ndarray, float, float]:
    """비-Slack 발전기에 단순 참여규칙을 적용한다.

    반환값은 변경된 gen, 비-Slack 발전기 조정량, Slack이 손실과 함께
    추가로 맞춰야 할 부하 변화량이다. 발전비용은 사용하지 않는다.
    """

    scenario_gen = gen.copy()
    demand_change = float(
        scenario_bus[:, PD].sum() - base_bus[:, PD].sum()
    )
    if generation_mode == "slack" or abs(demand_change) <= EPSILON:
        return scenario_gen, 0.0, demand_change

    conventional = scenario_gen[:original_gen_count]
    active = conventional[:, GEN_STATUS] > 0
    ref_bus_ids = set(
        int(value)
        for value in base_bus[base_bus[:, BUS_TYPE] == REF, BUS_I]
    )
    is_slack = np.asarray(
        [int(value) in ref_bus_ids for value in conventional[:, GEN_BUS]],
        dtype=bool,
    )
    participants = active & ~is_slack

    if demand_change > 0:
        capability = np.maximum(
            conventional[:, PMAX] - conventional[:, PG],
            0.0,
        )
        requested = demand_change
        direction = 1.0
    else:
        capability = np.maximum(
            conventional[:, PG] - conventional[:, PMIN],
            0.0,
        )
        requested = -demand_change
        direction = -1.0

    capability[~participants] = 0.0
    total_capability = float(capability.sum())
    scheduled_magnitude = min(requested, total_capability)
    if total_capability > EPSILON:
        conventional[:, PG] += (
            direction * scheduled_magnitude * capability / total_capability
        )

    scheduled_adjustment = direction * scheduled_magnitude
    remaining_for_slack = demand_change - scheduled_adjustment
    return scenario_gen, scheduled_adjustment, remaining_for_slack


def branch_loadings(result_branch: np.ndarray) -> np.ndarray:
    """양단 피상전력 중 큰 값을 RATE_A로 나눈 Branch 부하율(%)을 계산한다."""

    apparent_from = np.hypot(result_branch[:, PF], result_branch[:, QF])
    apparent_to = np.hypot(result_branch[:, PT], result_branch[:, QT])
    rating = result_branch[:, RATE_A]
    valid = (result_branch[:, BR_STATUS] > 0) & (rating > 0)

    loading = np.full(len(result_branch), np.nan)
    loading[valid] = (
        np.maximum(apparent_from[valid], apparent_to[valid])
        / rating[valid]
        * 100.0
    )
    return loading


def empty_result(scale_pct: float) -> dict[str, object]:
    return {
        "load_scale_pct": scale_pct,
        "generation_mode": "",
        "status": "",
        "converged": False,
        "network_limit_violation": False,
        "generation_limit_violation": False,
        "gross_load_mw": np.nan,
        "fixed_generation_mw": np.nan,
        "net_demand_mw": np.nan,
        "conventional_generation_mw": np.nan,
        "slack_generation_mw": np.nan,
        "slack_pmax_mw": np.nan,
        "scheduled_non_slack_adjustment_mw": np.nan,
        "remaining_for_slack_before_losses_mw": np.nan,
        "ac_loss_mw": np.nan,
        "max_loading_pct": np.nan,
        "worst_branch": "",
        "worst_from_bus": "",
        "worst_from_name": "",
        "worst_to_bus": "",
        "worst_to_name": "",
        "overloaded_branch_count": "",
        "min_voltage_pu": np.nan,
        "min_voltage_bus": "",
        "min_voltage_name": "",
        "voltage_violation_count": "",
        "generator_p_limit_violation_count": "",
        "generator_q_limit_violation_count": "",
        "message": "",
    }


def evaluate_scenario(
    scale_pct: float,
    base_bus: np.ndarray,
    gen: np.ndarray,
    branch: np.ndarray,
    original_gen_count: int,
    bus_names: dict[int, str],
    generation_mode: str,
) -> dict[str, object]:
    """한 개 부하 배율 시나리오의 AC-PF와 한계 위반을 계산한다."""

    row = empty_result(scale_pct)
    scenario_bus = scale_positive_loads(base_bus, scale_pct)
    scenario_gen, scheduled_adjustment, remaining_for_slack = schedule_generation(
        base_bus,
        scenario_bus,
        gen,
        original_gen_count,
        generation_mode,
    )

    row["generation_mode"] = generation_mode
    row["scheduled_non_slack_adjustment_mw"] = scheduled_adjustment
    row["remaining_for_slack_before_losses_mw"] = remaining_for_slack
    row["gross_load_mw"] = float(
        np.maximum(scenario_bus[:, PD], 0.0).sum()
    )
    row["fixed_generation_mw"] = float(
        -np.minimum(scenario_bus[:, PD], 0.0).sum()
    )
    row["net_demand_mw"] = float(scenario_bus[:, PD].sum())

    results, converged, message = run_ac_power_flow(
        scenario_bus,
        scenario_gen,
        branch,
    )
    row["message"] = message
    if results is None:
        row["status"] = "ERROR"
        return row
    if not converged:
        row["status"] = "NON_CONVERGED"
        return row

    result_bus = results["bus"]
    result_gen = results["gen"]
    result_branch = results["branch"]
    row["converged"] = True

    loading = branch_loadings(result_branch)
    valid_loading = np.flatnonzero(~np.isnan(loading))
    worst_index = int(valid_loading[np.argmax(loading[valid_loading])])
    worst = result_branch[worst_index]
    worst_from_bus = int(worst[F_BUS])
    worst_to_bus = int(worst[T_BUS])
    overloaded_count = int(np.sum(loading > 100.0 + EPSILON))

    voltages = result_bus[:, VM]
    lower_violation = np.maximum(result_bus[:, VMIN] - voltages, 0.0)
    upper_violation = np.maximum(voltages - result_bus[:, VMAX], 0.0)
    voltage_violation = np.maximum(lower_violation, upper_violation)
    voltage_violation_count = int(np.sum(voltage_violation > EPSILON))
    min_voltage_index = int(np.argmin(voltages))
    min_voltage_bus = int(result_bus[min_voltage_index, BUS_I])

    conventional = result_gen[:original_gen_count]
    active_conventional = conventional[:, GEN_STATUS] > 0
    p_violation = active_conventional & (
        (conventional[:, PG] > conventional[:, PMAX] + EPSILON)
        | (conventional[:, PG] < conventional[:, PMIN] - EPSILON)
    )
    q_violation = active_conventional & (
        (conventional[:, QG] > conventional[:, QMAX] + EPSILON)
        | (conventional[:, QG] < conventional[:, QMIN] - EPSILON)
    )

    ref_bus_ids = set(
        int(value)
        for value in result_bus[result_bus[:, BUS_TYPE] == REF, BUS_I]
    )
    slack_mask = active_conventional & np.asarray(
        [int(value) in ref_bus_ids for value in conventional[:, GEN_BUS]],
        dtype=bool,
    )

    total_active_generation = float(
        result_gen[result_gen[:, GEN_STATUS] > 0, PG].sum()
    )
    ac_loss_mw = total_active_generation - float(result_bus[:, PD].sum())

    network_violation = overloaded_count > 0 or voltage_violation_count > 0
    generation_violation = bool(np.any(p_violation) or np.any(q_violation))
    if network_violation and generation_violation:
        status = "NETWORK+GEN_LIMIT"
    elif network_violation:
        status = "NETWORK_LIMIT"
    elif generation_violation:
        status = "GEN_LIMIT"
    else:
        status = "OK"
    row.update(
        {
            "status": status,
            "network_limit_violation": network_violation,
            "generation_limit_violation": generation_violation,
            "conventional_generation_mw": float(
                conventional[active_conventional, PG].sum()
            ),
            "slack_generation_mw": float(conventional[slack_mask, PG].sum()),
            "slack_pmax_mw": float(conventional[slack_mask, PMAX].sum()),
            "ac_loss_mw": ac_loss_mw,
            "max_loading_pct": float(loading[worst_index]),
            "worst_branch": worst_index + 1,
            "worst_from_bus": worst_from_bus,
            "worst_from_name": bus_names.get(worst_from_bus, ""),
            "worst_to_bus": worst_to_bus,
            "worst_to_name": bus_names.get(worst_to_bus, ""),
            "overloaded_branch_count": overloaded_count,
            "min_voltage_pu": float(voltages[min_voltage_index]),
            "min_voltage_bus": min_voltage_bus,
            "min_voltage_name": bus_names.get(min_voltage_bus, ""),
            "voltage_violation_count": voltage_violation_count,
            "generator_p_limit_violation_count": int(np.sum(p_violation)),
            "generator_q_limit_violation_count": int(np.sum(q_violation)),
        }
    )
    return row


def write_csv(output_file: Path, rows: list[dict[str, object]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_number(value: object, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if np.isnan(numeric):
        return "-"
    return f"{numeric:.{digits}f}"


def print_scenario_table(rows: list[dict[str, object]]) -> None:
    print()
    print("[시나리오별 결과]")
    print(
        " 배율    총부하      최대부하율  과부하   최저전압  전압위반 "
        " Slack Pg      상태"
    )
    print("-" * 94)
    for row in rows:
        if row["converged"]:
            overload = str(row["overloaded_branch_count"])
            voltage_count = str(row["voltage_violation_count"])
        else:
            overload = "-"
            voltage_count = "-"
        print(
            f"{float(row['load_scale_pct']):5.1f}% "
            f"{display_number(row['gross_load_mw'], 1):>10} MW  "
            f"{display_number(row['max_loading_pct']):>10}%  "
            f"{overload:>5}개  "
            f"{display_number(row['min_voltage_pu'], 5):>8} pu  "
            f"{voltage_count:>5}개  "
            f"{display_number(row['slack_generation_mw'], 1):>9} MW  "
            f"{row['status']}"
        )


def print_findings(rows: list[dict[str, object]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["load_scale_pct"]))
    converged = [row for row in ordered if row["converged"]]
    violations = [
        row for row in ordered
        if row["network_limit_violation"]
    ]
    non_converged = [
        row for row in ordered
        if row["status"] in {"NON_CONVERGED", "ERROR"}
    ]

    print()
    print("[핵심 관찰]")
    if converged:
        highest = converged[-1]
        print(
            f"- 시험 범위에서 AC-PF가 수렴한 최고 부하 배율: "
            f"{float(highest['load_scale_pct']):.1f}%"
        )
    if violations:
        first = violations[0]
        print(
            f"- 처음 확인된 계통 한계 위반: {float(first['load_scale_pct']):.1f}% "
            f"(과부하 {int(first['overloaded_branch_count'])}개, "
            f"전압 위반 {int(first['voltage_violation_count'])}개)"
        )
        print(
            f"- 해당 시나리오의 최악 선로: B{int(first['worst_branch'])} "
            f"{first['worst_from_name']} → {first['worst_to_name']} "
            f"{float(first['max_loading_pct']):.2f}%"
        )
    else:
        print("- 시험한 배율에서는 선로·전압 한계 위반이 확인되지 않음")
    if non_converged:
        first = non_converged[0]
        print(
            f"- 처음 비수렴 또는 오류가 발생한 배율: "
            f"{float(first['load_scale_pct']):.1f}%"
        )

    p_limit_rows = [
        row for row in converged
        if int(row["generator_p_limit_violation_count"]) > 0
    ]
    q_limit_rows = [
        row for row in converged
        if int(row["generator_q_limit_violation_count"]) > 0
    ]
    if p_limit_rows:
        print(
            f"- 발전기 P 한계 초과가 처음 표시된 배율: "
            f"{float(p_limit_rows[0]['load_scale_pct']):.1f}%"
        )
    if q_limit_rows:
        print(
            f"- 발전기 Q 한계 초과가 처음 표시된 배율: "
            f"{float(q_limit_rows[0]['load_scale_pct']):.1f}%"
        )

    remaining_rows = [
        row for row in converged
        if float(row["remaining_for_slack_before_losses_mw"]) > EPSILON
    ]
    if remaining_rows:
        first = remaining_rows[0]
        print(
            f"- 비-Slack 발전기 P 여유용량으로 다 분담하지 못한 첫 배율: "
            f"{float(first['load_scale_pct']):.1f}% "
            f"(Slack 잔여분 {float(first['remaining_for_slack_before_losses_mw']):,.1f} MW + 손실)"
        )


def main() -> None:
    args = parse_args()
    if not args.scales:
        raise ValueError("--scales에 하나 이상의 배율이 필요합니다.")
    if any(scale <= 0 for scale in args.scales):
        raise ValueError("부하 배율은 0보다 커야 합니다.")

    scales = list(dict.fromkeys(float(value) for value in args.scales))
    case_text = CASE_FILE.read_text(encoding="utf-8")
    bus = load_matpower_matrix(case_text, "bus")
    conventional_gen = load_matpower_matrix(case_text, "gen")
    branch = load_matpower_matrix(case_text, "branch")
    dcline = load_matpower_matrix(case_text, "dcline")
    original_gen_count = len(conventional_gen)
    bus_names = load_bus_names(BUS_METADATA_FILE)

    bus, gen, active_dc_lines = add_fixed_hvdc_as_dummy_generators(
        bus.copy(),
        conventional_gen.copy(),
        dcline,
    )

    positive_load = bus[:, PD] > 0.0
    base_gross_load = float(bus[positive_load, PD].sum())
    fixed_generation = float(-np.minimum(bus[:, PD], 0.0).sum())

    print("=" * 94)
    print("KPG-193 v2.0 Day 3 - 부하 증가 AC 조류계산")
    print("=" * 94)
    print(
        f"기본 운전점: Bus {len(bus)} / 기존 발전기 {original_gen_count} / "
        f"AC Branch {len(branch)} / 운전 중 HVDC {active_dc_lines}"
    )
    print(
        f"기본 양(+) 부하 {base_gross_load:,.1f} MW / "
        f"고정 발전 {fixed_generation:,.1f} MW / 시나리오 {len(scales)}건"
    )
    if args.generation_mode == "headroom":
        print(
            "발전 분담: 비-Slack 발전기의 남은 P 여유용량 비례 "
            "/ 잔여분·손실은 Slack 담당"
        )
    else:
        print("발전 분담: 기존 발전기 설정 유지 / 부하 변화분·손실은 Slack 담당")
    print("변경 대상: 양(+) Pd와 해당 Bus의 Qd / 음수 Pd 고정 발전은 유지")

    rows: list[dict[str, object]] = []
    for scale_pct in scales:
        rows.append(
            evaluate_scenario(
                scale_pct,
                bus,
                gen,
                branch,
                original_gen_count,
                bus_names,
                args.generation_mode,
            )
        )

    write_csv(args.output, rows)
    print_scenario_table(rows)
    print_findings(rows)

    print()
    print("[해석 주의]")
    print(
        "이 발전 분담은 비용 최적화가 아닌 단순 참여규칙이며 Q 한계도 "
        "강제하지 않는다. 발전기 한계 초과가 표시된 시나리오는 AC-PF가 "
        "수렴했더라도 실제 운전 가능 상태로 보면 안 된다."
    )
    print(f"전체 결과 CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()
