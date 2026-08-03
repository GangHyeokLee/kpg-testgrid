"""KPG-193 v2.0 AC 선로 N-1 사고 스크리닝 실습.

기본 운전점에서 운전 중인 AC branch를 하나씩 개방한 뒤 AC 조류계산을
반복한다. 각 사고에 대해 계통 분리, 수렴 여부, 전압 한계 위반 및
잔존 선로 과부하를 확인하고 전체 결과를 CSV로 저장한다.

예:
    python kpg_day02.py
    python kpg_day02.py --top 10
    python kpg_day02.py --branch 264
    python kpg_day02.py --branch 264 --branch 43
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pypower.api import ppoption, runpf

from kpg_day01 import (
    BUS_METADATA_FILE,
    CASE_FILE,
    add_fixed_hvdc_as_dummy_generators,
    load_matpower_matrix,
)


# MATPOWER/PYPOWER 열 번호를 Python 0-based index로 표현한다.
BUS_I = 0
PD = 2
VM = 7
VMAX = 11
VMIN = 12

GEN_STATUS = 7

F_BUS = 0
T_BUS = 1
RATE_A = 5
BR_STATUS = 10
PF = 13
QF = 14
PT = 15
QT = 16

EPSILON = 1e-6


def load_bus_names(metadata_file: Path) -> dict[int, str]:
    """현재 kpg_day01.py와 같은 pandas 방식으로 Bus 이름을 읽는다."""

    metadata = pd.read_csv(metadata_file, encoding="utf-8-sig")
    return {
        int(bus_id): str(name)
        for bus_id, name in zip(
            metadata["bus_id"],
            metadata["name_Korean"],
        )
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KPG-193 AC 선로 N-1 사고 스크리닝",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="화면에 표시할 상위 사고 건수 (기본값: 10)",
    )
    parser.add_argument(
        "--branch",
        type=int,
        action="append",
        help=(
            "특정 Branch 번호만 계산한다. 여러 개면 옵션을 반복한다. "
            "생략하면 운전 중인 모든 AC Branch를 계산한다."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "kpg_day02_n-1_results.csv",
        help="전체 결과 CSV 경로",
    )
    return parser.parse_args()


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
    """AC 조류계산을 실행하고 경고·예외를 한 행의 문자열로 반환한다."""

    options = ppoption(
        VERBOSE=0,
        OUT_ALL=0,
        PF_ALG=1,
        ENFORCE_Q_LIMS=0,
    )

    caught_messages: list[str] = []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results, converged = runpf(make_case(bus, gen, branch), options)
        caught_messages = [str(item.message) for item in caught]
        return results, bool(converged), " | ".join(dict.fromkeys(caught_messages))
    except Exception as error:  # 사고별 실패를 전체 스크리닝 중단으로 전파하지 않는다.
        return None, False, f"{type(error).__name__}: {error}"


def branch_loadings(result_branch: np.ndarray) -> np.ndarray:
    """양단 피상전력 중 큰 값을 RATE_A로 나눈 Branch 부하율(%)을 계산한다."""

    apparent_from = np.hypot(result_branch[:, PF], result_branch[:, QF])
    apparent_to = np.hypot(result_branch[:, PT], result_branch[:, QT])
    rating = result_branch[:, RATE_A]
    in_service_and_rated = (
        (result_branch[:, BR_STATUS] > 0)
        & (rating > 0)
    )

    loading = np.full(len(result_branch), np.nan)
    loading[in_service_and_rated] = (
        np.maximum(
            apparent_from[in_service_and_rated],
            apparent_to[in_service_and_rated],
        )
        / rating[in_service_and_rated]
        * 100.0
    )
    return loading


def connected_components(
    bus: np.ndarray,
    branch: np.ndarray,
) -> list[list[int]]:
    """운전 중인 AC Branch만으로 구성한 Bus 연결 성분을 구한다."""

    bus_ids = [int(value) for value in bus[:, BUS_I]]
    adjacency = {bus_id: set() for bus_id in bus_ids}

    for row in branch:
        if row[BR_STATUS] <= 0:
            continue
        from_bus = int(row[F_BUS])
        to_bus = int(row[T_BUS])
        adjacency[from_bus].add(to_bus)
        adjacency[to_bus].add(from_bus)

    remaining = set(bus_ids)
    components: list[list[int]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: list[int] = []
        remaining.remove(start)

        while stack:
            current = stack.pop()
            component.append(current)
            new_neighbors = adjacency[current] & remaining
            remaining.difference_update(new_neighbors)
            stack.extend(new_neighbors)

        components.append(sorted(component))

    return sorted(components, key=len, reverse=True)


def endpoint_text(
    branch_row: np.ndarray,
    bus_names: dict[int, str],
) -> str:
    from_bus = int(branch_row[F_BUS])
    to_bus = int(branch_row[T_BUS])
    return (
        f"{bus_names.get(from_bus, f'Bus {from_bus}')} → "
        f"{bus_names.get(to_bus, f'Bus {to_bus}')}"
    )


def empty_result_row(
    branch_number: int,
    branch_row: np.ndarray,
    bus_names: dict[int, str],
    base_loading: np.ndarray,
) -> dict[str, object]:
    from_bus = int(branch_row[F_BUS])
    to_bus = int(branch_row[T_BUS])
    return {
        "outage_branch": branch_number,
        "outage_from_bus": from_bus,
        "outage_from_name": bus_names.get(from_bus, ""),
        "outage_to_bus": to_bus,
        "outage_to_name": bus_names.get(to_bus, ""),
        "outage_base_loading_pct": base_loading[branch_number - 1],
        "status": "",
        "component_count": 1,
        "component_sizes": "",
        "converged": False,
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
        "max_voltage_violation_pu": np.nan,
        "ac_loss_mw": np.nan,
        "message": "",
    }


def evaluate_contingency(
    branch_index: int,
    bus: np.ndarray,
    gen: np.ndarray,
    branch: np.ndarray,
    bus_names: dict[int, str],
    base_loading: np.ndarray,
) -> dict[str, object]:
    """한 개 AC Branch 개방 사고를 계산한다."""

    branch_number = branch_index + 1
    outage_branch = branch.copy()
    outage_branch[branch_index, BR_STATUS] = 0
    row = empty_result_row(
        branch_number,
        branch[branch_index],
        bus_names,
        base_loading,
    )

    components = connected_components(bus, outage_branch)
    if len(components) > 1:
        row.update(
            {
                "status": "ISLANDING",
                "component_count": len(components),
                "component_sizes": "+".join(str(len(item)) for item in components),
                "message": "AC 계통이 분리되어 단일 slack AC-PF를 생략함",
            }
        )
        return row

    results, converged, message = run_ac_power_flow(bus, gen, outage_branch)
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
    loading = branch_loadings(result_branch)

    valid_loading = np.flatnonzero(~np.isnan(loading))
    if len(valid_loading):
        worst_branch_index = int(
            valid_loading[np.argmax(loading[valid_loading])]
        )
        worst_branch_number = worst_branch_index + 1
        worst_branch_row = result_branch[worst_branch_index]
        worst_from_bus = int(worst_branch_row[F_BUS])
        worst_to_bus = int(worst_branch_row[T_BUS])
        max_loading = float(loading[worst_branch_index])
        overloaded_count = int(np.sum(loading > 100.0 + EPSILON))
    else:
        worst_branch_number = ""
        worst_from_bus = ""
        worst_to_bus = ""
        max_loading = np.nan
        overloaded_count = 0

    voltages = result_bus[:, VM]
    lower_violation = np.maximum(result_bus[:, VMIN] - voltages, 0.0)
    upper_violation = np.maximum(voltages - result_bus[:, VMAX], 0.0)
    voltage_violation = np.maximum(lower_violation, upper_violation)
    voltage_violation_count = int(np.sum(voltage_violation > EPSILON))
    max_voltage_violation = float(np.max(voltage_violation))
    min_voltage_index = int(np.argmin(voltages))
    min_voltage_bus = int(result_bus[min_voltage_index, BUS_I])

    active_generation = result_gen[:, GEN_STATUS] > 0
    ac_loss_mw = float(
        result_gen[active_generation, 1].sum() - result_bus[:, PD].sum()
    )

    has_violation = overloaded_count > 0 or voltage_violation_count > 0
    row.update(
        {
            "status": "LIMIT_VIOLATION" if has_violation else "OK",
            "converged": True,
            "max_loading_pct": max_loading,
            "worst_branch": worst_branch_number,
            "worst_from_bus": worst_from_bus,
            "worst_from_name": bus_names.get(worst_from_bus, ""),
            "worst_to_bus": worst_to_bus,
            "worst_to_name": bus_names.get(worst_to_bus, ""),
            "overloaded_branch_count": overloaded_count,
            "min_voltage_pu": float(voltages[min_voltage_index]),
            "min_voltage_bus": min_voltage_bus,
            "min_voltage_name": bus_names.get(min_voltage_bus, ""),
            "voltage_violation_count": voltage_violation_count,
            "max_voltage_violation_pu": max_voltage_violation,
            "ac_loss_mw": ac_loss_mw,
        }
    )
    return row


def format_float(value: object, digits: int = 2) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "-"
    if np.isnan(numeric):
        return "-"
    return f"{numeric:.{digits}f}"


def write_csv(output_file: Path, rows: list[dict[str, object]]) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_islanding_rows(
    rows: list[dict[str, object]],
    branch: np.ndarray,
    bus_names: dict[int, str],
    top: int,
) -> None:
    islanding_rows = [row for row in rows if row["status"] == "ISLANDING"]
    if not islanding_rows:
        return

    print()
    print("[계통 분리를 일으키는 사고]")
    for rank, row in enumerate(islanding_rows[:top], start=1):
        branch_index = int(row["outage_branch"]) - 1
        print(
            f"{rank:>2}. B{branch_index + 1:>3} "
            f"{endpoint_text(branch[branch_index], bus_names):<20} "
            f"분리 크기 {row['component_sizes']}"
        )


def print_loading_ranking(
    rows: list[dict[str, object]],
    branch: np.ndarray,
    bus_names: dict[int, str],
    top: int,
) -> None:
    converged_rows = [row for row in rows if row["converged"]]
    ranked = sorted(
        converged_rows,
        key=lambda row: float(row["max_loading_pct"]),
        reverse=True,
    )

    print()
    print(f"[사고 후 최대 선로 부하율 상위 {min(top, len(ranked))}건]")
    for rank, row in enumerate(ranked[:top], start=1):
        outage_index = int(row["outage_branch"]) - 1
        worst_index = int(row["worst_branch"]) - 1
        print(
            f"{rank:>2}. B{outage_index + 1:>3} "
            f"{endpoint_text(branch[outage_index], bus_names):<20} 탈락 → "
            f"B{worst_index + 1:>3} "
            f"{endpoint_text(branch[worst_index], bus_names):<20} "
            f"{float(row['max_loading_pct']):7.2f}%  "
            f"과부하 {int(row['overloaded_branch_count']):>2}개"
        )


def print_voltage_ranking(
    rows: list[dict[str, object]],
    branch: np.ndarray,
    bus_names: dict[int, str],
    top: int,
) -> None:
    converged_rows = [row for row in rows if row["converged"]]
    ranked = sorted(
        converged_rows,
        key=lambda row: (
            float(row["max_voltage_violation_pu"]),
            -float(row["min_voltage_pu"]),
        ),
        reverse=True,
    )

    print()
    print(f"[사고 후 전압 취약 상위 {min(top, len(ranked))}건]")
    for rank, row in enumerate(ranked[:top], start=1):
        outage_index = int(row["outage_branch"]) - 1
        print(
            f"{rank:>2}. B{outage_index + 1:>3} "
            f"{endpoint_text(branch[outage_index], bus_names):<20} 탈락 → "
            f"최저 {float(row['min_voltage_pu']):.5f} pu "
            f"(Bus {int(row['min_voltage_bus']):>3} "
            f"{row['min_voltage_name']}), "
            f"한계 위반 {int(row['voltage_violation_count']):>2}개"
        )


def main() -> None:
    args = parse_args()
    if args.top < 1:
        raise ValueError("--top은 1 이상이어야 합니다.")

    case_text = CASE_FILE.read_text(encoding="utf-8")
    bus = load_matpower_matrix(case_text, "bus")
    conventional_gen = load_matpower_matrix(case_text, "gen")
    branch = load_matpower_matrix(case_text, "branch")
    dcline = load_matpower_matrix(case_text, "dcline")
    bus_names = load_bus_names(BUS_METADATA_FILE)

    bus, gen, active_dc_lines = add_fixed_hvdc_as_dummy_generators(
        bus.copy(),
        conventional_gen.copy(),
        dcline,
    )

    base_components = connected_components(bus, branch)
    if len(base_components) != 1:
        raise RuntimeError(
            "기본 AC 계통이 이미 분리되어 있습니다: "
            + "+".join(str(len(item)) for item in base_components)
        )

    base_results, base_converged, base_message = run_ac_power_flow(
        bus,
        gen,
        branch,
    )
    if base_results is None or not base_converged:
        raise RuntimeError(
            f"기본 AC 조류계산이 수렴하지 않았습니다. {base_message}"
        )
    base_loading = branch_loadings(base_results["branch"])

    active_indices = np.flatnonzero(branch[:, BR_STATUS] > 0)
    if args.branch:
        requested_indices: list[int] = []
        for branch_number in dict.fromkeys(args.branch):
            branch_index = branch_number - 1
            if branch_index < 0 or branch_index >= len(branch):
                raise ValueError(
                    f"Branch {branch_number}는 범위 1~{len(branch)} 밖입니다."
                )
            if branch[branch_index, BR_STATUS] <= 0:
                raise ValueError(
                    f"Branch {branch_number}는 기본 상태에서 이미 정지 중입니다."
                )
            requested_indices.append(branch_index)
        study_indices = np.asarray(requested_indices, dtype=int)
    else:
        study_indices = active_indices

    print("=" * 86)
    print("KPG-193 v2.0 AC 선로 N-1 사고 스크리닝")
    print("=" * 86)
    print(
        f"기본 운전점: Bus {len(bus)} / AC Branch {len(branch)} / "
        f"운전 중 HVDC {active_dc_lines}"
    )
    print(
        f"기본 최대 선로 부하율: {np.nanmax(base_loading):.2f}% / "
        f"계산할 사고: {len(study_indices)}건"
    )
    print("판정 기준: Branch RATE_A, Bus별 VMIN/VMAX, AC-PF 수렴 여부")
    print()

    rows: list[dict[str, object]] = []
    total = len(study_indices)
    for order, branch_index in enumerate(study_indices, start=1):
        rows.append(
            evaluate_contingency(
                int(branch_index),
                bus,
                gen,
                branch,
                bus_names,
                base_loading,
            )
        )
        if total > 20 and (order % 50 == 0 or order == total):
            print(f"진행: {order:>3}/{total}건")

    write_csv(args.output, rows)

    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in (
            "OK",
            "LIMIT_VIOLATION",
            "ISLANDING",
            "NON_CONVERGED",
            "ERROR",
        )
    }
    overload_cases = sum(
        bool(row["converged"])
        and int(row["overloaded_branch_count"]) > 0
        for row in rows
    )
    voltage_cases = sum(
        bool(row["converged"])
        and int(row["voltage_violation_count"]) > 0
        for row in rows
    )

    print()
    print("[전체 요약]")
    print(f"정상(한계 내)              : {counts['OK']:>3}건")
    print(f"한계 위반                  : {counts['LIMIT_VIOLATION']:>3}건")
    print(f"  ├─ 선로 과부하 발생 사고 : {overload_cases:>3}건")
    print(f"  └─ 전압 한계 위반 사고   : {voltage_cases:>3}건")
    print(f"계통 분리                  : {counts['ISLANDING']:>3}건")
    print(f"AC-PF 비수렴               : {counts['NON_CONVERGED']:>3}건")
    print(f"계산 오류                  : {counts['ERROR']:>3}건")

    print_islanding_rows(rows, branch, bus_names, args.top)
    print_loading_ranking(rows, branch, bus_names, args.top)
    print_voltage_ranking(rows, branch, bus_names, args.top)

    print()
    print("[주의]")
    print(
        "이 결과는 정적 AC N-1 스크리닝이다. 보호계전·과도안정도·주파수·"
        "자동 재폐로·운영자 corrective action은 모의하지 않는다."
    )
    print(f"전체 결과 CSV: {args.output.resolve()}")


if __name__ == "__main__":
    main()