"""KPG-193 v2.0 Day 4 - DC 최적조류계산(DC-OPF) 실습.

Day 3에서는 사람이 정한 단순 참여규칙으로 발전량을 배분했다. Day 4에서는
발전비용, 발전기 Pmin/Pmax, 선로 RATE_A를 사용해 DC-OPF가 발전기 Pg를
직접 결정하게 한다. 이어서 같은 Pg를 AC-PF에 적용해 전압, AC 선로 MVA
부하율, 발전기 P/Q 한계와 Slack 발전기의 손실 부담을 재검증한다.

예:
    python kpg_day04.py
    python kpg_day04.py --scales 100 105
    python kpg_day04.py --scales 103 104 105 --top 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypower.api import ppoption, rundcopf, runpf


SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PRACTICE_DIR))

try:
    from day01.kpg_day01 import (
        BUS_METADATA_FILE,
        CASE_FILE,
        add_fixed_hvdc_as_dummy_generators,
        load_matpower_matrix,
    )
except ModuleNotFoundError:
    # 예전 로컬 작업본처럼 day01이 저장소 루트에 있을 때의 시험용 fallback.
    sys.path.insert(0, str(SCRIPT_DIR.parents[1]))
    from kpg_day01 import (  # type: ignore[no-redef]
        BUS_METADATA_FILE,
        CASE_FILE,
        add_fixed_hvdc_as_dummy_generators,
        load_matpower_matrix,
    )


# MATPOWER/PYPOWER 열 번호의 Python 0-based index.
BUS_I = 0
BUS_TYPE = 1
PD = 2
QD = 3
VM = 7
VMAX = 11
VMIN = 12
LAM_P = 13

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

BINDING_THRESHOLD_PCT = 99.9
LIMIT_TOLERANCE = 1e-4
DEFAULT_SCALES = [100.0, 105.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KPG-193 DC-OPF")
    parser.add_argument(
        "--scales",
        type=float,
        nargs="+",
        default=DEFAULT_SCALES,
        help="기본 양(+) 부하에 적용할 백분율 목록 (기본값: 100 105)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="발전기 재급전 변화량 상위 출력 개수 (기본값: 5)",
    )
    return parser.parse_args()


def load_bus_names() -> dict[int, str]:
    metadata = pd.read_csv(BUS_METADATA_FILE, encoding="utf-8-sig")
    return {
        int(bus_id): str(name)
        for bus_id, name in zip(metadata["bus_id"], metadata["name_Korean"])
    }


def scale_positive_loads(bus: np.ndarray, scale_pct: float) -> np.ndarray:
    """양(+) Pd와 해당 Bus의 Qd만 같은 비율로 조정한다."""

    scaled = bus.copy()
    positive_load = scaled[:, PD] > 0.0
    factor = scale_pct / 100.0
    scaled[positive_load, PD] *= factor
    scaled[positive_load, QD] *= factor
    return scaled


def prepare_fixed_hvdc_for_opf(
    gen: np.ndarray,
    gencost: np.ndarray,
    original_gen_count: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """HVDC 더미 발전기의 P를 고정하고 0 비용행을 추가한다.

    PYPOWER에는 MATPOWER dcline의 OPF 확장 기능이 없다. Day 1에서 만든
    HVDC 양단 더미 발전기는 고정 주입으로만 사용해야 하므로 Pmin과 Pmax를
    현재 Pg로 잠근다. 고정 변수의 추가 비용은 0으로 둔다.
    """

    prepared_gen = gen.copy()
    dummy_count = len(prepared_gen) - original_gen_count
    if dummy_count <= 0:
        return prepared_gen, gencost.copy(), 0

    dummy = prepared_gen[original_gen_count:]
    dummy[:, PMAX] = dummy[:, PG]
    dummy[:, PMIN] = dummy[:, PG]

    # polynomial model, startup, shutdown, n, c2, c1, c0
    zero_cost_rows = np.tile(
        np.asarray([2.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]),
        (dummy_count, 1),
    )
    return prepared_gen, np.vstack([gencost, zero_cost_rows]), dummy_count


def dc_branch_loadings(result_branch: np.ndarray) -> np.ndarray:
    """DC 유효전력 흐름을 RATE_A로 나눈 선로 부하율을 계산한다."""

    rate = result_branch[:, RATE_A]
    valid = (result_branch[:, BR_STATUS] > 0) & (rate > 0)
    loading = np.full(len(result_branch), np.nan)
    loading[valid] = (
        np.maximum(
            np.abs(result_branch[valid, PF]),
            np.abs(result_branch[valid, PT]),
        )
        / rate[valid]
        * 100.0
    )
    return loading


def ac_branch_loadings(result_branch: np.ndarray) -> np.ndarray:
    """AC 양단 피상전력 중 큰 값을 RATE_A로 나눈 부하율을 계산한다."""

    rate = result_branch[:, RATE_A]
    valid = (result_branch[:, BR_STATUS] > 0) & (rate > 0)
    loading = np.full(len(result_branch), np.nan)
    from_mva = np.hypot(result_branch[:, PF], result_branch[:, QF])
    to_mva = np.hypot(result_branch[:, PT], result_branch[:, QT])
    loading[valid] = (
        np.maximum(from_mva[valid], to_mva[valid])
        / rate[valid]
        * 100.0
    )
    return loading


def print_top_redispatch(
    base_gen: np.ndarray,
    optimized_gen: np.ndarray,
    bus_names: dict[int, str],
    top: int,
) -> None:
    """기존 Pg 대비 DC-OPF Pg 변화량이 큰 발전기를 출력한다."""

    active = base_gen[:, GEN_STATUS] > 0
    change = optimized_gen[:, PG] - base_gen[:, PG]
    candidates = np.flatnonzero(active)
    order = candidates[np.argsort(np.abs(change[candidates]))[::-1]]

    print(f"\n[발전기 재급전 | 절대 변화량 상위 {top}개]")
    print(" 발전기     Bus         기존 Pg      OPF Pg       변화량")
    print("-" * 66)
    for index in order[:top]:
        bus_id = int(base_gen[index, GEN_BUS])
        name = bus_names.get(bus_id, f"Bus {bus_id}")
        print(
            f"G{index + 1:>3}  B{bus_id:>3} {name:<8} "
            f"{base_gen[index, PG]:>10.1f}  "
            f"{optimized_gen[index, PG]:>10.1f}  "
            f"{change[index]:>+10.1f} MW"
        )


def validate_ac_result(
    scenario_bus: np.ndarray,
    base_gen: np.ndarray,
    branch: np.ndarray,
    optimized_gen: np.ndarray,
    original_gen_count: int,
    bus_names: dict[int, str],
) -> None:
    """DC-OPF의 Pg를 AC-PF에 적용하고 AC 계통 한계를 재검증한다.

    AC-PF는 발전기 Q 한계를 자동으로 안전 판정하지 않는다. 따라서 수렴
    이후 전압, 선로 MVA 부하율, 발전기 P/Q 한계를 별도로 검사한다.
    AC 손실은 REF(Slack) Bus의 발전기가 자동으로 부담한다.
    """

    ac_gen = base_gen.copy()
    ac_gen[:, PG] = optimized_gen[:, PG]
    ac_case = {
        "version": "2",
        "baseMVA": 100.0,
        "bus": scenario_bus.copy(),
        "gen": ac_gen,
        "branch": branch.copy(),
    }
    ac_results, converged = runpf(
        ac_case,
        ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1),
    )

    print("\n[DC-OPF 결과의 AC 재검증]")
    print(f"AC-PF 수렴          : {bool(converged)}")
    if not converged:
        print("최종 판정           : 실패 (AC-PF 미수렴)")
        return

    result_bus = ac_results["bus"]
    result_gen = ac_results["gen"]
    result_branch = ac_results["branch"]

    voltage_violation = (
        (result_bus[:, VM] < result_bus[:, VMIN] - LIMIT_TOLERANCE)
        | (result_bus[:, VM] > result_bus[:, VMAX] + LIMIT_TOLERANCE)
    )
    voltage_violation_indices = np.flatnonzero(voltage_violation)
    min_voltage_index = int(np.argmin(result_bus[:, VM]))
    max_voltage_index = int(np.argmax(result_bus[:, VM]))

    loading = ac_branch_loadings(result_branch)
    rated = np.flatnonzero(~np.isnan(loading))
    worst_index = int(rated[np.argmax(loading[rated])])
    overloaded_indices = np.flatnonzero(
        loading > 100.0 + LIMIT_TOLERANCE
    )
    from_bus = int(result_branch[worst_index, F_BUS])
    to_bus = int(result_branch[worst_index, T_BUS])

    generator_indices = np.arange(len(result_gen))
    active = result_gen[:, GEN_STATUS] > 0
    conventional = generator_indices < original_gen_count
    active_conventional = active & conventional
    p_violation = active_conventional & (
        (result_gen[:, PG] > result_gen[:, PMAX] + LIMIT_TOLERANCE)
        | (result_gen[:, PG] < result_gen[:, PMIN] - LIMIT_TOLERANCE)
    )
    q_violation = active & (
        (result_gen[:, QG] > result_gen[:, QMAX] + LIMIT_TOLERANCE)
        | (result_gen[:, QG] < result_gen[:, QMIN] - LIMIT_TOLERANCE)
    )
    p_violation_indices = np.flatnonzero(p_violation)
    q_violation_indices = np.flatnonzero(q_violation)

    reference_bus_ids = scenario_bus[
        scenario_bus[:, BUS_TYPE] == REF, BUS_I
    ].astype(int)
    slack_generators = active_conventional & np.isin(
        result_gen[:, GEN_BUS].astype(int),
        reference_bus_ids,
    )
    slack_pg_change = float(
        (result_gen[slack_generators, PG] - optimized_gen[slack_generators, PG]).sum()
    )
    slack_p_violation_count = int(np.sum(p_violation & slack_generators))

    net_demand = float(result_bus[:, PD].sum())
    ac_loss = float(result_gen[:, PG].sum() - net_demand)

    min_bus = int(result_bus[min_voltage_index, BUS_I])
    max_bus = int(result_bus[max_voltage_index, BUS_I])
    print(
        f"전압 범위           : {result_bus[min_voltage_index, VM]:.5f} pu "
        f"(B{min_bus} {bus_names.get(min_bus, min_bus)}) ~ "
        f"{result_bus[max_voltage_index, VM]:.5f} pu "
        f"(B{max_bus} {bus_names.get(max_bus, max_bus)})"
    )
    print(f"전압 위반 Bus      : {len(voltage_violation_indices)}개")
    print(
        f"최대 AC 선로 부하율: {loading[worst_index]:.2f}% "
        f"(Branch {worst_index + 1}: "
        f"B{from_bus} {bus_names.get(from_bus, from_bus)} → "
        f"B{to_bus} {bus_names.get(to_bus, to_bus)})"
    )
    print(f"AC 선로 과부하     : {len(overloaded_indices)}개")
    print(f"발전기 P 한계 위반 : {len(p_violation_indices)}개 (기존 발전기)")
    print(
        f"Slack Bus Pg 변화  : {slack_pg_change:+,.1f} MW / "
        f"P 한계 위반 {slack_p_violation_count}개"
    )
    print(
        f"발전기 Q 한계 위반 : {len(q_violation_indices)}개 "
        "(HVDC 더미 포함)"
    )
    print(f"AC 유효전력 손실   : {ac_loss:,.1f} MW")

    if len(voltage_violation_indices):
        print("  전압 위반 예시:")
        for index in voltage_violation_indices[:3]:
            bus_id = int(result_bus[index, BUS_I])
            print(
                f"    B{bus_id} {bus_names.get(bus_id, bus_id)}: "
                f"{result_bus[index, VM]:.5f} pu "
                f"(허용 {result_bus[index, VMIN]:.3f}~"
                f"{result_bus[index, VMAX]:.3f})"
            )

    if len(overloaded_indices):
        print("  선로 과부하 상위:")
        order = overloaded_indices[
            np.argsort(loading[overloaded_indices])[::-1]
        ]
        for index in order[:3]:
            f_bus = int(result_branch[index, F_BUS])
            t_bus = int(result_branch[index, T_BUS])
            print(
                f"    Branch {index + 1}: "
                f"B{f_bus} {bus_names.get(f_bus, f_bus)} → "
                f"B{t_bus} {bus_names.get(t_bus, t_bus)}  "
                f"{loading[index]:.2f}%"
            )

    if len(p_violation_indices):
        print("  P 한계 위반 예시:")
        for index in p_violation_indices[:3]:
            bus_id = int(result_gen[index, GEN_BUS])
            print(
                f"    G{index + 1} B{bus_id} {bus_names.get(bus_id, bus_id)}: "
                f"Pg={result_gen[index, PG]:.1f} MW "
                f"(허용 {result_gen[index, PMIN]:.1f}~"
                f"{result_gen[index, PMAX]:.1f})"
            )

    if len(q_violation_indices):
        print("  Q 한계 위반 예시:")
        for index in q_violation_indices[:3]:
            bus_id = int(result_gen[index, GEN_BUS])
            generator_label = (
                f"G{index + 1}"
                if index < original_gen_count
                else f"HVDC더미{index - original_gen_count + 1}"
            )
            print(
                f"    {generator_label} B{bus_id} "
                f"{bus_names.get(bus_id, bus_id)}: "
                f"Qg={result_gen[index, QG]:.1f} Mvar "
                f"(허용 {result_gen[index, QMIN]:.1f}~"
                f"{result_gen[index, QMAX]:.1f})"
            )

    passed = not (
        len(voltage_violation_indices)
        or len(overloaded_indices)
        or len(p_violation_indices)
        or len(q_violation_indices)
    )
    print(
        "최종 판정           : "
        + ("통과" if passed else "위반 있음 (AC 보정 필요)")
    )


def run_scenario(
    scale_pct: float,
    base_bus: np.ndarray,
    base_gen: np.ndarray,
    branch: np.ndarray,
    gencost: np.ndarray,
    original_gen_count: int,
    bus_names: dict[int, str],
    top: int,
) -> None:
    scenario_bus = scale_positive_loads(base_bus, scale_pct)
    case = {
        "version": "2",
        "baseMVA": 100.0,
        "bus": scenario_bus,
        "gen": base_gen.copy(),
        "branch": branch.copy(),
        "gencost": gencost.copy(),
    }
    results = rundcopf(case, ppoption(VERBOSE=0, OUT_ALL=0))

    print("\n" + "=" * 82)
    print(f"부하 {scale_pct:.1f}% 시나리오")
    print("=" * 82)
    print(f"DC-OPF 성공       : {bool(results['success'])}")
    if not results["success"]:
        message = results.get("raw", {}).get("output", {}).get(
            "message", "원인 메시지 없음"
        )
        print(f"실패 메시지        : {message}")
        return

    result_bus = results["bus"]
    result_gen = results["gen"]
    result_branch = results["branch"]
    loading = dc_branch_loadings(result_branch)
    rated = np.flatnonzero(~np.isnan(loading))
    worst_index = int(rated[np.argmax(loading[rated])])
    from_bus = int(result_branch[worst_index, F_BUS])
    to_bus = int(result_branch[worst_index, T_BUS])

    active_conventional = (
        result_gen[:original_gen_count, GEN_STATUS] > 0
    )
    conventional_generation = float(
        result_gen[:original_gen_count][active_conventional, PG].sum()
    )
    pmax_count = int(
        np.sum(
            active_conventional
            & (
                result_gen[:original_gen_count, PMAX]
                - result_gen[:original_gen_count, PG]
                <= 1e-4
            )
        )
    )

    print(f"순수요             : {result_bus[:, PD].sum():,.1f} MW")
    print(f"기존 발전기 합계   : {conventional_generation:,.1f} MW")
    print(f"비용함수 값        : {float(results['f']):,.1f} (원 데이터 단위)")
    print(
        f"최대 선로 부하율   : {loading[worst_index]:.2f}% "
        f"(Branch {worst_index + 1}: "
        f"B{from_bus} {bus_names.get(from_bus, from_bus)} → "
        f"B{to_bus} {bus_names.get(to_bus, to_bus)})"
    )
    print(
        f"한계 도달 선로     : "
        f"{int(np.sum(loading >= BINDING_THRESHOLD_PCT))}개 "
        f"({BINDING_THRESHOLD_PCT:.1f}% 이상)"
    )
    print(f"Pmax 도달 발전기   : {pmax_count}개")
    print(
        f"LMP 범위           : "
        f"{np.min(result_bus[:, LAM_P]):.2f} ~ "
        f"{np.max(result_bus[:, LAM_P]):.2f} (원 데이터 단위/MWh)"
    )
    print_top_redispatch(
        base_gen[:original_gen_count],
        result_gen[:original_gen_count],
        bus_names,
        top,
    )
    validate_ac_result(
        scenario_bus,
        base_gen,
        branch,
        result_gen,
        original_gen_count,
        bus_names,
    )


def main() -> None:
    args = parse_args()
    if not args.scales or any(scale <= 0 for scale in args.scales):
        raise ValueError("--scales에는 0보다 큰 배율이 하나 이상 필요합니다.")
    if args.top <= 0:
        raise ValueError("--top은 1 이상이어야 합니다.")

    scales = list(dict.fromkeys(float(value) for value in args.scales))
    case_text = CASE_FILE.read_text(encoding="utf-8")
    bus = load_matpower_matrix(case_text, "bus")
    conventional_gen = load_matpower_matrix(case_text, "gen")
    branch = load_matpower_matrix(case_text, "branch")
    dcline = load_matpower_matrix(case_text, "dcline")
    gencost = load_matpower_matrix(case_text, "gencost")
    original_gen_count = len(conventional_gen)
    bus_names = load_bus_names()

    bus, gen, active_dc_lines = add_fixed_hvdc_as_dummy_generators(
        bus.copy(),
        conventional_gen.copy(),
        dcline,
    )
    gen, gencost, dummy_count = prepare_fixed_hvdc_for_opf(
        gen,
        gencost,
        original_gen_count,
    )

    print("=" * 82)
    print("KPG-193 v2.0 Day 4 - DC 최적조류계산")
    print("=" * 82)
    print(
        f"Bus {len(bus)} / 기존 발전기 {original_gen_count} / "
        f"AC Branch {len(branch)} / 운전 중 HVDC {active_dc_lines}"
    )
    print(
        f"HVDC 처리: 양단 더미 발전기 {dummy_count}개의 P 고정 / "
        "원본 발전비용 사용"
    )
    print("고려: 발전비용·Pmin/Pmax·선로 유효전력 한계")
    print("DC-OPF 생략: 무효전력 Q·전압 크기·AC 손실")
    print("후속 검증: DC-OPF의 Pg를 AC-PF에 적용해 AC 한계 재확인")

    for scale_pct in scales:
        run_scenario(
            scale_pct,
            bus,
            gen,
            branch,
            gencost,
            original_gen_count,
            bus_names,
            args.top,
        )

    print("\n[해석 주의]")
    print(
        "AC 재검증은 DC-OPF와 동일한 Pg를 시작점으로 사용하되, "
        "AC 손실은 Slack Bus 발전기가 추가로 부담한다."
    )
    print(
        "비용함수와 LMP는 KPG 원본 단위를 그대로 표시하므로 "
        "통화를 임의로 원 또는 달러로 해석하지 않는다."
    )
    print(
        "AC 기본운전점이 통과해도 N-1 안전까지 보장되지는 않는다. "
        "최종 확인에는 N-1 AC 재검증이 필요하다."
    )


if __name__ == "__main__":
    main()
