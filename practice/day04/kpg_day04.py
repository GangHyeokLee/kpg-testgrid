"""KPG-193 v2.0 Day 4 - DC 최적조류계산(DC-OPF) 실습.

Day 3에서는 사람이 정한 단순 참여규칙으로 발전량을 배분했다. Day 4에서는
발전비용, 발전기 Pmin/Pmax, 선로 RATE_A를 사용해 DC-OPF가 발전기 Pg를
직접 결정하게 한다.

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
from pypower.api import ppoption, rundcopf


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
PD = 2
QD = 3
LAM_P = 13

GEN_BUS = 0
PG = 1
GEN_STATUS = 7
PMAX = 8
PMIN = 9

F_BUS = 0
T_BUS = 1
RATE_A = 5
BR_STATUS = 10
PF = 13
PT = 15

BINDING_THRESHOLD_PCT = 99.9
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


def branch_loadings(result_branch: np.ndarray) -> np.ndarray:
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
    loading = branch_loadings(result_branch)
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
        f"(B{worst_index + 1} "
        f"{bus_names.get(from_bus, from_bus)} → "
        f"{bus_names.get(to_bus, to_bus)})"
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
    print("생략: 무효전력 Q·전압 크기·AC 손실")

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
        "DC-OPF 성공은 DC 근사모델에서 제약을 만족했다는 뜻이다. "
        "Q·전압·손실과 AC 피상전력 한계까지 안전하다는 뜻은 아니다."
    )
    print(
        "비용함수와 LMP는 KPG 원본 단위를 그대로 표시하므로 "
        "통화를 임의로 원 또는 달러로 해석하지 않는다."
    )


if __name__ == "__main__":
    main()