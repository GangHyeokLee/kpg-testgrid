"""KPG-193 v2.0 기본 AC 조류계산 예제.

실행 위치와 관계없이 이 파일이 있는 KPG 저장소를 기준으로 데이터를 찾는다.
MATPOWER의 고정 HVDC 모델은 PYPOWER가 직접 지원하지 않으므로,
MATPOWER toggle_dcline의 단순 조류계산 방식을 따라 양단 dummy generator로 변환한다.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from pypower.api import ppoption, runpf


ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = ROOT.parents[1]
CASE_FILE = REPOSITORY_ROOT / "kpg193_v2_0" / "network" / "m" / "KPG193_ver2_0.m"
BUS_METADATA_FILE = (
    REPOSITORY_ROOT
    / "kpg193_v2_0"
    / "network"
    / "metadata"
    / "bus_metadata"
    / "bus_metadata_2025.csv"
)


def load_matpower_matrix(case_text: str, field: str) -> np.ndarray:
    """MATPOWER .m 파일에서 mpc.<field> = [...] 숫자 행렬을 읽는다."""

    match = re.search(
        rf"mpc\.{re.escape(field)}\s*=\s*\[(.*?)\];",
        case_text,
        re.DOTALL,
    )
    if match is None:
        raise ValueError(f"mpc.{field} 행렬을 찾지 못했습니다.")

    rows: list[list[float]] = []
    for raw_line in match.group(1).splitlines():
        line = raw_line.split("%", 1)[0].strip().rstrip(";").strip()
        if line:
            rows.append([float(value) for value in line.split()])

    return np.asarray(rows, dtype=float)


def add_fixed_hvdc_as_dummy_generators(
    bus: np.ndarray,
    gen: np.ndarray,
    dcline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """MATPOWER의 고정 HVDC 단순 조류계산 모델을 PYPOWER용으로 변환한다."""

    # MATPOWER bus/gen 열 번호를 Python의 0-based index로 표현한다.
    BUS_I = 0
    BUS_TYPE = 1
    PV = 2
    REF = 3

    GEN_BUS = 0
    PG = 1
    QG = 2
    QMAX = 3
    QMIN = 4
    VG = 5
    MBASE = 6
    GEN_STATUS = 7
    PMAX = 8
    PMIN = 9

    from_generators: list[np.ndarray] = []
    to_generators: list[np.ndarray] = []
    active_dc_lines = 0

    for row in dcline:
        (
            f_bus,
            t_bus,
            status,
            pf,
            _pt,
            qf,
            qt,
            vf,
            vt,
            pmin,
            pmax,
            qmin_f,
            qmax_f,
            qmin_t,
            qmax_t,
            loss0,
            loss1,
        ) = row[:17]

        if status <= 0:
            continue

        active_dc_lines += 1
        pt = pf - (loss0 + loss1 * pf)

        from_gen = np.zeros(gen.shape[1], dtype=float)
        to_gen = np.zeros(gen.shape[1], dtype=float)

        for dummy_gen in (from_gen, to_gen):
            dummy_gen[MBASE] = 100.0
            dummy_gen[GEN_STATUS] = status
            dummy_gen[PMAX] = np.inf
            dummy_gen[PMIN] = -np.inf

        from_gen[[GEN_BUS, PG, QG, QMAX, QMIN, VG]] = [
            f_bus,
            -pf,
            qf,
            qmax_f,
            qmin_f,
            vf,
        ]
        to_gen[[GEN_BUS, PG, QG, QMAX, QMIN, VG]] = [
            t_bus,
            pt,
            qt,
            qmax_t,
            qmin_t,
            vt,
        ]

        # MATPOWER toggle_dcline의 방향별 유효전력 제한 처리.
        if pmin >= 0:
            from_gen[PMAX] = -pmin
        if pmax >= 0:
            from_gen[PMIN] = -pmax
        if pmin < 0:
            to_gen[PMIN] = pmin
        if pmax < 0:
            to_gen[PMAX] = pmax

        from_generators.append(from_gen)
        to_generators.append(to_gen)

        # HVDC 양단을 PV bus로 설정하되 기준(REF) bus는 유지한다.
        for bus_id in (int(f_bus), int(t_bus)):
            bus_index = np.flatnonzero(bus[:, BUS_I] == bus_id)
            if len(bus_index) != 1:
                raise ValueError(f"HVDC 연결 bus {bus_id}를 하나로 특정하지 못했습니다.")
            if bus[bus_index[0], BUS_TYPE] != REF:
                bus[bus_index[0], BUS_TYPE] = PV

    if active_dc_lines:
        gen = np.vstack([gen, from_generators, to_generators])

    return bus, gen, active_dc_lines


def main() -> None:
    case_text = CASE_FILE.read_text(encoding="utf-8")

    bus = load_matpower_matrix(case_text, "bus")
    gen = load_matpower_matrix(case_text, "gen")
    branch = load_matpower_matrix(case_text, "branch")
    dcline = load_matpower_matrix(case_text, "dcline")
    original_gen_count = len(gen)

    metadata = pd.read_csv(BUS_METADATA_FILE, encoding="utf-8-sig")
    bus_names = dict(zip(metadata["bus_id"], metadata["name_Korean"]))

    bus, gen, active_dc_lines = add_fixed_hvdc_as_dummy_generators(
        bus.copy(),
        gen.copy(),
        dcline,
    )

    case = {
        "version": "2",
        "baseMVA": 100.0,
        "bus": bus,
        "gen": gen,
        "branch": branch,
    }
    options = ppoption(VERBOSE=0, OUT_ALL=0, PF_ALG=1)
    results, converged = runpf(case, options)

    if not converged:
        raise RuntimeError("AC 조류계산이 수렴하지 않았습니다.")

    result_bus = results["bus"]
    result_gen = results["gen"]
    result_branch = results["branch"]

    gross_load_mw = np.maximum(result_bus[:, 2], 0).sum()
    fixed_generation_mw = -np.minimum(result_bus[:, 2], 0).sum()
    net_demand_mw = result_bus[:, 2].sum()
    conventional_generation_mw = result_gen[:original_gen_count, 1].sum()
    ac_loss_mw = result_gen[:, 1].sum() - net_demand_mw

    min_voltage_index = int(np.argmin(result_bus[:, 7]))
    max_voltage_index = int(np.argmax(result_bus[:, 7]))

    print("=" * 72)
    print("KPG-193 v2.0 기본 AC 조류계산")
    print("=" * 72)
    print(f"수렴 여부                 : {bool(converged)}")
    print(f"Bus / 기존 발전기 / AC 선로: {len(bus)} / {original_gen_count} / {len(branch)}")
    print(f"운전 중인 HVDC 선로       : {active_dc_lines}")
    print()
    print("[계통 합계]")
    print(f"총 양(+) 부하             : {gross_load_mw:,.1f} MW")
    print(f"Bus에 반영된 고정 발전    : {fixed_generation_mw:,.1f} MW")
    print(f"순수요                    : {net_demand_mw:,.1f} MW")
    print(f"기존 발전기 발전량        : {conventional_generation_mw:,.1f} MW")
    print(f"AC 계통 유효전력 손실     : {ac_loss_mw:,.1f} MW")
    print()
    print("[전압 최저·최고 Bus]")

    for label, index in (
        ("최저", min_voltage_index),
        ("최고", max_voltage_index),
    ):
        row = result_bus[index]
        bus_id = int(row[0])
        print(
            f"{label}: Bus {bus_id:>3} {bus_names.get(bus_id, '이름 없음'):<8} "
            f"{row[7]:.5f} pu  {row[8]:>9.3f} deg  {row[9]:.0f} kV"
        )

    # branch 결과 열: PF=13, QF=14, PT=15, QT=16 (Python 0-based)
    rate_a = result_branch[:, 5]
    apparent_from = np.hypot(result_branch[:, 13], result_branch[:, 14])
    apparent_to = np.hypot(result_branch[:, 15], result_branch[:, 16])
    loading = np.full(len(result_branch), np.nan)
    rated = rate_a > 0
    loading[rated] = np.maximum(apparent_from[rated], apparent_to[rated]) / rate_a[rated] * 100

    top_indices = np.argsort(np.nan_to_num(loading, nan=-1.0))[-5:][::-1]
    print()
    print("[AC 선로 부하율 상위 5개]")
    for rank, index in enumerate(top_indices, start=1):
        f_bus, t_bus = map(int, result_branch[index, :2])
        print(
            f"{rank}. Branch {index + 1:>3}: "
            f"{bus_names.get(f_bus, f'Bus {f_bus}')} → "
            f"{bus_names.get(t_bus, f'Bus {t_bus}')}  "
            f"{loading[index]:6.2f}%"
        )


if __name__ == "__main__":
    main()
