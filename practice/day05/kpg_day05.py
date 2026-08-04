"""KPG-193 v2.0 Day 5 - 24시간 시계열 시뮬레이션 실습.

KPG가 제공하는 하루 24시간의 Bus별 수요와 재생에너지 프로파일을
계통에 반영한다. 매시간 DC-OPF로 발전기 Pg를 재배분한 뒤, 같은 Pg를
AC-PF에 적용해 전압과 AC 선로 부하율을 재검증한다.

이 실습의 목적은 새로운 조류계산 기법을 배우는 것이 아니라 Day 1~4의
계산을 여러 운전 시나리오에 반복 적용해 AI EMS용 입력·결과 데이터가
어떻게 만들어지는지 확인하는 것이다.

예:
    python kpg_day05.py
    python kpg_day05.py --day 1
    python kpg_day05.py --day 180
    python kpg_day05.py --day 1 --output my_day05_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
from pypower.api import ppoption, rundcopf, runpf


SCRIPT_DIR = Path(__file__).resolve().parent
PRACTICE_DIR = SCRIPT_DIR.parent
REPOSITORY_DIR = SCRIPT_DIR.parents[1]
DATA_DIR = REPOSITORY_DIR / "kpg193_v2_0"

for import_path in (PRACTICE_DIR, REPOSITORY_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from day04.kpg_day04 import (  # noqa: E402
    BR_STATUS,
    BUS_I,
    F_BUS,
    GEN_STATUS,
    GEN_BUS,
    LIMIT_TOLERANCE,
    PD,
    PF,
    PG,
    PMAX,
    PMIN,
    PT,
    QD,
    QF,
    QG,
    QMAX,
    QMIN,
    QT,
    RATE_A,
    T_BUS,
    VM,
    VMAX,
    VMIN,
    prepare_fixed_hvdc_for_opf,
)
from day01.kpg_day01 import (  # noqa: E402
    BUS_METADATA_FILE,
    CASE_FILE,
    add_fixed_hvdc_as_dummy_generators,
    load_matpower_matrix,
)


DEFAULT_DAY = 8
HOURS_PER_DAY = 24


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KPG-193 하루 24시간 DC-OPF + AC-PF 시뮬레이션"
    )
    parser.add_argument(
        "--day",
        type=int,
        default=DEFAULT_DAY,
        help="KPG 프로파일의 일 번호 1~365 (기본값: 1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="시간별 결과 CSV 경로 (기본값: 실행 날짜 번호를 포함한 파일명)",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="결과 그래프 PNG 생성을 생략",
    )
    return parser.parse_args()


def load_bus_names() -> dict[int, str]:
    metadata = pd.read_csv(BUS_METADATA_FILE, encoding="utf-8-sig")
    return {
        int(bus_id): str(name)
        for bus_id, name in zip(metadata["bus_id"], metadata["name_Korean"])
    }


def read_day_profiles(
    day: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    demand_file = DATA_DIR / "profile" / "demand" / f"demand_{day}.csv"
    renewable_file = (
        DATA_DIR / "profile" / "renewables" / f"renewables_{day}.csv"
    )
    commitment_file = (
        DATA_DIR
        / "profile"
        / "commitment_decision"
        / f"commitment_decision_{day}.csv"
    )

    if not demand_file.is_file():
        raise FileNotFoundError(f"수요 프로파일을 찾지 못했습니다: {demand_file}")
    if not renewable_file.is_file():
        raise FileNotFoundError(
            f"재생에너지 프로파일을 찾지 못했습니다: {renewable_file}"
        )
    if not commitment_file.is_file():
        raise FileNotFoundError(
            f"발전기 기동·정지 프로파일을 찾지 못했습니다: {commitment_file}"
        )

    demand = pd.read_csv(demand_file)
    renewable = pd.read_csv(renewable_file)
    commitment = pd.read_csv(commitment_file, encoding="utf-8-sig")
    expected_rows = HOURS_PER_DAY * 193
    if len(demand) != expected_rows or len(renewable) != expected_rows:
        raise ValueError(
            "하루 프로파일은 24시간 × 193 Bus 행이어야 합니다. "
            f"현재 수요 {len(demand)}행 / 재생에너지 {len(renewable)}행"
        )

    expected_hours = set(range(1, HOURS_PER_DAY + 1))
    if set(demand["hour"].unique()) != expected_hours:
        raise ValueError("수요 프로파일의 hour가 1~24로 완전하지 않습니다.")
    if set(renewable["hour"].unique()) != expected_hours:
        raise ValueError("재생에너지 프로파일의 hour가 1~24로 완전하지 않습니다.")
    if set(commitment["hour"].unique()) != expected_hours:
        raise ValueError("발전기 기동·정지 프로파일의 hour가 1~24로 완전하지 않습니다.")

    return demand, renewable, commitment


def load_renewable_capacities() -> pd.DataFrame:
    """Bus별 태양광·풍력·수력 설비용량을 하나의 표로 합친다."""

    capacity_columns: dict[str, pd.Series] = {}
    for source, file_stem in (
        ("solar", "solar_generators_2025.csv"),
        ("wind", "wind_generators_2025.csv"),
        ("hydro", "hydro_generators_2025.csv"),
    ):
        path = DATA_DIR / "renewables_capacity" / file_stem
        data = pd.read_csv(path, encoding="utf-8-sig")
        capacity_columns[source] = data.groupby("bus_ID")["Pmax [MW]"].sum()

    capacities = pd.DataFrame(capacity_columns).fillna(0.0)
    capacities.index = capacities.index.astype(int)
    capacities.index.name = "bus_id"
    return capacities


def build_hourly_bus(
    base_bus: np.ndarray,
    hourly_demand: pd.DataFrame,
    hourly_renewable: pd.DataFrame,
    capacities: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    """시간별 수요와 재생에너지 출력을 Bus 데이터에 반영한다.

    재생에너지는 별도 제어 발전기가 아니라 해당 Bus의 음(-)의 부하로
    표현한다. 따라서 DC-OPF가 공급해야 하는 순수요는
    `총수요 - 재생에너지 발전량`이 된다.
    """

    demand = hourly_demand.set_index("bus_id").sort_index()
    renewable = hourly_renewable.set_index("bus_id").sort_index()
    bus_ids = base_bus[:, BUS_I].astype(int)

    missing_demand = set(bus_ids) - set(demand.index)
    missing_renewable = set(bus_ids) - set(renewable.index)
    if missing_demand or missing_renewable:
        raise ValueError(
            "프로파일에서 일부 Bus를 찾지 못했습니다: "
            f"수요 누락 {sorted(missing_demand)} / "
            f"재생에너지 누락 {sorted(missing_renewable)}"
        )

    demand = demand.reindex(bus_ids)
    renewable = renewable.reindex(bus_ids)
    capacity = capacities.reindex(bus_ids).fillna(0.0)

    solar_mw = (
        renewable["pv_profile_ratio"].to_numpy()
        * capacity["solar"].to_numpy()
    )
    wind_mw = (
        renewable["wind_profile_ratio"].to_numpy()
        * capacity["wind"].to_numpy()
    )
    hydro_mw = (
        renewable["hydro_profile_ratio"].to_numpy()
        * capacity["hydro"].to_numpy()
    )
    renewable_mw = solar_mw + wind_mw + hydro_mw

    scenario_bus = base_bus.copy()
    gross_demand = demand["demandP"].to_numpy(dtype=float)
    scenario_bus[:, PD] = gross_demand - renewable_mw
    scenario_bus[:, QD] = demand["demandQ"].to_numpy(dtype=float)

    totals = {
        "gross_demand_mw": float(gross_demand.sum()),
        "solar_mw": float(solar_mw.sum()),
        "wind_mw": float(wind_mw.sum()),
        "hydro_mw": float(hydro_mw.sum()),
        "renewable_mw": float(renewable_mw.sum()),
        "net_demand_mw": float(scenario_bus[:, PD].sum()),
    }
    return scenario_bus, totals


def apply_hourly_commitment(
    base_gen: np.ndarray,
    hourly_commitment: pd.DataFrame,
    original_gen_count: int,
) -> tuple[np.ndarray, int]:
    """KPG의 시간별 발전기 ON/OFF 결정을 GEN_STATUS에 반영한다."""

    commitment = hourly_commitment.set_index("generator_id")["status"]
    expected_ids = np.arange(1, original_gen_count + 1)
    missing_ids = set(expected_ids) - set(commitment.index)
    if missing_ids:
        raise ValueError(
            "기동·정지 프로파일에서 발전기가 누락되었습니다: "
            f"{sorted(missing_ids)}"
        )

    hourly_gen = base_gen.copy()
    status = commitment.reindex(expected_ids).to_numpy(dtype=float)
    hourly_gen[:original_gen_count, GEN_STATUS] = status
    return hourly_gen, int(status.sum())


def ac_branch_loadings(result_branch: np.ndarray) -> np.ndarray:
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


def failed_row(hour: int, totals: dict[str, float], dc_success: bool) -> dict:
    return {
        "hour": hour,
        **totals,
        "dc_opf_success": dc_success,
        "ac_pf_converged": False,
        "min_voltage_pu": np.nan,
        "min_voltage_bus": np.nan,
        "min_voltage_bus_name": "",
        "voltage_violation_count": np.nan,
        "max_branch_loading_pct": np.nan,
        "max_loading_branch": np.nan,
        "max_loading_from_bus": np.nan,
        "max_loading_to_bus": np.nan,
        "max_loading_route": "",
        "overloaded_branch_count": np.nan,
        "generator_p_violation_count": np.nan,
        "generator_q_violation_count": np.nan,
        "ac_loss_mw": np.nan,
        "risk": True,
    }


def simulate_hour(
    hour: int,
    scenario_bus: np.ndarray,
    totals: dict[str, float],
    base_gen: np.ndarray,
    branch: np.ndarray,
    gencost: np.ndarray,
    original_gen_count: int,
    bus_names: dict[int, str],
) -> dict:
    """한 시간의 DC-OPF와 AC-PF 재검증 결과를 한 행으로 반환한다."""

    dc_case = {
        "version": "2",
        "baseMVA": 100.0,
        "bus": scenario_bus.copy(),
        "gen": base_gen.copy(),
        "branch": branch.copy(),
        "gencost": gencost.copy(),
    }
    dc_results = rundcopf(dc_case, ppoption(VERBOSE=0, OUT_ALL=0))
    dc_success = bool(dc_results["success"])
    if not dc_success:
        return failed_row(hour, totals, dc_success=False)

    ac_gen = base_gen.copy()
    ac_gen[:, PG] = dc_results["gen"][:, PG]
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
    if not converged:
        return failed_row(hour, totals, dc_success=True)

    result_bus = ac_results["bus"]
    result_gen = ac_results["gen"]
    result_branch = ac_results["branch"]

    min_voltage_index = int(np.argmin(result_bus[:, VM]))
    min_voltage_bus = int(result_bus[min_voltage_index, BUS_I])
    voltage_violation = (
        (result_bus[:, VM] < result_bus[:, VMIN] - LIMIT_TOLERANCE)
        | (result_bus[:, VM] > result_bus[:, VMAX] + LIMIT_TOLERANCE)
    )

    loading = ac_branch_loadings(result_branch)
    rated = np.flatnonzero(~np.isnan(loading))
    worst_branch = int(rated[np.argmax(loading[rated])])
    from_bus = int(result_branch[worst_branch, F_BUS])
    to_bus = int(result_branch[worst_branch, T_BUS])
    overloaded = loading > 100.0 + LIMIT_TOLERANCE

    conventional_gen = result_gen[:original_gen_count]
    active = conventional_gen[:, GEN_STATUS] > 0
    p_violation = active & (
        (conventional_gen[:, PG] > conventional_gen[:, PMAX] + LIMIT_TOLERANCE)
        | (conventional_gen[:, PG] < conventional_gen[:, PMIN] - LIMIT_TOLERANCE)
    )
    q_violation = active & (
        (conventional_gen[:, QG] > conventional_gen[:, QMAX] + LIMIT_TOLERANCE)
        | (conventional_gen[:, QG] < conventional_gen[:, QMIN] - LIMIT_TOLERANCE)
    )

    voltage_violation_count = int(np.sum(voltage_violation))
    overloaded_count = int(np.sum(overloaded))
    p_violation_count = int(np.sum(p_violation))
    q_violation_count = int(np.sum(q_violation))
    ac_loss_mw = float(result_gen[:, PG].sum() - totals["net_demand_mw"])
    risk = bool(
        voltage_violation_count
        or overloaded_count
        or p_violation_count
        or q_violation_count
    )

    return {
        "hour": hour,
        **totals,
        "dc_opf_success": True,
        "ac_pf_converged": True,
        "min_voltage_pu": float(result_bus[min_voltage_index, VM]),
        "min_voltage_bus": min_voltage_bus,
        "min_voltage_bus_name": bus_names.get(min_voltage_bus, ""),
        "voltage_violation_count": voltage_violation_count,
        "max_branch_loading_pct": float(loading[worst_branch]),
        "max_loading_branch": worst_branch + 1,
        "max_loading_from_bus": from_bus,
        "max_loading_to_bus": to_bus,
        "max_loading_route": (
            f"{bus_names.get(from_bus, f'B{from_bus}')} → "
            f"{bus_names.get(to_bus, f'B{to_bus}')}"
        ),
        "overloaded_branch_count": overloaded_count,
        "generator_p_violation_count": p_violation_count,
        "generator_q_violation_count": q_violation_count,
        "ac_loss_mw": ac_loss_mw,
        "risk": risk,
    }


def print_hour(row: dict) -> None:
    if not row["dc_opf_success"]:
        print(
            f"{row['hour']:02d}시 | 순수요 {row['net_demand_mw']:>8,.1f} MW | "
            "DC-OPF 실패"
        )
        return
    if not row["ac_pf_converged"]:
        print(
            f"{row['hour']:02d}시 | 순수요 {row['net_demand_mw']:>8,.1f} MW | "
            "AC-PF 미수렴"
        )
        return

    risk_label = "주의" if row["risk"] else "정상"
    print(
        f"{row['hour']:02d}시 | "
        f"수요 {row['gross_demand_mw']:>8,.1f} | "
        f"재생 {row['renewable_mw']:>8,.1f} | "
        f"순수요 {row['net_demand_mw']:>8,.1f} MW | "
        f"Vmin {row['min_voltage_pu']:.4f} | "
        f"선로 {row['max_branch_loading_pct']:>6.2f}% | "
        f"{risk_label}"
    )


def save_chart(results: pd.DataFrame, chart_path: Path) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print("그래프 생략: matplotlib가 설치되어 있지 않습니다.")
        return False

    hours = results["hour"]
    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(
        hours,
        results["gross_demand_mw"],
        marker="o",
        label="Gross demand",
    )
    axes[0].plot(
        hours,
        results["net_demand_mw"],
        marker="o",
        label="Net demand",
    )
    axes[0].plot(
        hours,
        results["renewable_mw"],
        marker="o",
        label="Renewables",
    )
    axes[0].set_ylabel("MW")
    axes[0].set_title("KPG-193 Day 5: 24-hour simulation")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=3)

    axes[1].plot(
        hours,
        results["max_branch_loading_pct"],
        color="tab:red",
        marker="o",
    )
    axes[1].axhline(100.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Max loading (%)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        hours,
        results["min_voltage_pu"],
        color="tab:green",
        marker="o",
    )
    axes[2].axhline(0.95, color="black", linestyle="--", linewidth=1)
    axes[2].set_ylabel("Min voltage (pu)")
    axes[2].set_xlabel("Hour")
    axes[2].set_xticks(range(1, HOURS_PER_DAY + 1))
    axes[2].grid(alpha=0.3)

    figure.tight_layout()
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(chart_path, dpi=160)
    plt.close(figure)
    return True


def print_summary(results: pd.DataFrame) -> None:
    valid = results[results["ac_pf_converged"]].copy()
    risk_hours = results.loc[results["risk"], "hour"].astype(int).tolist()

    print("\n" + "=" * 92)
    print("24시간 요약")
    print("=" * 92)
    print(
        f"DC-OPF 성공        : {int(results['dc_opf_success'].sum())}/24시간"
    )
    print(
        f"AC-PF 수렴         : {int(results['ac_pf_converged'].sum())}/24시간"
    )
    print(
        "주의 시간대        : "
        + (", ".join(f"{hour:02d}시" for hour in risk_hours) if risk_hours else "없음")
    )

    if valid.empty:
        print("수렴한 AC-PF 결과가 없어 상세 요약을 생략합니다.")
        return

    peak_net = valid.loc[valid["net_demand_mw"].idxmax()]
    peak_renewable = valid.loc[valid["renewable_mw"].idxmax()]
    worst_loading = valid.loc[valid["max_branch_loading_pct"].idxmax()]
    worst_voltage = valid.loc[valid["min_voltage_pu"].idxmin()]

    print(
        f"최대 순수요        : {int(peak_net['hour']):02d}시 / "
        f"{peak_net['net_demand_mw']:,.1f} MW"
    )
    print(
        f"최대 재생에너지    : {int(peak_renewable['hour']):02d}시 / "
        f"{peak_renewable['renewable_mw']:,.1f} MW"
    )
    print(
        f"최대 AC 선로 부하율: {int(worst_loading['hour']):02d}시 / "
        f"{worst_loading['max_branch_loading_pct']:.2f}% / "
        f"Branch {int(worst_loading['max_loading_branch'])} "
        f"{worst_loading['max_loading_route']}"
    )
    print(
        f"최저 전압          : {int(worst_voltage['hour']):02d}시 / "
        f"{worst_voltage['min_voltage_pu']:.5f} pu / "
        f"B{int(worst_voltage['min_voltage_bus'])} "
        f"{worst_voltage['min_voltage_bus_name']}"
    )


def main() -> None:
    args = parse_args()
    if not 1 <= args.day <= 365:
        raise ValueError("--day는 1~365 범위여야 합니다.")

    case_text = CASE_FILE.read_text(encoding="utf-8")
    bus = load_matpower_matrix(case_text, "bus")
    conventional_gen = load_matpower_matrix(case_text, "gen")
    branch = load_matpower_matrix(case_text, "branch")
    dcline = load_matpower_matrix(case_text, "dcline")
    gencost = load_matpower_matrix(case_text, "gencost")
    original_gen_count = len(conventional_gen)

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

    demand_profile, renewable_profile, commitment_profile = read_day_profiles(
        args.day
    )
    capacities = load_renewable_capacities()
    bus_names = load_bus_names()

    print("=" * 92)
    print(f"KPG-193 v2.0 Day 5 - {args.day}일차 24시간 시계열 시뮬레이션")
    print("=" * 92)
    print(
        f"Bus {len(bus)} / 기존 발전기 {original_gen_count} / "
        f"AC Branch {len(branch)} / 운전 중 HVDC {active_dc_lines}"
    )
    print(
        f"HVDC 처리: 양단 더미 발전기 {dummy_count}개의 P 고정 / "
        "재생에너지: Bus의 음(-)의 부하로 반영"
    )
    print(
        "매시간 계산: 수요·재생에너지·발전기 ON/OFF 반영 "
        "→ DC-OPF → AC-PF 재검증"
    )
    print("-" * 92)

    rows: list[dict] = []
    for hour in range(1, HOURS_PER_DAY + 1):
        scenario_bus, totals = build_hourly_bus(
            bus,
            demand_profile[demand_profile["hour"] == hour],
            renewable_profile[renewable_profile["hour"] == hour],
            capacities,
        )
        hourly_gen, online_generator_count = apply_hourly_commitment(
            gen,
            commitment_profile[commitment_profile["hour"] == hour],
            original_gen_count,
        )
        totals["online_generator_count"] = online_generator_count
        row = simulate_hour(
            hour,
            scenario_bus,
            totals,
            hourly_gen,
            branch,
            gencost,
            original_gen_count,
            bus_names,
        )
        rows.append(row)
        print_hour(row)

    results = pd.DataFrame(rows)
    print_summary(results)

    default_output = SCRIPT_DIR / f"kpg_day05_day{args.day:03d}_24h_results.csv"
    output_path = (args.output or default_output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장            : {output_path}")

    if not args.no_chart:
        chart_path = output_path.with_suffix(".png")
        if save_chart(results, chart_path):
            print(f"그래프 저장         : {chart_path}")

    print("\n[Day 5 결론]")
    print(
        "시간대별 수요와 재생에너지 변화는 순수요·전압·선로 혼잡을 바꾸며, "
        "이 반복 계산 결과를 AI EMS의 학습·평가 데이터로 사용할 수 있다."
    )
    print(
        "DC-OPF와 AC-PF 결과가 모두 있어도 N-1 안전까지 보장하는 것은 아니다."
    )


if __name__ == "__main__":
    main()
