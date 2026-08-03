"""KPG-193 AC 조류계산 결과를 대화형 계통 지도로 만든다.

필요 패키지:
    python -m pip install "numpy<2.4" scipy pypower

실행:
    python kpg_map.py

결과:
    같은 폴더의 kpg_map.html을 생성하고 기본 브라우저로 연다.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import webbrowser
from pathlib import Path

import numpy as np

# PYPOWER 5.1.19는 NumPy 2.4에서 제거된 이름을 아직 가져온다.
# 사용자가 최신 NumPy를 쓰더라도 지도 스크립트는 실행할 수 있도록 호환 별칭을 둔다.
if not hasattr(np, "in1d"):
    np.in1d = np.isin  # type: ignore[attr-defined]

from pypower.api import ppoption, runpf

from day01.kpg_day01 import (
    BUS_METADATA_FILE,
    CASE_FILE,
    add_fixed_hvdc_as_dummy_generators,
    load_matpower_matrix,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "kpg_map.html"


def load_bus_metadata() -> dict[int, dict[str, object]]:
    """Bus 번호별 위치와 한·영문 이름을 읽는다."""

    with BUS_METADATA_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return {
            int(row["bus_id"]): {
                "lat": float(row["Latitude"]),
                "lon": float(row["Longitude"]),
                "name_ko": row["name_Korean"],
                "name_en": row["name_English"],
            }
            for row in rows
        }


def solve_kpg_case() -> tuple[dict[str, object], bool, int, np.ndarray]:
    """KPG-193 기본 상태의 AC 조류계산을 수행한다."""

    case_text = CASE_FILE.read_text(encoding="utf-8")
    bus = load_matpower_matrix(case_text, "bus")
    gen = load_matpower_matrix(case_text, "gen")
    branch = load_matpower_matrix(case_text, "branch")
    dcline = load_matpower_matrix(case_text, "dcline")
    original_gen_count = len(gen)

    bus, gen, _ = add_fixed_hvdc_as_dummy_generators(
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
    return results, bool(converged), original_gen_count, dcline


def round_or_none(value: float, digits: int = 3) -> float | None:
    if not math.isfinite(value):
        return None
    return round(float(value), digits)


def build_map_data() -> dict[str, object]:
    """브라우저 지도에 필요한 Bus·선로·요약 결과를 구성한다."""

    metadata = load_bus_metadata()
    results, converged, original_gen_count, dcline = solve_kpg_case()

    if not converged:
        raise RuntimeError("AC 조류계산이 수렴하지 않아 지도를 만들 수 없습니다.")

    bus = np.asarray(results["bus"])
    gen = np.asarray(results["gen"])
    branch = np.asarray(results["branch"])
    bus_by_id = {int(row[0]): row for row in bus}

    gross_load_mw = float(np.maximum(bus[:, 2], 0).sum())
    fixed_generation_mw = float(-np.minimum(bus[:, 2], 0).sum())
    net_demand_mw = float(bus[:, 2].sum())
    conventional_generation_mw = float(gen[:original_gen_count, 1].sum())
    ac_loss_mw = float(gen[:, 1].sum() - net_demand_mw)

    buses: list[dict[str, object]] = []
    for row in bus:
        bus_id = int(row[0])
        meta = metadata.get(bus_id)
        if meta is None:
            raise ValueError(f"Bus {bus_id}의 위치 메타데이터가 없습니다.")

        buses.append(
            {
                "id": bus_id,
                "name": meta["name_ko"],
                "nameEn": meta["name_en"],
                "lat": round(float(meta["lat"]), 6),
                "lon": round(float(meta["lon"]), 6),
                "load": round(float(row[2]), 2),
                "reactiveLoad": round(float(row[3]), 2),
                "voltage": round(float(row[7]), 5),
                "angle": round(float(row[8]), 3),
                "baseKv": round(float(row[9]), 1),
            }
        )

    # PYPOWER branch 결과: PF=13, QF=14, PT=15, QT=16 (0-based)
    rate_a = branch[:, 5]
    apparent_from = np.hypot(branch[:, 13], branch[:, 14])
    apparent_to = np.hypot(branch[:, 15], branch[:, 16])
    loading = np.full(len(branch), np.nan)
    rated = rate_a > 0
    loading[rated] = (
        np.maximum(apparent_from[rated], apparent_to[rated])
        / rate_a[rated]
        * 100
    )

    branches: list[dict[str, object]] = []
    for index, row in enumerate(branch):
        from_bus = int(row[0])
        to_bus = int(row[1])
        from_meta = metadata[from_bus]
        to_meta = metadata[to_bus]
        base_kv = max(
            float(bus_by_id[from_bus][9]),
            float(bus_by_id[to_bus][9]),
        )

        branches.append(
            {
                "id": index + 1,
                "from": from_bus,
                "to": to_bus,
                "fromName": from_meta["name_ko"],
                "toName": to_meta["name_ko"],
                "fromLon": round(float(from_meta["lon"]), 6),
                "fromLat": round(float(from_meta["lat"]), 6),
                "toLon": round(float(to_meta["lon"]), 6),
                "toLat": round(float(to_meta["lat"]), 6),
                "baseKv": round(base_kv, 1),
                "rate": round_or_none(float(row[5]), 1),
                "loading": round_or_none(float(loading[index]), 2),
                "pf": round(float(row[13]), 2),
                "qf": round(float(row[14]), 2),
                "pt": round(float(row[15]), 2),
                "qt": round(float(row[16]), 2),
            }
        )

    dc_lines: list[dict[str, object]] = []
    for index, row in enumerate(dcline):
        if row[2] <= 0:
            continue
        from_bus = int(row[0])
        to_bus = int(row[1])
        from_meta = metadata[from_bus]
        to_meta = metadata[to_bus]
        dc_lines.append(
            {
                "id": index + 1,
                "from": from_bus,
                "to": to_bus,
                "fromName": from_meta["name_ko"],
                "toName": to_meta["name_ko"],
                "fromLon": round(float(from_meta["lon"]), 6),
                "fromLat": round(float(from_meta["lat"]), 6),
                "toLon": round(float(to_meta["lon"]), 6),
                "toLat": round(float(to_meta["lat"]), 6),
                "pf": round(float(row[3]), 2),
                "pt": round(float(row[4]), 2),
            }
        )

    min_bus = min(buses, key=lambda item: float(item["voltage"]))
    max_bus = max(buses, key=lambda item: float(item["voltage"]))
    rated_branches = [
        item for item in branches if item["loading"] is not None
    ]
    max_branch = max(
        rated_branches,
        key=lambda item: float(item["loading"]),
    )

    return {
        "summary": {
            "converged": converged,
            "busCount": len(buses),
            "generatorCount": original_gen_count,
            "branchCount": len(branches),
            "dcLineCount": len(dc_lines),
            "grossLoadMw": round(gross_load_mw, 1),
            "fixedGenerationMw": round(fixed_generation_mw, 1),
            "netDemandMw": round(net_demand_mw, 1),
            "generationMw": round(conventional_generation_mw, 1),
            "lossMw": round(ac_loss_mw, 1),
            "lossPercent": round(ac_loss_mw / net_demand_mw * 100, 2),
            "minVoltageBus": min_bus,
            "maxVoltageBus": max_bus,
            "maxBranch": max_branch,
        },
        "buses": buses,
        "branches": branches,
        "dcLines": dc_lines,
    }


FRAGMENT_TEMPLATE = r"""
<div id="kpg-grid-map-root">
  <div class="viz-grid kpg-summary" aria-label="계통 요약">
    <div class="card viz-stat">
      <div class="text-muted">순수요</div>
      <div class="viz-stat-value" id="kpg-demand"></div>
      <div class="text-small text-muted">AC 조류계산 기준</div>
    </div>
    <div class="card viz-stat">
      <div class="text-muted">AC 손실</div>
      <div class="viz-stat-value" id="kpg-loss"></div>
      <div class="text-small text-muted" id="kpg-loss-rate"></div>
    </div>
    <div class="card viz-stat">
      <div class="text-muted">최대 선로 부하율</div>
      <div class="viz-stat-value" id="kpg-max-loading"></div>
      <div class="text-small text-muted" id="kpg-max-branch"></div>
    </div>
  </div>

  <div class="viz-controls" aria-label="선로 색상 기준">
    <button type="button" class="btn btn-primary" id="kpg-mode-loading" aria-pressed="true">
      부하율
    </button>
    <button type="button" class="btn" id="kpg-mode-voltage" aria-pressed="false">
      전압 등급
    </button>
    <span class="text-small text-muted" id="kpg-map-status" aria-live="polite"></span>
  </div>

  <div class="kpg-map-stage">
    <svg id="kpg-map-svg" viewBox="0 0 720 760" role="img"
         aria-labelledby="kpg-map-title kpg-map-desc">
      <title id="kpg-map-title">KPG-193 AC 조류계산 계통 지도</title>
      <desc id="kpg-map-desc">
        193개 Bus의 위치, 전압, AC 선로 부하율과 HVDC 연결을 표시한다.
      </desc>
      <g id="kpg-basemap-layer"></g>
      <g id="kpg-ac-layer"></g>
      <g id="kpg-dc-layer"></g>
      <g id="kpg-bus-layer"></g>
    </svg>
    <div id="kpg-tooltip" class="tooltip kpg-tooltip" hidden></div>
  </div>

  <div class="viz-row kpg-legend" id="kpg-legend" aria-label="범례"></div>

  <div class="card kpg-selection" id="kpg-selection" aria-live="polite">
    선로나 Bus를 선택하면 상세 결과가 표시됩니다.
  </div>
</div>

<style>
  #kpg-grid-map-root {
    color: var(--foreground);
    width: 100%;
  }
  #kpg-grid-map-root .kpg-summary {
    margin-bottom: 0.75rem;
  }
  #kpg-grid-map-root .viz-controls {
    margin-bottom: 0.5rem;
  }
  #kpg-grid-map-root .kpg-map-stage {
    position: relative;
    width: 100%;
  }
  #kpg-grid-map-root #kpg-map-svg {
    display: block;
    width: 100%;
    height: auto;
    max-height: 760px;
    touch-action: manipulation;
  }
  #kpg-grid-map-root .kpg-land {
    fill: color-mix(in srgb, var(--muted) 72%, transparent);
    stroke: var(--border);
    stroke-width: 1.2;
  }
  #kpg-grid-map-root .kpg-border {
    fill: none;
    stroke: var(--border);
    stroke-width: 0.65;
  }
  #kpg-grid-map-root .kpg-branch {
    fill: none;
    stroke-linecap: round;
    cursor: pointer;
    opacity: 0.78;
    transition: opacity 120ms ease, stroke-width 120ms ease;
  }
  #kpg-grid-map-root .kpg-branch:hover,
  #kpg-grid-map-root .kpg-branch.is-selected {
    opacity: 1;
    stroke-width: 5;
  }
  #kpg-grid-map-root .kpg-dcline {
    fill: none;
    stroke: var(--viz-series-5);
    stroke-width: 3;
    stroke-dasharray: 7 4;
    cursor: pointer;
  }
  #kpg-grid-map-root .kpg-bus {
    stroke: var(--background);
    stroke-width: 1.2;
    cursor: pointer;
    transition: r 120ms ease, stroke-width 120ms ease;
  }
  #kpg-grid-map-root .kpg-bus:hover,
  #kpg-grid-map-root .kpg-bus.is-selected {
    r: 7;
    stroke-width: 2.5;
  }
  #kpg-grid-map-root .kpg-tooltip {
    position: absolute;
    pointer-events: none;
    max-width: 260px;
    z-index: 2;
  }
  #kpg-grid-map-root .kpg-legend {
    justify-content: center;
    margin-top: 0.35rem;
    margin-bottom: 0.75rem;
  }
  #kpg-grid-map-root .kpg-legend-item {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
  }
  #kpg-grid-map-root .kpg-swatch {
    width: 1.5rem;
    height: 0.25rem;
    border-radius: 999px;
    background: var(--swatch);
  }
  #kpg-grid-map-root .kpg-selection {
    min-height: 2.5rem;
  }
  @media (prefers-reduced-motion: reduce) {
    #kpg-grid-map-root .kpg-branch,
    #kpg-grid-map-root .kpg-bus {
      transition: none;
    }
  }
</style>

<script type="module">
const root = document.getElementById("kpg-grid-map-root");
const data = __KPG_DATA__;
const svg = root.querySelector("#kpg-map-svg");
const basemapLayer = root.querySelector("#kpg-basemap-layer");
const acLayer = root.querySelector("#kpg-ac-layer");
const dcLayer = root.querySelector("#kpg-dc-layer");
const busLayer = root.querySelector("#kpg-bus-layer");
const legend = root.querySelector("#kpg-legend");
const selection = root.querySelector("#kpg-selection");
const tooltip = root.querySelector("#kpg-tooltip");
const mapStatus = root.querySelector("#kpg-map-status");
const loadingButton = root.querySelector("#kpg-mode-loading");
const voltageButton = root.querySelector("#kpg-mode-voltage");
const ns = "http://www.w3.org/2000/svg";

let mode = "loading";
let projection = null;
let selectedElement = null;

const css = name =>
  getComputedStyle(root).getPropertyValue(name).trim();

const series = {
  one: css("--viz-series-1"),
  two: css("--viz-series-2"),
  three: css("--viz-series-3"),
  four: css("--viz-series-4"),
  five: css("--viz-series-5"),
  six: css("--viz-series-6"),
  danger: css("--destructive"),
  border: css("--border"),
  background: css("--background")
};

const summary = data.summary;
root.querySelector("#kpg-demand").textContent =
  `${summary.netDemandMw.toLocaleString("ko-KR")} MW`;
root.querySelector("#kpg-loss").textContent =
  `${summary.lossMw.toLocaleString("ko-KR")} MW`;
root.querySelector("#kpg-loss-rate").textContent =
  `순수요의 ${summary.lossPercent.toFixed(2)}%`;
root.querySelector("#kpg-max-loading").textContent =
  `${summary.maxBranch.loading.toFixed(2)}%`;
root.querySelector("#kpg-max-branch").textContent =
  `${summary.maxBranch.fromName} → ${summary.maxBranch.toName}`;

function fallbackProjection([lon, lat]) {
  const lons = data.buses.map(bus => bus.lon);
  const lats = data.buses.map(bus => bus.lat);
  const minLon = Math.min(...lons) - 0.35;
  const maxLon = Math.max(...lons) + 0.35;
  const minLat = Math.min(...lats) - 0.25;
  const maxLat = Math.max(...lats) + 0.25;
  const x = 55 + ((lon - minLon) / (maxLon - minLon)) * 610;
  const y = 710 - ((lat - minLat) / (maxLat - minLat)) * 660;
  return [x, y];
}

function point(lon, lat) {
  return projection ? projection([lon, lat]) : fallbackProjection([lon, lat]);
}

function createSvgElement(tag, attributes) {
  const element = document.createElementNS(ns, tag);
  Object.entries(attributes).forEach(([key, value]) => {
    element.setAttribute(key, String(value));
  });
  return element;
}

function loadingColor(value) {
  if (value === null) return series.border;
  if (value >= 100) return series.danger;
  if (value >= 90) return series.four;
  if (value >= 80) return series.three;
  return series.two;
}

function voltageColor(kv) {
  if (kv >= 700) return series.danger;
  if (kv >= 300) return series.one;
  return series.border;
}

function busColor(voltage) {
  if (voltage < 0.98) return series.four;
  if (voltage > 1.02) return series.three;
  return series.one;
}

function setSelected(element, text) {
  if (selectedElement) selectedElement.classList.remove("is-selected");
  selectedElement = element;
  selectedElement.classList.add("is-selected");
  selection.textContent = text;
}

function positionTooltip(event, text) {
  tooltip.textContent = text;
  tooltip.hidden = false;
  const stageRect = svg.parentElement.getBoundingClientRect();
  const tipRect = tooltip.getBoundingClientRect();
  const x = Math.min(
    Math.max(event.clientX - stageRect.left + 12, 4),
    stageRect.width - tipRect.width - 4
  );
  const y = Math.min(
    Math.max(event.clientY - stageRect.top + 12, 4),
    stageRect.height - tipRect.height - 4
  );
  tooltip.style.left = `${x}px`;
  tooltip.style.top = `${y}px`;
}

function hideTooltip() {
  tooltip.hidden = true;
}

function branchText(branch) {
  const loading = branch.loading === null
    ? "정격정보 없음"
    : `부하율 ${branch.loading.toFixed(2)}%`;
  return `Branch ${branch.id} · ${branch.fromName} → ${branch.toName} · ` +
    `${branch.baseKv.toFixed(0)} kV · ${loading} · ` +
    `송단 ${branch.pf.toFixed(1)} MW / ${branch.qf.toFixed(1)} Mvar`;
}

function busText(bus) {
  return `Bus ${bus.id} · ${bus.name} · ${bus.baseKv.toFixed(0)} kV · ` +
    `${bus.voltage.toFixed(5)} pu · ${bus.angle.toFixed(3)}° · ` +
    `부하 ${bus.load.toFixed(1)} MW`;
}

function dcText(line) {
  return `HVDC ${line.id} · ${line.fromName} → ${line.toName} · ` +
    `송단 ${line.pf.toFixed(1)} MW · 수단 ${line.pt.toFixed(1)} MW`;
}

function drawNetwork() {
  acLayer.replaceChildren();
  dcLayer.replaceChildren();
  busLayer.replaceChildren();

  data.branches.forEach(branch => {
    const [x1, y1] = point(branch.fromLon, branch.fromLat);
    const [x2, y2] = point(branch.toLon, branch.toLat);
    const element = createSvgElement("line", {
      x1,
      y1,
      x2,
      y2,
      class: "kpg-branch",
      stroke: mode === "loading"
        ? loadingColor(branch.loading)
        : voltageColor(branch.baseKv),
      "stroke-width": branch.loading !== null && branch.loading >= 90 ? 3.2 : 1.7
    });
    const text = branchText(branch);
    element.addEventListener("mouseenter", event => positionTooltip(event, text));
    element.addEventListener("mousemove", event => positionTooltip(event, text));
    element.addEventListener("mouseleave", hideTooltip);
    element.addEventListener("click", () => setSelected(element, text));
    acLayer.appendChild(element);
  });

  data.dcLines.forEach(line => {
    const [x1, y1] = point(line.fromLon, line.fromLat);
    const [x2, y2] = point(line.toLon, line.toLat);
    const element = createSvgElement("line", {
      x1,
      y1,
      x2,
      y2,
      class: "kpg-dcline"
    });
    const text = dcText(line);
    element.addEventListener("mouseenter", event => positionTooltip(event, text));
    element.addEventListener("mousemove", event => positionTooltip(event, text));
    element.addEventListener("mouseleave", hideTooltip);
    element.addEventListener("click", () => setSelected(element, text));
    dcLayer.appendChild(element);
  });

  data.buses.forEach(bus => {
    const [cx, cy] = point(bus.lon, bus.lat);
    const radius = 3.1 + Math.min(2.7, Math.sqrt(Math.max(bus.load, 0)) / 18);
    const element = createSvgElement("circle", {
      cx,
      cy,
      r: radius,
      class: "kpg-bus",
      fill: busColor(bus.voltage)
    });
    const text = busText(bus);
    element.addEventListener("mouseenter", event => positionTooltip(event, text));
    element.addEventListener("mousemove", event => positionTooltip(event, text));
    element.addEventListener("mouseleave", hideTooltip);
    element.addEventListener("click", () => setSelected(element, text));
    busLayer.appendChild(element);
  });
}

function updateLegend() {
  const items = mode === "loading"
    ? [
        ["80% 미만", series.two],
        ["80–90%", series.three],
        ["90–100%", series.four],
        ["100% 이상", series.danger],
        ["HVDC", series.five]
      ]
    : [
        ["154 kV", series.border],
        ["345 kV", series.one],
        ["765 kV", series.danger],
        ["HVDC", series.five]
      ];

  legend.replaceChildren();
  items.forEach(([label, color]) => {
    const item = document.createElement("span");
    item.className = "kpg-legend-item text-small";
    const swatch = document.createElement("span");
    swatch.className = "kpg-swatch";
    swatch.style.setProperty("--swatch", color);
    const text = document.createElement("span");
    text.textContent = label;
    item.append(swatch, text);
    legend.appendChild(item);
  });
}

function setMode(nextMode) {
  mode = nextMode;
  const isLoading = mode === "loading";
  loadingButton.classList.toggle("btn-primary", isLoading);
  voltageButton.classList.toggle("btn-primary", !isLoading);
  loadingButton.setAttribute("aria-pressed", String(isLoading));
  voltageButton.setAttribute("aria-pressed", String(!isLoading));
  drawNetwork();
  updateLegend();
}

loadingButton.addEventListener("click", () => setMode("loading"));
voltageButton.addEventListener("click", () => setMode("voltage"));

async function initializeMap() {
  try {
    const [atlasModule, topoModule, geoModule] = await Promise.all([
      import("https://esm.sh/@d3-maps/atlas@1.0.0/world/countries/countries-110m"),
      import("https://esm.sh/topojson-client@3.1.0"),
      import("https://esm.sh/d3-geo@3.1.1")
    ]);
    const world = atlasModule.default;
    const countries = topoModule.feature(
      world,
      world.objects.features
    ).features;
    const korea = countries.find(item => item.properties.id === "KOR");
    if (!korea) throw new Error("대한민국 경계를 찾지 못했습니다.");

    projection = geoModule.geoMercator().fitExtent(
      [[42, 30], [678, 725]],
      korea
    );
    const path = geoModule.geoPath(projection);
    const land = createSvgElement("path", {
      d: path(korea),
      class: "kpg-land"
    });
    basemapLayer.appendChild(land);
    mapStatus.textContent = "대한민국 경계와 계통을 표시했습니다.";
  } catch (error) {
    projection = null;
    mapStatus.textContent = "오프라인 계통도 모드";
  }

  drawNetwork();
  updateLegend();
}

initializeMap();
</script>
""".strip()


STANDALONE_PREFIX = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KPG-193 AC 조류계산 지도</title>
  <style>
    :root {
      color-scheme: light;
      --background: #f7f9fc;
      --foreground: #172033;
      --card: #ffffff;
      --card-foreground: #172033;
      --muted: #dce5ef;
      --muted-foreground: #5f6c80;
      --border: #8794a8;
      --primary: #23395d;
      --primary-foreground: #ffffff;
      --destructive: #d92d20;
      --viz-series-1: #2563eb;
      --viz-series-2: #1f9d75;
      --viz-series-3: #e3a008;
      --viz-series-4: #f97316;
      --viz-series-5: #b832d9;
      --viz-series-6: #6b7280;
      font-family: "Segoe UI", "Malgun Gothic", sans-serif;
    }
    @media (prefers-color-scheme: dark) {
      :root {
        color-scheme: dark;
        --background: #101522;
        --foreground: #e7edf7;
        --card: #171f2f;
        --card-foreground: #e7edf7;
        --muted: #273349;
        --muted-foreground: #aab6c9;
        --border: #66758d;
        --primary: #dbe7ff;
        --primary-foreground: #172033;
        --destructive: #ff6257;
        --viz-series-1: #67a0ff;
        --viz-series-2: #5bd1a5;
        --viz-series-3: #f4c94d;
        --viz-series-4: #ff9858;
        --viz-series-5: #df78f4;
        --viz-series-6: #aab6c9;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 24px;
      color: var(--foreground);
      background: var(--background);
    }
    main { max-width: 980px; margin: 0 auto; }
    h1 { margin: 0 0 16px; font-size: 1.5rem; font-weight: 500; }
    .viz-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }
    .card {
      padding: 12px 14px;
      color: var(--card-foreground);
      background: var(--card);
      border: 1px solid color-mix(in srgb, var(--border) 45%, transparent);
      border-radius: 10px;
    }
    .viz-stat-value { margin: 4px 0; font-size: 1.35rem; font-weight: 500; }
    .text-muted { color: var(--muted-foreground); }
    .text-small { font-size: 0.82rem; }
    .viz-controls, .viz-row {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
    }
    .btn {
      appearance: none;
      padding: 7px 12px;
      color: var(--foreground);
      background: transparent;
      border: 1px solid var(--border);
      border-radius: 8px;
      cursor: pointer;
      font: inherit;
    }
    .btn-primary {
      color: var(--primary-foreground);
      background: var(--primary);
      border-color: var(--primary);
    }
    .tooltip {
      padding: 7px 9px;
      color: var(--card-foreground);
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 7px;
      box-shadow: 0 8px 24px color-mix(in srgb, var(--foreground) 16%, transparent);
      font-size: 0.82rem;
    }
    @media (max-width: 540px) {
      body { padding: 12px; }
    }
  </style>
</head>
<body>
  <main>
    <h1>KPG-193 AC 조류계산 지도</h1>
"""

STANDALONE_SUFFIX = """
  </main>
</body>
</html>
"""


def render_fragment(data: dict[str, object]) -> str:
    encoded = json.dumps(
        data,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return FRAGMENT_TEMPLATE.replace("__KPG_DATA__", encoded)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KPG-193 AC 조류계산 결과를 대화형 지도로 생성합니다."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="생성할 HTML 경로",
    )
    parser.add_argument(
        "--fragment-output",
        type=Path,
        help="미리보기용 HTML fragment도 저장",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="생성 후 브라우저를 열지 않음",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = build_map_data()
    fragment = render_fragment(data)
    standalone = STANDALONE_PREFIX + fragment + STANDALONE_SUFFIX

    output = args.output.resolve()
    output.write_text(standalone, encoding="utf-8")

    if args.fragment_output:
        fragment_output = args.fragment_output.resolve()
        fragment_output.write_text(fragment, encoding="utf-8")

    summary = data["summary"]
    assert isinstance(summary, dict)
    max_branch = summary["maxBranch"]
    assert isinstance(max_branch, dict)

    print("=" * 72)
    print("KPG-193 AC 조류계산 지도 생성 완료")
    print("=" * 72)
    print(f"출력 파일       : {output}")
    print(f"Bus / AC / HVDC : {summary['busCount']} / "
          f"{summary['branchCount']} / {summary['dcLineCount']}")
    print(f"최대 선로 부하율: {max_branch['fromName']} → "
          f"{max_branch['toName']} {max_branch['loading']:.2f}%")

    if not args.no_open:
        webbrowser.open(output.as_uri())


if __name__ == "__main__":
    main()
