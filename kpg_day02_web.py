"""KPG-193 N-1 결과 CSV를 오프라인 HTML 대시보드로 변환한다.

실행:
    python kpg_day02_web.py
    python kpg_day02_web.py --input kpg_day02_n-1_results.csv

생성:
    kpg_day02_dashboard.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kpg_day01 import (
    BUS_METADATA_FILE,
    CASE_FILE,
    load_matpower_matrix,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_FILE = ROOT / "kpg_day02_n-1_results.csv"
DEFAULT_OUTPUT_FILE = ROOT / "kpg_day02_dashboard.html"


HTML_TEMPLATE = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KPG-193 N-1 사고 분석</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --panel-soft: #f7f9fc;
      --text: #182231;
      --muted: #617087;
      --border: #d7deea;
      --grid: #aab5c5;
      --bus: #52657d;
      --outage: #7c3aed;
      --worst: #dc2626;
      --voltage: #d97706;
      --ok: #15803d;
      --warning: #b45309;
      --danger: #b91c1c;
      --shadow: 0 10px 28px rgba(15, 23, 42, 0.08);
    }

    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0e1521;
        --panel: #151f2e;
        --panel-soft: #111a27;
        --text: #e8eef7;
        --muted: #aab7ca;
        --border: #314057;
        --grid: #586880;
        --bus: #9aa9bd;
        --outage: #a78bfa;
        --worst: #fb7185;
        --voltage: #fbbf24;
        --ok: #4ade80;
        --warning: #fbbf24;
        --danger: #fb7185;
        --shadow: 0 12px 32px rgba(0, 0, 0, 0.25);
      }
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family:
        Pretendard, "Noto Sans KR", "Malgun Gothic", system-ui, sans-serif;
    }

    button,
    select {
      font: inherit;
    }

    .page {
      width: min(1500px, 100%);
      margin: 0 auto;
      padding: 24px;
    }

    .heading {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 16px;
      margin-bottom: 18px;
    }

    h1 {
      margin: 0 0 5px;
      font-size: clamp(24px, 3vw, 36px);
      font-weight: 700;
    }

    .subtitle {
      margin: 0;
      color: var(--muted);
    }

    .control-panel {
      display: grid;
      grid-template-columns: minmax(150px, 0.7fr) minmax(280px, 2fr) auto auto;
      gap: 10px;
      padding: 14px;
      margin-bottom: 14px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .field {
      min-width: 0;
    }

    .field label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 600;
    }

    select,
    button {
      min-height: 42px;
      border: 1px solid var(--border);
      border-radius: 9px;
      background: var(--panel-soft);
      color: var(--text);
    }

    select {
      width: 100%;
      padding: 0 11px;
    }

    button {
      align-self: end;
      padding: 0 16px;
      cursor: pointer;
    }

    button:hover {
      border-color: var(--grid);
    }

    button:focus-visible,
    select:focus-visible {
      outline: 3px solid color-mix(in srgb, var(--outage) 45%, transparent);
      outline-offset: 2px;
    }

    .stats {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }

    .stat {
      min-width: 0;
      padding: 15px 17px;
      border: 1px solid var(--border);
      border-radius: 13px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .stat-label {
      color: var(--muted);
      font-size: 13px;
    }

    .stat-value {
      margin-top: 6px;
      overflow: hidden;
      font-size: clamp(20px, 2.5vw, 29px);
      font-weight: 700;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .layout {
      display: grid;
      grid-template-columns: minmax(0, 2.2fr) minmax(300px, 0.9fr);
      gap: 14px;
      align-items: start;
    }

    .map-panel,
    .detail-panel {
      border: 1px solid var(--border);
      border-radius: 14px;
      background: var(--panel);
      box-shadow: var(--shadow);
    }

    .map-panel {
      padding: 12px;
    }

    #network-map {
      display: block;
      width: 100%;
      height: min(72vh, 840px);
      min-height: 540px;
      border-radius: 10px;
      background:
        radial-gradient(circle at 50% 40%, var(--panel-soft), var(--panel));
    }

    .base-branch {
      stroke: var(--grid);
      stroke-width: 0.7;
      opacity: 0.42;
      vector-effect: non-scaling-stroke;
    }

    .base-bus {
      fill: var(--bus);
      opacity: 0.68;
    }

    .outage-line {
      stroke: var(--outage);
      stroke-width: 5;
      stroke-dasharray: 10 6;
      vector-effect: non-scaling-stroke;
    }

    .worst-line {
      stroke: var(--worst);
      stroke-width: 5;
      vector-effect: non-scaling-stroke;
    }

    .voltage-bus {
      fill: var(--voltage);
      stroke: var(--panel);
      stroke-width: 3;
      vector-effect: non-scaling-stroke;
    }

    .map-label {
      fill: var(--text);
      paint-order: stroke;
      stroke: var(--panel);
      stroke-width: 4px;
      stroke-linejoin: round;
      font-size: 13px;
      font-weight: 700;
    }

    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 9px 18px;
      padding: 10px 4px 2px;
      color: var(--muted);
      font-size: 13px;
    }

    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }

    .swatch {
      width: 25px;
      height: 4px;
      border-radius: 4px;
      background: var(--grid);
    }

    .swatch.outage {
      background: var(--outage);
    }

    .swatch.worst {
      background: var(--worst);
    }

    .swatch.voltage {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--voltage);
    }

    .detail-panel {
      padding: 18px;
    }

    .detail-panel h2 {
      margin: 0 0 6px;
      font-size: 21px;
    }

    .status-line {
      margin-bottom: 17px;
      font-weight: 700;
    }

    .status-ok {
      color: var(--ok);
    }

    .status-warning {
      color: var(--warning);
    }

    .status-danger {
      color: var(--danger);
    }

    dl {
      margin: 0;
    }

    .detail-row {
      display: grid;
      grid-template-columns: 125px minmax(0, 1fr);
      gap: 12px;
      padding: 11px 0;
      border-bottom: 1px solid var(--border);
    }

    .detail-row:last-child {
      border-bottom: 0;
    }

    dt {
      color: var(--muted);
    }

    dd {
      min-width: 0;
      margin: 0;
      overflow-wrap: anywhere;
      font-weight: 600;
    }

    .note {
      margin: 16px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }

    @media (max-width: 900px) {
      .control-panel {
        grid-template-columns: 1fr 2fr;
      }

      .stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }

      .layout {
        grid-template-columns: 1fr;
      }

      #network-map {
        height: 680px;
      }
    }

    @media (max-width: 580px) {
      .page {
        padding: 14px;
      }

      .heading {
        display: block;
      }

      .control-panel {
        grid-template-columns: 1fr 1fr;
      }

      .field.scenario-field {
        grid-column: 1 / -1;
      }

      .stats {
        grid-template-columns: 1fr 1fr;
      }

      .stat {
        padding: 13px;
      }

      #network-map {
        height: 560px;
        min-height: 500px;
      }

      .detail-row {
        grid-template-columns: 105px minmax(0, 1fr);
      }
    }
  </style>
</head>
<body>
  <main class="page">
    <header class="heading">
      <div>
        <h1>KPG-193 N-1 사고 분석</h1>
        <p class="subtitle">AC Branch 탈락 사고 385건의 정적 AC 조류계산 결과</p>
      </div>
    </header>

    <section class="control-panel" aria-label="사고 선택">
      <div class="field">
        <label for="status-filter">상태 필터</label>
        <select id="status-filter">
          <option value="ALL">전체</option>
          <option value="LIMIT_VIOLATION">한계 위반</option>
          <option value="ISLANDING">계통 분리</option>
          <option value="OK">한계 내</option>
          <option value="NON_CONVERGED">비수렴</option>
          <option value="ERROR">오류</option>
        </select>
      </div>

      <div class="field scenario-field">
        <label for="scenario-select">Branch 탈락 사고</label>
        <select id="scenario-select"></select>
      </div>

      <button id="prev-button" type="button">이전</button>
      <button id="next-button" type="button">다음</button>
    </section>

    <section class="stats" aria-label="선택 사고 요약">
      <article class="stat">
        <div class="stat-label">사고 상태</div>
        <div class="stat-value" id="metric-status">-</div>
      </article>
      <article class="stat">
        <div class="stat-label">사고 후 최대 부하율</div>
        <div class="stat-value" id="metric-loading">-</div>
      </article>
      <article class="stat">
        <div class="stat-label">사고 후 최저 전압</div>
        <div class="stat-value" id="metric-voltage">-</div>
      </article>
      <article class="stat">
        <div class="stat-label">한계 위반</div>
        <div class="stat-value" id="metric-violations">-</div>
      </article>
    </section>

    <section class="layout">
      <div class="map-panel">
        <svg id="network-map" viewBox="0 0 780 900" role="img"
             aria-label="KPG-193 계통과 선택한 사고 위치">
          <g id="base-branches"></g>
          <g id="base-buses"></g>
          <g id="highlights"></g>
          <g id="labels"></g>
        </svg>

        <div class="legend" aria-label="지도 범례">
          <span class="legend-item"><span class="swatch"></span>기존 계통</span>
          <span class="legend-item"><span class="swatch outage"></span>탈락 선로</span>
          <span class="legend-item"><span class="swatch worst"></span>사고 후 최대 부하 선로</span>
          <span class="legend-item"><span class="swatch voltage"></span>최저전압 Bus</span>
        </div>
      </div>

      <aside class="detail-panel">
        <h2 id="detail-title">-</h2>
        <div class="status-line" id="detail-status">-</div>
        <dl>
          <div class="detail-row">
            <dt>사고 전 부하율</dt>
            <dd id="detail-base-loading">-</dd>
          </div>
          <div class="detail-row">
            <dt>최대 부하 선로</dt>
            <dd id="detail-worst-branch">-</dd>
          </div>
          <div class="detail-row">
            <dt>과부하 선로</dt>
            <dd id="detail-overloads">-</dd>
          </div>
          <div class="detail-row">
            <dt>최저전압 Bus</dt>
            <dd id="detail-min-voltage">-</dd>
          </div>
          <div class="detail-row">
            <dt>전압 위반 Bus</dt>
            <dd id="detail-voltage-violations">-</dd>
          </div>
          <div class="detail-row">
            <dt>계통 분리</dt>
            <dd id="detail-islanding">-</dd>
          </div>
          <div class="detail-row">
            <dt>AC 손실</dt>
            <dd id="detail-loss">-</dd>
          </div>
        </dl>
        <p class="note">
          CSV에는 각 사고의 대표 결과가 저장되므로 지도에는 탈락 선로,
          최대 부하 선로와 최저전압 Bus를 표시합니다. 계통 분리 사고는
          단일 Slack AC-PF를 생략했기 때문에 사고 후 전압·부하율이 없습니다.
        </p>
      </aside>
    </section>
  </main>

  <script>
    "use strict";

    const DATA = __DATA__;
    const SVG_NS = "http://www.w3.org/2000/svg";
    const statusFilter = document.getElementById("status-filter");
    const scenarioSelect = document.getElementById("scenario-select");
    const prevButton = document.getElementById("prev-button");
    const nextButton = document.getElementById("next-button");
    const baseBranchesGroup = document.getElementById("base-branches");
    const baseBusesGroup = document.getElementById("base-buses");
    const highlightsGroup = document.getElementById("highlights");
    const labelsGroup = document.getElementById("labels");

    const busById = new Map(DATA.buses.map((bus) => [Number(bus.id), bus]));
    const branchById = new Map(
      DATA.branches.map((branch) => [Number(branch.id), branch])
    );

    const longitudeValues = DATA.buses.map((bus) => Number(bus.longitude));
    const latitudeValues = DATA.buses.map((bus) => Number(bus.latitude));
    const minLongitude = Math.min(...longitudeValues);
    const maxLongitude = Math.max(...longitudeValues);
    const minLatitude = Math.min(...latitudeValues);
    const maxLatitude = Math.max(...latitudeValues);
    const mapPaddingX = 46;
    const mapPaddingY = 40;
    const mapWidth = 780;
    const mapHeight = 900;

    function xPosition(longitude) {
      return mapPaddingX
        + ((Number(longitude) - minLongitude) / (maxLongitude - minLongitude))
        * (mapWidth - 2 * mapPaddingX);
    }

    function yPosition(latitude) {
      return mapPaddingY
        + ((maxLatitude - Number(latitude)) / (maxLatitude - minLatitude))
        * (mapHeight - 2 * mapPaddingY);
    }

    function createSvgElement(tagName, attributes = {}) {
      const element = document.createElementNS(SVG_NS, tagName);
      for (const [key, value] of Object.entries(attributes)) {
        element.setAttribute(key, String(value));
      }
      return element;
    }

    function lineForBranch(branch, className) {
      const fromBus = busById.get(Number(branch.from_bus));
      const toBus = busById.get(Number(branch.to_bus));
      if (!fromBus || !toBus) {
        return null;
      }

      const line = createSvgElement("line", {
        x1: xPosition(fromBus.longitude),
        y1: yPosition(fromBus.latitude),
        x2: xPosition(toBus.longitude),
        y2: yPosition(toBus.latitude),
        class: className,
      });
      const title = createSvgElement("title");
      title.textContent =
        "B" + branch.id + " " + fromBus.name + " - " + toBus.name;
      line.appendChild(title);
      return line;
    }

    function drawBaseNetwork() {
      for (const branch of DATA.branches) {
        if (!branch.in_service) {
          continue;
        }
        const line = lineForBranch(branch, "base-branch");
        if (line) {
          baseBranchesGroup.appendChild(line);
        }
      }

      for (const bus of DATA.buses) {
        const circle = createSvgElement("circle", {
          cx: xPosition(bus.longitude),
          cy: yPosition(bus.latitude),
          r: 2.5,
          class: "base-bus",
        });
        const title = createSvgElement("title");
        title.textContent =
          "Bus " + bus.id + " " + bus.name + " (" + bus.base_kv + " kV)";
        circle.appendChild(title);
        baseBusesGroup.appendChild(circle);
      }
    }

    function numberOrNull(value) {
      if (value === null || value === undefined || value === "") {
        return null;
      }
      const number = Number(value);
      return Number.isFinite(number) ? number : null;
    }

    function formatNumber(value, digits, suffix = "") {
      const number = numberOrNull(value);
      return number === null ? "-" : number.toFixed(digits) + suffix;
    }

    function statusLabel(status) {
      const labels = {
        OK: "한계 내",
        LIMIT_VIOLATION: "한계 위반",
        ISLANDING: "계통 분리",
        NON_CONVERGED: "비수렴",
        ERROR: "계산 오류",
      };
      return labels[status] || status || "-";
    }

    function statusClass(status) {
      if (status === "OK") {
        return "status-ok";
      }
      if (status === "LIMIT_VIOLATION") {
        return "status-warning";
      }
      return "status-danger";
    }

    function currentFilteredResults() {
      const selectedStatus = statusFilter.value;
      if (selectedStatus === "ALL") {
        return DATA.results;
      }
      return DATA.results.filter((row) => row.status === selectedStatus);
    }

    function rebuildScenarioOptions(preferredBranch = null) {
      const filtered = currentFilteredResults();
      scenarioSelect.replaceChildren();

      for (const row of filtered) {
        const option = document.createElement("option");
        option.value = String(row.outage_branch);
        option.textContent =
          "[" + statusLabel(row.status) + "] B" + row.outage_branch + " "
          + row.outage_from_name + " → " + row.outage_to_name;
        scenarioSelect.appendChild(option);
      }

      if (preferredBranch !== null) {
        const matchingOption = Array.from(scenarioSelect.options).find(
          (option) => Number(option.value) === Number(preferredBranch)
        );
        if (matchingOption) {
          scenarioSelect.value = matchingOption.value;
        }
      }

      if (scenarioSelect.options.length > 0) {
        renderSelectedScenario();
      } else {
        clearDashboard();
      }
    }

    function selectedResult() {
      const branchNumber = Number(scenarioSelect.value);
      return DATA.results.find(
        (row) => Number(row.outage_branch) === branchNumber
      );
    }

    function addMapLabel(bus, text, dx, dy) {
      const label = createSvgElement("text", {
        x: xPosition(bus.longitude) + dx,
        y: yPosition(bus.latitude) + dy,
        class: "map-label",
      });
      label.textContent = text;
      labelsGroup.appendChild(label);
    }

    function renderMap(row) {
      highlightsGroup.replaceChildren();
      labelsGroup.replaceChildren();

      const outageBranch = branchById.get(Number(row.outage_branch));
      if (outageBranch) {
        const outageLine = lineForBranch(outageBranch, "outage-line");
        if (outageLine) {
          highlightsGroup.appendChild(outageLine);
        }
        const outageFrom = busById.get(Number(outageBranch.from_bus));
        if (outageFrom) {
          addMapLabel(
            outageFrom,
            "탈락 B" + row.outage_branch + " "
              + row.outage_from_name + "–" + row.outage_to_name,
            8,
            -10
          );
        }
      }

      const worstBranchNumber = numberOrNull(row.worst_branch);
      const worstBranch = branchById.get(worstBranchNumber);
      if (worstBranch) {
        const worstLine = lineForBranch(worstBranch, "worst-line");
        if (worstLine) {
          highlightsGroup.appendChild(worstLine);
        }
        const worstFrom = busById.get(Number(worstBranch.from_bus));
        if (worstFrom) {
          addMapLabel(
            worstFrom,
            "최대 " + formatNumber(row.max_loading_pct, 2, "%"),
            8,
            18
          );
        }
      }

      const minVoltageBusId = numberOrNull(row.min_voltage_bus);
      const minVoltageBus = busById.get(minVoltageBusId);
      if (minVoltageBus) {
        const marker = createSvgElement("circle", {
          cx: xPosition(minVoltageBus.longitude),
          cy: yPosition(minVoltageBus.latitude),
          r: 8,
          class: "voltage-bus",
        });
        highlightsGroup.appendChild(marker);
        addMapLabel(
          minVoltageBus,
          "최저 " + formatNumber(row.min_voltage_pu, 5, " pu"),
          11,
          4
        );
      }
    }

    function setText(id, value) {
      document.getElementById(id).textContent = value;
    }

    function renderSelectedScenario() {
      const row = selectedResult();
      if (!row) {
        clearDashboard();
        return;
      }

      const statusText = statusLabel(row.status);
      const overloadCount = numberOrNull(row.overloaded_branch_count);
      const voltageViolationCount = numberOrNull(row.voltage_violation_count);
      const violationText =
        (overloadCount === null ? "-" : overloadCount + "선로")
        + " / "
        + (voltageViolationCount === null ? "-" : voltageViolationCount + "Bus");

      setText("metric-status", statusText);
      setText("metric-loading", formatNumber(row.max_loading_pct, 2, "%"));
      setText("metric-voltage", formatNumber(row.min_voltage_pu, 5, " pu"));
      setText("metric-violations", violationText);

      setText(
        "detail-title",
        "B" + row.outage_branch + " "
          + row.outage_from_name + " → " + row.outage_to_name
      );
      const detailStatus = document.getElementById("detail-status");
      detailStatus.textContent = statusText;
      detailStatus.className = "status-line " + statusClass(row.status);

      setText(
        "detail-base-loading",
        formatNumber(row.outage_base_loading_pct, 2, "%")
      );

      if (numberOrNull(row.worst_branch) !== null) {
        setText(
          "detail-worst-branch",
          "B" + row.worst_branch + " "
            + row.worst_from_name + " → " + row.worst_to_name + " / "
            + formatNumber(row.max_loading_pct, 2, "%")
        );
      } else {
        setText("detail-worst-branch", "-");
      }

      setText(
        "detail-overloads",
        overloadCount === null ? "-" : overloadCount + "개"
      );

      if (numberOrNull(row.min_voltage_bus) !== null) {
        setText(
          "detail-min-voltage",
          "Bus " + row.min_voltage_bus + " " + row.min_voltage_name + " / "
            + formatNumber(row.min_voltage_pu, 5, " pu")
        );
      } else {
        setText("detail-min-voltage", "-");
      }

      setText(
        "detail-voltage-violations",
        voltageViolationCount === null ? "-" : voltageViolationCount + "개"
      );
      setText(
        "detail-islanding",
        row.status === "ISLANDING"
          ? row.component_count + "개 계통 (" + row.component_sizes + ")"
          : "없음"
      );
      setText("detail-loss", formatNumber(row.ac_loss_mw, 2, " MW"));

      renderMap(row);
    }

    function clearDashboard() {
      for (const id of [
        "metric-status",
        "metric-loading",
        "metric-voltage",
        "metric-violations",
        "detail-title",
        "detail-status",
        "detail-base-loading",
        "detail-worst-branch",
        "detail-overloads",
        "detail-min-voltage",
        "detail-voltage-violations",
        "detail-islanding",
        "detail-loss",
      ]) {
        setText(id, "-");
      }
      highlightsGroup.replaceChildren();
      labelsGroup.replaceChildren();
    }

    function moveSelection(offset) {
      const optionCount = scenarioSelect.options.length;
      if (optionCount === 0) {
        return;
      }
      const nextIndex =
        (scenarioSelect.selectedIndex + offset + optionCount) % optionCount;
      scenarioSelect.selectedIndex = nextIndex;
      renderSelectedScenario();
    }

    statusFilter.addEventListener("change", () => {
      rebuildScenarioOptions();
    });
    scenarioSelect.addEventListener("change", renderSelectedScenario);
    prevButton.addEventListener("click", () => moveSelection(-1));
    nextButton.addEventListener("click", () => moveSelection(1));

    drawBaseNetwork();
    statusFilter.value = "LIMIT_VIOLATION";
    rebuildScenarioOptions(DATA.default_branch);
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KPG-193 N-1 결과를 오프라인 HTML로 변환",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RESULT_FILE,
        help="kpg_day02.py가 생성한 CSV 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_FILE,
        help="생성할 HTML 경로",
    )
    return parser.parse_args()


def dataframe_records(dataframe: pd.DataFrame) -> list[dict[str, object]]:
    """NaN과 NumPy 자료형을 JSON 호환 값으로 변환한다."""

    return json.loads(
        dataframe.to_json(
            orient="records",
            force_ascii=False,
        )
    )


def main() -> None:
    args = parse_args()
    input_file = args.input.resolve()
    output_file = args.output.resolve()

    if not input_file.exists():
        raise FileNotFoundError(
            f"결과 CSV를 찾지 못했습니다: {input_file}\n"
            "먼저 python .\\kpg_day02.py 를 실행하세요."
        )

    case_text = CASE_FILE.read_text(encoding="utf-8")
    bus_matrix = load_matpower_matrix(case_text, "bus")
    branch_matrix = load_matpower_matrix(case_text, "branch")
    metadata = pd.read_csv(BUS_METADATA_FILE, encoding="utf-8-sig")
    results = pd.read_csv(input_file, encoding="utf-8-sig")

    bus_kv = {
        int(row[0]): float(row[9])
        for row in bus_matrix
    }
    buses = [
        {
            "id": int(row.bus_id),
            "name": str(row.name_Korean),
            "latitude": round(float(row.Latitude), 7),
            "longitude": round(float(row.Longitude), 7),
            "base_kv": bus_kv.get(int(row.bus_id)),
        }
        for row in metadata.itertuples(index=False)
    ]
    branches = [
        {
            "id": index + 1,
            "from_bus": int(row[0]),
            "to_bus": int(row[1]),
            "rate_a": round(float(row[5]), 3),
            "in_service": bool(row[10] > 0),
        }
        for index, row in enumerate(branch_matrix)
    ]

    result_records = dataframe_records(results)
    violation_rows = [
        row
        for row in result_records
        if row.get("status") == "LIMIT_VIOLATION"
    ]
    default_branch = (
        max(
            violation_rows,
            key=lambda row: float(row.get("max_loading_pct") or -1),
        )["outage_branch"]
        if violation_rows
        else result_records[0]["outage_branch"]
    )

    payload = {
        "buses": buses,
        "branches": branches,
        "results": result_records,
        "default_branch": default_branch,
    }
    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("</", "<\\/")

    html = HTML_TEMPLATE.replace("__DATA__", payload_json)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")

    print("=" * 72)
    print("KPG-193 N-1 웹 대시보드 생성 완료")
    print("=" * 72)
    print(f"입력 CSV : {input_file}")
    print(f"사고 건수: {len(result_records)}")
    print(f"출력 HTML: {output_file}")
    print()
    print("PowerShell에서 열기:")
    print(f'Invoke-Item "{output_file}"')


if __name__ == "__main__":
    main()