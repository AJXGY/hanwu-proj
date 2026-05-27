const state = {
  config: null,
  environment: null,
  runs: [],
  activeRunId: null,
  pollTimer: null,
  envPollTimer: null,
};

const form = document.getElementById("run-form");
const prepareButton = document.getElementById("prepare-button");
const stopEnvButton = document.getElementById("stop-env-button");
const runButton = document.getElementById("run-button");
const refreshButton = document.getElementById("refresh-button");
const localGpuCountEl = document.getElementById("local-gpu-count");
const deploymentModeEl = document.getElementById("deployment-mode");
const derivedConfigNoteEl = document.getElementById("derived-config-note");
const runnerNoteEl = document.getElementById("runner-note");
const environmentNoteEl = document.getElementById("environment-note");
const runList = document.getElementById("run-list");
const statusBanner = document.getElementById("status-banner");
const titleEl = document.getElementById("results-title");
const metaEl = document.getElementById("results-meta");
const pipelineMetaEl = document.getElementById("pipeline-meta");
const pipelineCardsEl = document.getElementById("pipeline-cards");
const pipelineStepsEl = document.getElementById("pipeline-steps");
const estimateCardsEl = document.getElementById("estimate-cards");
const executionCardsEl = document.getElementById("execution-cards");
const measuredMetaEl = document.getElementById("measured-meta");
const measuredCardsEl = document.getElementById("measured-cards");
const estimateEl = document.getElementById("estimate-breakdowns");
const measuredEl = document.getElementById("measured-breakdowns");
const calibrationEl = document.getElementById("calibration-grid");
const profileEl = document.getElementById("profile-tables");
const rankTimingsEl = document.getElementById("rank-timings");
const commTablesEl = document.getElementById("comm-tables");
const operatorCompareEl = document.getElementById("operator-compare");
const operatorPhaseFilterEl = document.getElementById("operator-phase-filter");
const operatorRankFilterEl = document.getElementById("operator-rank-filter");
const operatorSortEl = document.getElementById("operator-sort");
const operatorMinErrorEl = document.getElementById("operator-min-error");
const topOpsEl = document.getElementById("top-ops");
const logsEl = document.getElementById("run-logs");
const graphMetaEl = document.getElementById("graph-meta");
const graphSelectEl = document.getElementById("graph-asset-select");
const graphLinksEl = document.getElementById("graph-links");
const graphViewerEl = document.getElementById("graph-viewer");
const remoteFieldNames = [
  "remote_host",
  "remote_ssh_port",
  "remote_physical_devices",
  "remote_python_bin",
  "remote_model_path",
  "remote_container_name",
];

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload.error || `请求失败: ${response.status}`);
  }
  return response.json();
}

function statusLabel(status) {
  return {
    queued: "排队中",
    running: "运行中",
    estimate_ready: "Estimate 已完成",
    measurement_ready: "Measured 已完成",
    rendering_graph: "正在生成图视图",
    pending: "待生成",
    disabled: "已关闭",
    completed: "已完成",
    failed: "失败",
  }[status] || status;
}

function environmentStatusLabel(status) {
  return {
    unprepared: "未准备",
    preparing: "准备中",
    ready: "已就绪",
    failed: "失败",
    stopped: "已停止",
  }[status] || status;
}

function phaseLabel(phase) {
  return {
    prefill: "Prefill",
    decode_step: "Decode Step",
    decode: "Decode",
    request: "Request",
  }[phase] || phase;
}

function setBanner(text, kind) {
  statusBanner.textContent = text;
  statusBanner.className = `status-banner ${kind}`;
}

function managedEnvironment() {
  return state.environment || state.config?.environment || null;
}

function scheduleEnvironmentPoll() {
  clearTimeout(state.envPollTimer);
  state.envPollTimer = setTimeout(
    () => fetchEnvironment().catch((error) => setBanner(error.message, "failed")),
    2500,
  );
}

function renderEnvironmentStatus() {
  const environment = managedEnvironment();
  const runnerField = namedField("runner");
  const imageField = namedField("image_name");
  const pythonField = namedField("python_bin");
  const configField = namedField("config_path");
  const locked = Boolean(state.config?.environment_locked);
  if (configField) {
    configField.value = state.config?.environment_config_path || environment?.config_path || "";
  }
  if (!environment) {
    environmentNoteEl.textContent = "正在读取环境状态。";
    runButton.disabled = true;
    prepareButton.disabled = true;
    stopEnvButton.disabled = true;
    return;
  }
  const details = environment.details || {};
  const runnerText = environment.runner || runnerField.value || "unknown";
  const preparedAt = environment.prepared_at
    ? new Date(environment.prepared_at * 1000).toLocaleString()
    : "-";
  const locationText = environment.container_name
    ? `container=${environment.container_name}`
    : (details.python_bin || "host python");
  const mounts = Array.isArray(details.project_mounts) && details.project_mounts.length
    ? ` 挂载 ${details.project_mounts.join(", ")}`
    : "";
  const remoteText = details.remote
    ? ` 远端 ${details.remote.host}:${details.remote.ssh_port} container=${details.remote.container_name}`
    : "";
  const errorText = environment.last_error ? ` 失败原因: ${environment.last_error}` : "";
  environmentNoteEl.textContent = `配置文件固定 runner=${runnerText}。当前环境 ${environmentStatusLabel(environment.status)}，${locationText}，prepared_at=${preparedAt}.${mounts}${remoteText}${errorText}`;
  if (locked) {
    runnerField.disabled = true;
    imageField.disabled = true;
    pythonField.disabled = true;
  }
  runButton.disabled = !environment.ready;
  prepareButton.disabled = environment.status === "preparing";
  stopEnvButton.disabled = environment.status === "preparing" || environment.status === "unprepared";
  if (environment.status === "preparing") {
    scheduleEnvironmentPoll();
  } else {
    clearTimeout(state.envPollTimer);
  }
}

async function fetchEnvironment() {
  state.environment = await fetchJson("/api/environment");
  renderEnvironmentStatus();
}

function fillForm(values) {
  for (const [key, value] of Object.entries(values)) {
    const field = form.elements.namedItem(key);
    if (field) {
      if (field.type === "checkbox") {
        field.checked = Boolean(value);
      } else {
        field.value = value;
      }
    }
  }
}

function namedField(name) {
  return form.elements.namedItem(name);
}

function parsePhysicalDevices(value) {
  const devices = [];
  for (const part of String(value || "").split(",")) {
    const device = Number(part.trim());
    if (Number.isInteger(device) && !devices.includes(device)) {
      devices.push(device);
    }
  }
  return devices;
}

function localGpuInventory() {
  return Array.isArray(state.config?.local_gpus) ? state.config.local_gpus : [];
}

function acceleratorKind() {
  return state.config?.accelerator_kind || "cuda";
}

function acceleratorLabel() {
  return state.config?.accelerator_label || (acceleratorKind() === "mlu" ? "MLU" : "GPU");
}

function devicePrefix() {
  return acceleratorKind() === "mlu" ? "mlu" : "cuda";
}

function singleHostRequested() {
  if (deploymentModeEl?.value === "multi_host") return false;
  return Number(namedField("nnodes").value || 1) <= 1;
}

function inferredLocalGpuCount() {
  if (!singleHostRequested()) {
    return Math.max(1, Number(namedField("nproc_per_node").value || 1));
  }
  if (namedField("parallel_mode").value === "single") {
    return 1;
  }
  return Math.max(
    parsePhysicalDevices(namedField("physical_devices").value).length,
    Number(namedField("tp_size").value || 0),
    Number(namedField("world_size").value || 0),
    1,
  );
}

function preferredDeviceOrder() {
  const current = parsePhysicalDevices(namedField("physical_devices").value);
  const inventory = localGpuInventory()
    .map((gpu) => Number(gpu.index))
    .filter((gpu) => Number.isInteger(gpu));
  const ordered = [...current];
  for (const gpu of inventory) {
    if (!ordered.includes(gpu)) {
      ordered.push(gpu);
    }
  }
  return ordered;
}

function deviceListForCount(count) {
  const devices = preferredDeviceOrder().slice(0, count);
  let fallback = 0;
  while (devices.length < count) {
    if (!devices.includes(fallback)) {
      devices.push(fallback);
    }
    fallback += 1;
  }
  return devices;
}

function localGpuChoices() {
  const inventoryCount = localGpuInventory().length;
  const currentCount = parsePhysicalDevices(namedField("physical_devices").value).length;
  const inferredCount = inferredLocalGpuCount();
  const maxCount = Math.max(inventoryCount, currentCount, inferredCount, 1);
  return Array.from({ length: maxCount }, (_, index) => index + 1);
}

function renderLocalGpuCountOptions() {
  const choices = localGpuChoices();
  const targetCount = Math.min(
    Math.max(1, Number(localGpuCountEl.value || inferredLocalGpuCount())),
    choices[choices.length - 1],
  );
  localGpuCountEl.innerHTML = choices.map((count) => {
    const devices = deviceListForCount(count);
    const prefix = devicePrefix();
    const suffix = devices.length ? ` · ${prefix}${devices.join(`, ${prefix}`)}` : "";
    return `<option value="${count}">${count} 卡${suffix}</option>`;
  }).join("");
  localGpuCountEl.value = String(targetCount);
}

function refreshPlacementControls({ updatePhysicalDevices = false } = {}) {
  renderLocalGpuCountOptions();
  const multiHost = !singleHostRequested();
  for (const fieldName of remoteFieldNames) {
    const field = namedField(fieldName);
    if (!field) continue;
    field.disabled = !multiHost;
  }
  if (!singleHostRequested()) {
    localGpuCountEl.disabled = true;
    if (!namedField("remote_physical_devices").value.trim()) {
      namedField("remote_physical_devices").value = namedField("physical_devices").value;
    }
    if (!namedField("remote_ssh_port").value) {
      namedField("remote_ssh_port").value = "22";
    }
    derivedConfigNoteEl.textContent = "已切换到跨机自动拉起模式。请维护 nnodes、nproc_per_node、tp_size、master_addr，以及 remote_host/remote_physical_devices。启动时会自动 SSH 到远端准备环境并拉起 rank1。";
    return;
  }
  localGpuCountEl.disabled = false;
  const count = Math.max(1, Number(localGpuCountEl.value || 1));
  const devices = deviceListForCount(count);
  namedField("parallel_mode").value = count > 1 ? "tp" : "single";
  namedField("world_size").value = String(count);
  namedField("tp_size").value = String(count);
  namedField("nnodes").value = "1";
  namedField("nproc_per_node").value = String(count);
  namedField("node_rank").value = "0";
  namedField("master_addr").value = "127.0.0.1";
  if (updatePhysicalDevices || !parsePhysicalDevices(namedField("physical_devices").value).length) {
    namedField("physical_devices").value = devices.join(",");
  }
  const gpuSummary = localGpuInventory().length
    ? `检测到本机 ${localGpuInventory().length} 张 ${acceleratorLabel()}。`
    : `未从宿主机探测到 ${acceleratorLabel()} 清单。`;
  const modeLabel = count > 1 ? `单机 ${count} 卡 TP` : "单卡";
  derivedConfigNoteEl.textContent = `${gpuSummary} 当前按 ${modeLabel} 运行，自动使用 physical_devices=${namedField("physical_devices").value}，并匹配 world_size=${count}、tp_size=${count}、nproc_per_node=${count}。`;
}

function refreshDeploymentControls() {
  const nnodes = Number(namedField("nnodes").value || 1);
  if (deploymentModeEl) {
    deploymentModeEl.value = nnodes > 1 ? "multi_host" : "single_host";
  }
  refreshPlacementControls({ updatePhysicalDevices: singleHostRequested() });
}

function refreshRunnerControls() {
  const environment = managedEnvironment();
  const locked = Boolean(state.config?.environment_locked);
  const configuredRunner = environment?.runner || namedField("runner").value;
  if (locked && configuredRunner) {
    namedField("runner").value = configuredRunner;
  }
  const isLocal = configuredRunner === "local_python";
  const imageField = namedField("image_name");
  const pythonBinField = namedField("python_bin");
  namedField("runner").disabled = locked;
  imageField.disabled = locked || isLocal;
  pythonBinField.disabled = locked || !isLocal;
  if (isLocal) {
    runnerNoteEl.textContent = `环境由配置文件固定为本机 Host Python：${pythonBinField.value}。Dashboard 启动后会先校验该解释器可用，再允许直接开始估测/实测。`;
  } else {
    runnerNoteEl.textContent = `环境由配置文件固定为 Docker 镜像 ${imageField.value}，Dashboard 会在后台常驻维护该容器；开始任务时直接在已准备好的环境里执行，不再为每次运行重复拉起容器。`;
  }
}

function formPayload() {
  refreshPlacementControls({ updatePhysicalDevices: singleHostRequested() });
  refreshRunnerControls();
  const payload = {};
  for (const element of form.elements) {
    if (!element.name) continue;
    payload[element.name] = element.type === "checkbox" ? element.checked : element.value;
  }
  payload.max_new_tokens = Number(payload.max_new_tokens);
  payload.world_size = Number(payload.world_size);
  payload.tp_size = Number(payload.tp_size);
  payload.nnodes = Number(payload.nnodes);
  payload.nproc_per_node = Number(payload.nproc_per_node);
  payload.node_rank = Number(payload.node_rank);
  payload.master_port = Number(payload.master_port);
  payload.remote_ssh_port = Number(payload.remote_ssh_port || 22);
  payload.dist_timeout_minutes = Number(payload.dist_timeout_minutes);
  payload.warmup = Number(payload.warmup);
  payload.benchmark_repeat = Number(payload.benchmark_repeat);
  payload.profile_repeat = Number(payload.profile_repeat);
  return payload;
}

function executionInfo(report, request = {}) {
  if (report?.execution) return report.execution;
  const device = request.device || `${devicePrefix()}:0`;
  const physical = request.physical_devices
    ? String(request.physical_devices).split(",").map((item) => Number(item.trim())).filter((item) => !Number.isNaN(item))
    : device.includes(":") ? [Number(device.split(":")[1])] : [0];
  return {
    parallel_mode: request.parallel_mode || "single",
    physical_devices: physical,
    visible_devices: physical.join(","),
    world_size: request.world_size || 1,
    tp_size: request.tp_size || 1,
    nnodes: request.nnodes || 1,
    nproc_per_node: request.nproc_per_node || request.world_size || 1,
    node_rank: request.node_rank || 0,
    master_addr: request.master_addr || "127.0.0.1",
    master_port: request.master_port || 29500,
    interconnect: request.interconnect || report?.execution?.interconnect || "local",
    topology: report?.execution?.topology || "local",
    local_topology: report?.execution?.local_topology || report?.execution?.topology || "local",
    placements: report?.execution?.placements || physical.map((device, index) => ({
      rank: index,
      host: index === 0 ? "local" : `rank-${index}`,
      node_rank: 0,
      local_rank: index,
      physical_device: device,
    })),
  };
}

function placementSummary(execution) {
  const placements = execution.placements || [];
  if (!placements.length) return execution.physical_devices.join(",") || "-";
  return placements.map((item) => `r${item.rank}@${item.host}:${devicePrefix()}${item.physical_device}`).join(" · ");
}

function operatorRows(report) {
  return [
    ...(report?.operator_compare?.prefill || []),
    ...(report?.operator_compare?.decode_step || []),
  ];
}

function reportHasOperatorCompare(report) {
  return operatorRows(report).length > 0 || report?.operator_compare?.summary?.status === "unavailable";
}

function reportHasRankMeasurements(report) {
  return Boolean(report?.rank_measurements?.prefill?.length || report?.rank_measurements?.decode_step?.length || report?.rank_measurements?.request?.length);
}

function reportHasComm(report) {
  return Boolean(report?.comm?.prefill || report?.comm?.decode_step);
}

function rankGapText(rows) {
  if (!rows?.length) return "-";
  const means = rows.map((row) => Number(row.mean_ms || 0));
  return formatMs(Math.max(...means) - Math.min(...means));
}

function graphAssetLabel(name) {
  return name.replace(/_/g, " ").replace(/\.svg$|\.html$|\.json$|\.txt$/g, "");
}

function preferredGraphAsset(graph) {
  if (!graph?.assets?.length) return null;
  return graph.assets.find((item) => item.name.endsWith("_estimate_graph.svg"))
    || graph.assets.find((item) => item.name.endsWith("index.html"))
    || graph.assets[0];
}

function formatMs(value) {
  return typeof value === "number" ? `${value.toFixed(4)} ms` : "-";
}

function formatPct(value) {
  return typeof value === "number" ? `${value.toFixed(2)}%` : "-";
}

function formatSeconds(value) {
  return typeof value === "number" ? `${value.toFixed(1)}s` : "-";
}

function durationText(run) {
  if (!run.started_at) return "排队中";
  if (typeof run.timings?.total_wall_time_s === "number") {
    return `${run.timings.total_wall_time_s.toFixed(1)}s`;
  }
  if (
    typeof run.timings?.predictor_total_wall_time_s === "number"
    && ["measurement_ready", "rendering_graph", "completed", "failed"].includes(run.status)
  ) {
    return `${run.timings.predictor_total_wall_time_s.toFixed(1)}s`;
  }
  const end = run.finished_at || Date.now() / 1000;
  return `${(end - run.started_at).toFixed(1)}s`;
}

function isActiveStatus(status) {
  return ["queued", "running", "estimate_ready", "measurement_ready", "rendering_graph"].includes(status);
}

function renderRunList() {
  if (!state.runs.length) {
    runList.innerHTML = '<p class="muted">暂时还没有运行记录。</p>';
    return;
  }
  runList.innerHTML = state.runs.map((run) => {
    const report = run.report;
    const promptTokens = report?.model?.prompt_tokens ?? run.request?.prompt?.split(/\s+/).length ?? "?";
    const execution = executionInfo(report, run.request);
    const activeClass = run.run_id === state.activeRunId ? "active" : "";
    return `
      <article class="run-card ${activeClass}" data-run-id="${run.run_id}">
        <h4>${run.request.model_path.split("/").slice(-1)[0]}</h4>
        <div class="run-meta">状态: ${statusLabel(run.status)} · tokens: ${promptTokens}/${run.request.max_new_tokens}</div>
        <div class="run-meta">Runner: ${run.request.runner} · ${execution.parallel_mode} · ${placementSummary(execution)}</div>
        <div class="run-meta">已耗时: ${durationText(run)}</div>
      </article>
    `;
  }).join("");
  for (const card of runList.querySelectorAll(".run-card")) {
    card.addEventListener("click", () => selectRun(card.dataset.runId));
  }
}

function barRows(entries, alt = false) {
  const max = Math.max(...entries.map((entry) => entry.value), 1e-9);
  return entries.map((entry) => `
    <div class="bar-row">
      <div class="bar-label"><span>${entry.label}</span><strong>${entry.value.toFixed(4)} ms</strong></div>
      <div class="bar-track"><div class="bar-fill ${alt ? "alt" : ""}" style="width:${(entry.value / max) * 100}%"></div></div>
    </div>
  `).join("");
}

function renderEstimateBreakdowns(report) {
  const phases = ["prefill", "decode_step"];
  estimateEl.innerHTML = phases.map((phaseName) => {
    const phase = report.estimate[phaseName];
    const opEntries = Object.entries(phase.op_family_breakdown_ms).map(([label, value]) => ({ label, value }));
    const regionEntries = phase.top_regions.map((entry) => ({ label: entry.region, value: entry.estimated_time_ms }));
    return `
      <section class="phase-block">
        <h4>${phaseLabel(phaseName)}</h4>
        <div class="inline-pills">
          <span class="pill">阶段总计 ${formatMs(phase.end_to_end_time_ms)}</span>
          <span class="pill">graph compute ${formatMs(phase.graph_compute_time_ms)}</span>
          <span class="pill">graph comm ${formatMs(phase.graph_comm_time_ms)}</span>
          <span class="pill">overhead ${formatMs(phase.runtime_overhead_time_ms)}</span>
          <span class="pill">节点数 ${phase.node_count}</span>
        </div>
        <div class="bar-group">
          <strong>按 op family</strong>
          ${barRows(opEntries)}
        </div>
        <div class="bar-group">
          <strong>按 region</strong>
          ${barRows(regionEntries, true)}
        </div>
      </section>
    `;
  }).join("");
}

function renderMeasured(report) {
  const rows = [
    ["prefill", report.measured.prefill, report.comparison.prefill_relative_error_pct],
    ["decode_step", report.measured.decode_step, report.comparison.decode_step_relative_error_pct],
    ["request", report.measured.request, report.comparison.request_relative_error_pct],
  ];
  measuredEl.innerHTML = rows.map(([label, data, error]) => `
    <section class="phase-block">
      <h4>${phaseLabel(label)}</h4>
      <div class="inline-pills">
        <span class="pill">mean ${formatMs(data.mean_ms)}</span>
        <span class="pill">median ${formatMs(data.median_ms)}</span>
        <span class="pill">min ${formatMs(data.min_ms)}</span>
        <span class="pill">max ${formatMs(data.max_ms)}</span>
        <span class="pill">误差 ${formatPct(error)}</span>
      </div>
      ${barRows(data.samples_ms.map((value, index) => ({ label: `样本 ${index + 1}`, value })))}
    </section>
  `).join("");
}

function renderCalibration(report) {
  const entries = Object.entries(report.calibration);
  calibrationEl.innerHTML = entries.map(([key, value]) => `
    <div class="kv-row"><span>${key}</span><strong>${value}</strong></div>
  `).join("");
}

function renderModuleProfiles(report) {
  const phases = [
    ["prefill", report?.module_profile?.prefill || []],
    ["decode_step", report?.module_profile?.decode_step || []],
  ];
  return phases.map(([name, rows]) => {
    if (!rows.length) {
      return `
        <section class="phase-block">
          <h4>${phaseLabel(name)} Module Profiles</h4>
          <div class="muted">该 phase 暂无 module profile。</div>
        </section>
      `;
    }
    return `
      <section class="phase-block">
        <h4>${phaseLabel(name)} Module Profiles</h4>
        <table>
          <thead><tr><th>Scope</th><th>Kind</th><th>Mean</th><th>Median</th><th>Samples</th></tr></thead>
          <tbody>
            ${rows.map((row) => `<tr><td>${row.module_scope}</td><td>${row.module_kind}</td><td>${formatMs(row.mean_ms)}</td><td>${formatMs(row.median_ms)}</td><td>${row.samples_ms.length}</td></tr>`).join("")}
          </tbody>
        </table>
      </section>
    `;
  }).join("");
}

function renderProfiles(report) {
  const profileStatus = report?.profile?.status || "available";
  const profileReason = report?.profile?.reason;
  const phases = [
    ["prefill", report.profile.prefill_top_cuda_ops],
    ["decode", report.profile.decode_top_cuda_ops],
  ];
  const operatorSections = phases.map(([name, rows]) => {
    if (!rows.length) {
      return `
        <section class="phase-block">
          <h4>${phaseLabel(name)} Operator Profiler</h4>
          <div class="muted">${profileStatus === "unavailable" ? (profileReason || "当前没有可用的 operator profiler 数据。") : "该 phase 暂无 operator profiler 数据。"}</div>
        </section>
      `;
    }
    return `
      <section class="phase-block">
        <h4>${phaseLabel(name)} Operator Profiler</h4>
        <table>
          <thead><tr><th>Op</th><th>CUDA ms</th><th>调用次数</th></tr></thead>
          <tbody>
            ${rows.map((row) => `<tr><td>${row.op}</td><td>${row.self_cuda_time_ms.toFixed(4)}</td><td>${row.calls}</td></tr>`).join("")}
          </tbody>
        </table>
      </section>
    `;
  }).join("");
  profileEl.innerHTML = `${operatorSections}${renderModuleProfiles(report)}`;
}

function renderTopOps(report) {
  const phases = ["prefill", "decode_step"];
  topOpsEl.innerHTML = phases.map((phaseName) => {
    const phase = report.estimate[phaseName];
    return `
      <section class="phase-block">
        <h4>${phaseLabel(phaseName)}</h4>
        <table>
          <thead><tr><th>节点 Node</th><th>区域 Region</th><th>估计时间 Estimate</th></tr></thead>
          <tbody>
            ${phase.top_ops.map((item) => `<tr><td>${item.node_name}</td><td>${item.region}</td><td>${item.estimated_time_ms.toFixed(4)}</td></tr>`).join("")}
          </tbody>
        </table>
      </section>
    `;
  }).join("");
}

function renderPipeline(run) {
  const timings = run.timings || {};
  const estimation = timings.estimation_wall_time_s;
  const validation = timings.measurement_wall_time_s;
  const predictor = timings.predictor_total_wall_time_s || timings.predictor_wall_time_s;
  const graph = timings.graph_export_wall_time_s;
  const total = timings.total_wall_time_s;
  const estimateMode = run.request?.estimate_mode || "online";
  const graphState = run.request.generate_graph_viz ? (run.graph?.status || (run.status === "completed" ? "pending" : run.status)) : "disabled";
  pipelineMetaEl.textContent = "这里把 estimate 阶段拆开显示，便于区分 calibration、模型加载、图提取和 analytical estimates 的实际耗时。";
  const cards = [
    ["Estimate 就绪", formatSeconds(estimation), run.status === "estimate_ready" ? "estimate 已可查看，validation 仍在继续" : "包含 calibration、graph、module profiles 与 analytical estimate"],
    ["Analytical Estimates", formatSeconds(timings.analytical_estimate_wall_time_s), "节点估算、通信预测、phase 汇总与 request estimate"],
    ["Validation 阶段", formatSeconds(validation), reportHasMeasured(run.report) ? "prefill/decode/request 的 measured 对比与误差统计已完成" : "正在采集 measured timings 与 profiler 数据"],
    ["Predictor 总耗时", formatSeconds(predictor), "从开始到最终报告的 predictor wall time"],
    ["Graph Export", formatSeconds(graph), `graph 状态 ${statusLabel(graphState)}`],
    ["分析总耗时", formatSeconds(total), "本次 dashboard 运行消耗的总 wall time"],
    ["请求 tokens", `${run.report?.model?.prompt_tokens ?? "?"}/${run.report?.model?.max_new_tokens ?? run.request.max_new_tokens}`, "prefill tokens / max new tokens"],
    ["Calibration", formatSeconds(timings.calibration_wall_time_s), "硬件标定 microbench：GEMM、attention、bandwidth、launch overhead"],
    ["Load Model", formatSeconds(timings.model_load_wall_time_s), "checkpoint 加载，以及模型搬运到 device"],
    ["Extract Graph", formatSeconds(timings.graph_extract_wall_time_s), "准备输入并导出 prefill/decode inference graphs"],
    ["Runtime Inputs", formatSeconds(timings.runtime_inputs_wall_time_s), "真实执行一次 prefill，生成 next token 与 decode past_key_values"],
    ["Torch Export", formatSeconds(timings.torch_export_wall_time_s), "将 prefill/decode 路径导出为 torch.export graph"],
  ];
  if (typeof timings.table_lookup_wall_time_s === "number" && timings.table_lookup_wall_time_s > 0) {
    cards.push(["Table Lookup", formatSeconds(timings.table_lookup_wall_time_s), `estimate_mode=${estimateMode} 时从 table DB 读取 module profiles / phase adjustments`]);
  }
  if (typeof timings.graph_cache_load_wall_time_s === "number" && timings.graph_cache_load_wall_time_s > 0) {
    cards.push(["Graph Cache Load", formatSeconds(timings.graph_cache_load_wall_time_s), `graph cache ${run.report?.graph_cache?.status || "unknown"}，直接读取已记录的 graph-derived estimates 与节点统计`]);
  }
  if (typeof timings.graph_cache_write_wall_time_s === "number" && timings.graph_cache_write_wall_time_s > 0) {
    cards.push(["Graph Cache Write", formatSeconds(timings.graph_cache_write_wall_time_s), "首次 miss 后将 graph-derived estimates 与节点统计持久化，供后续复用"]);
  }
  if (typeof timings.module_profile_wall_time_s === "number" && timings.module_profile_wall_time_s > 0) {
    cards.push(["Module Profiles", formatSeconds(timings.module_profile_wall_time_s), `estimate_mode=${estimateMode} 时的 online module profiling 与可选 writeback`]);
  }
  if (typeof timings.runtime_prepare_wall_time_s === "number" && timings.runtime_prepare_wall_time_s > 0) {
    cards.push(["Runtime Prepare", formatSeconds(timings.runtime_prepare_wall_time_s), "为后续 measured 阶段重新加载 TP runtime 并准备 cache / decode 输入"]);
  }
  pipelineCardsEl.innerHTML = cards.map(([label, value, note]) => `
    <article class="pipeline-card">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${value}</div>
      <div>${note}</div>
    </article>
  `).join("");
  pipelineStepsEl.innerHTML = [
    ["1. 标定硬件", `在 ${run.report.calibration.device_name} 上测量 GEMM、attention、memory bandwidth 与 launch overhead。`],
    ["2. 提取真实推理图", `导出 prefill 与 decode 的 graph，并估算 ${run.report.graph.prefill_call_function_nodes} + ${run.report.graph.decode_call_function_nodes} 个 call_function 节点。`],
    ["3. 用 module profiles 修正", `将被覆盖的节点组替换为 profile_repeat=${run.request.profile_repeat} 次采样得到的 self_attn/mlp module timings。`],
    ["4. 与实测对照", `先给出 request estimate，再用 measured 的 prefill/decode/request timings 校验误差。`],
  ].map(([title, body]) => `
    <article class="step-card">
      <h4>${title}</h4>
      <p>${body}</p>
    </article>
  `).join("");
}

function reportHasEstimate(report) {
  return Boolean(report?.estimate?.prefill && report?.estimate?.decode_step);
}

function reportHasMeasured(report) {
  return Boolean(report?.measured?.prefill && report?.measured?.decode_step && report?.measured?.request);
}

function reportHasProfiles(report) {
  return Boolean(
    report?.profile?.prefill_top_cuda_ops
    || report?.profile?.decode_top_cuda_ops
    || report?.profile?.status
  );
}

function renderGraphViewer(run) {
  graphLinksEl.innerHTML = "";
  graphSelectEl.innerHTML = "";
  if (!run?.graph) {
    graphMetaEl.textContent = "每次运行可按需开启 graph export，完成后会显示在这里。";
    graphViewerEl.innerHTML = '<div class="muted">这次运行暂时还没有 graph 数据。</div>';
    return;
  }
  const graph = run.graph;
  if (!graph.assets?.length) {
    graphMetaEl.textContent = graph.error || `graph 状态: ${statusLabel(graph.status)}`;
    graphViewerEl.innerHTML = `<div class="muted">${graph.error || "没有生成 graph 产物。"}</div>`;
    return;
  }
  const promptTokens = graph.summary?.prompt_tokens ?? "?";
  graphMetaEl.textContent = `graph 状态: ${statusLabel(graph.status)} · prompt tokens: ${promptTokens} · 输出目录: ${graph.output_dir}`;
  graphSelectEl.innerHTML = graph.assets.map((asset) => `<option value="${asset.path}">${graphAssetLabel(asset.name)}</option>`).join("");
  graphLinksEl.className = "asset-links";
  graphLinksEl.innerHTML = graph.assets.map((asset) => `<a href="${asset.path}" target="_blank" rel="noreferrer">${asset.name}</a>`).join("");
  const initialAsset = preferredGraphAsset(graph);
  graphSelectEl.value = initialAsset.path;
  updateGraphPreview(initialAsset.path);
}

function updateGraphPreview(path) {
  if (!path) {
    graphViewerEl.innerHTML = '<div class="muted">尚未选择 graph 产物。</div>';
    return;
  }
  if (path.endsWith(".svg")) {
    graphViewerEl.innerHTML = `<img src="${path}" alt="Graph visualization" />`;
    return;
  }
  if (path.endsWith(".html")) {
    graphViewerEl.innerHTML = `<iframe src="${path}" title="Graph index"></iframe>`;
    return;
  }
  graphViewerEl.innerHTML = `<iframe src="${path}" title="Graph artifact"></iframe>`;
}

function renderEstimateOverview(report) {
  const cards = [
    ["Prefill", report.estimate.prefill.end_to_end_time_ms, report.estimate.prefill.node_count],
    ["Decode Step", report.estimate.decode_step.end_to_end_time_ms, report.estimate.decode_step.node_count],
    ["Request", report.estimate.request_end_to_end_time_ms, report.graph.prefill_call_function_nodes + report.graph.decode_call_function_nodes],
  ];
  estimateCardsEl.innerHTML = cards.map(([label, estimate, nodeCount]) => `
    <article class="metric-card">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${estimate.toFixed(2)} ms</div>
      <div>estimate 在执行前完成</div>
      <div>分析节点数 ${nodeCount}</div>
    </article>
  `).join("");
}

function renderMeasuredOverview(report) {
  measuredMetaEl.textContent = `Measured runtime 在 estimate 完成之后采集。这些数字用于验证 estimate 的效果，而不是 estimator 的输入。`;
  const cards = [
    ["Prefill Mean", report.measured.prefill.mean_ms, report.estimate.prefill.end_to_end_time_ms, report.comparison.prefill_relative_error_pct],
    ["Decode Mean", report.measured.decode_step.mean_ms, report.estimate.decode_step.end_to_end_time_ms, report.comparison.decode_step_relative_error_pct],
    ["Request Mean", report.measured.request.mean_ms, report.estimate.request_end_to_end_time_ms, report.comparison.request_relative_error_pct],
  ];
  measuredCardsEl.innerHTML = cards.map(([label, measured, estimate, error]) => `
    <article class="pipeline-card">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${measured.toFixed(2)} ms</div>
      <div>estimate ${estimate.toFixed(2)} ms</div>
      <div>误差 ${error.toFixed(2)}%</div>
    </article>
  `).join("");
}

function renderExecutionOverview(report, request) {
  const execution = executionInfo(report, request);
  const prefillRanks = report?.rank_measurements?.prefill || [];
  const commMs = (report?.comm?.prefill?.total_measured_ms || 0) + (report?.comm?.decode_step?.total_measured_ms || 0);
  const predictedCommMs = (report?.comm?.prefill?.predicted_total_ms || 0) + (report?.comm?.decode_step?.predicted_total_ms || 0);
  const cards = [
    ["Parallel Mode", execution.parallel_mode, `world_size ${execution.world_size} · tp_size ${execution.tp_size}`],
    ["Placement", placementSummary(execution), `nnodes ${execution.nnodes} · nproc_per_node ${execution.nproc_per_node}`],
    ["Topology", execution.topology || "unknown", `local ${execution.local_topology || "unknown"} · link ${execution.interconnect || "local"}`],
    ["Rank Gap", rankGapText(prefillRanks), "prefill mean 的 max/min gap"],
    ["Comm Profiled", formatMs(commMs), "prefill + decode_step 的 profiler collective critical path"],
    ["Comm Predicted", formatMs(predictedCommMs), "graph_comm_time_ms 汇总"],
  ];
  executionCardsEl.innerHTML = cards.map(([label, value, note]) => `
    <article class="metric-card">
      <div class="metric-label">${label}</div>
      <div class="metric-value">${value}</div>
      <div>${note}</div>
    </article>
  `).join("");
}

function renderRankTimings(report) {
  if (!reportHasRankMeasurements(report)) {
    rankTimingsEl.innerHTML = '<div class="muted">旧报表没有 per-rank timings；单卡报表会保持当前布局。</div>';
    return;
  }
  const phases = ["prefill", "decode_step", "request"];
  rankTimingsEl.innerHTML = phases.map((phaseName) => {
    const rows = report.rank_measurements?.[phaseName] || [];
    if (!rows.length) return "";
    return `
      <section class="phase-block">
        <h4>${phaseLabel(phaseName)}</h4>
        <div class="inline-pills">
          <span class="pill">rank gap ${rankGapText(rows)}</span>
          <span class="pill">rank count ${rows.length}</span>
        </div>
        <div class="table-wrap">
          <table>
            <thead><tr><th>Rank</th><th>Host</th><th>Device</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th></tr></thead>
            <tbody>
              ${rows.map((row) => `<tr><td>${row.rank}</td><td>${row.host || "-"}</td><td>${devicePrefix()}${row.device}</td><td>${formatMs(row.mean_ms)}</td><td>${formatMs(row.median_ms)}</td><td>${formatMs(row.min_ms)}</td><td>${formatMs(row.max_ms)}</td></tr>`).join("")}
            </tbody>
          </table>
        </div>
      </section>
    `;
  }).join("") || '<div class="muted">暂无 rank timing 数据。</div>';
}

function renderComm(report) {
  if (!reportHasComm(report)) {
    commTablesEl.innerHTML = '<div class="muted">旧报表没有 communication 统计。</div>';
    return;
  }
  const phases = ["prefill", "decode_step"];
  commTablesEl.innerHTML = phases.map((phaseName) => {
    const comm = report.comm?.[phaseName];
    if (!comm) return "";
    const rows = comm.collectives || [];
    const predictedRows = comm.predicted_collectives || [];
    return `
      <section class="phase-block">
        <h4>${phaseLabel(phaseName)}</h4>
        <div class="inline-pills">
          <span class="pill">collectives ${rows.length}</span>
          <span class="pill">profiled ${formatMs(comm.total_measured_ms)}</span>
          <span class="pill">predicted ${formatMs(comm.predicted_total_ms || 0)}</span>
        </div>
        ${rows.length ? `<div class="table-wrap"><table>
          <thead><tr><th>Collective</th><th>Count</th><th>Profiled ms</th><th>Per-rank</th></tr></thead>
          <tbody>
            ${rows.map((row) => `<tr><td>${row.collective}</td><td>${row.count}</td><td>${formatMs(row.total_measured_ms)}</td><td>${(row.per_rank || []).map((item) => `r${item.rank}@${item.host || "-"}:${devicePrefix()}${item.device}:${item.count}/${item.total_measured_ms.toFixed(4)}ms`).join(" · ")}</td></tr>`).join("")}
          </tbody>
        </table></div>` : '<div class="muted">该 phase 未观测到 collective。</div>'}
        ${predictedRows.length ? `<div class="table-wrap"><table>
          <thead><tr><th>Predicted</th><th>Scope</th><th>Count</th><th>Bytes</th><th>Pred ms</th></tr></thead>
          <tbody>
            ${predictedRows.map((row) => `<tr><td>${row.collective}</td><td>${row.scope}</td><td>${row.count}</td><td>${Math.round(row.bytes)}</td><td>${formatMs(row.predicted_ms)}</td></tr>`).join("")}
          </tbody>
        </table></div>` : '<div class="muted">该 phase 暂无预测通信行。</div>'}
      </section>
    `;
  }).join("") || '<div class="muted">暂无通信数据。</div>';
}

function updateOperatorRankFilter(report) {
  const rows = operatorRows(report);
  const ranks = [...new Set(rows.map((row) => String(row.rank)).filter((value) => value !== "undefined"))].sort((a, b) => Number(a) - Number(b));
  const current = operatorRankFilterEl.value || "all";
  operatorRankFilterEl.innerHTML = [`<option value="all">all</option>`, ...ranks.map((rank) => `<option value="${rank}">${rank}</option>`)].join("");
  operatorRankFilterEl.value = ranks.includes(current) ? current : "all";
}

function renderOperatorCompare(report) {
  const summary = report?.operator_compare?.summary || {};
  if (summary.status === "unavailable") {
    operatorCompareEl.innerHTML = `<div class="muted">${summary.reason || "当前运行未生成算子级 compare。"}</div>`;
    updateOperatorRankFilter({ operator_compare: { prefill: [], decode_step: [] } });
    return;
  }
  if (!reportHasOperatorCompare(report)) {
    operatorCompareEl.innerHTML = '<div class="muted">旧报表没有算子级 compare；会继续显示阶段级 compare。</div>';
    updateOperatorRankFilter({ operator_compare: { prefill: [], decode_step: [] } });
    return;
  }
  updateOperatorRankFilter(report);
  const phaseFilter = operatorPhaseFilterEl.value || "all";
  const rankFilter = operatorRankFilterEl.value || "all";
  const sortKey = operatorSortEl.value || "abs_err_ms";
  const minError = Number(operatorMinErrorEl.value || 0);
  const rows = operatorRows(report)
    .filter((row) => phaseFilter === "all" || row.phase === phaseFilter)
    .filter((row) => rankFilter === "all" || String(row.rank) === rankFilter)
    .filter((row) => Number(row.abs_err_ms || 0) >= minError)
    .sort((a, b) => Number(b[sortKey] || 0) - Number(a[sortKey] || 0));
  operatorCompareEl.innerHTML = `
    <div class="inline-pills operator-summary">
      <span class="pill">rows ${rows.length}</span>
      <span class="pill">matched ${summary.matched_rows ?? 0}</span>
      <span class="pill">coverage ${formatPct(summary.coverage_estimate_ms_pct ?? 0)}</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Phase</th><th>Rank</th><th>Host</th><th>Node</th><th>Target</th><th>Scope</th><th>Est</th><th>Measured</th><th>Abs Err</th><th>Rel Err</th><th>Calls</th><th>Match</th></tr></thead>
        <tbody>
          ${rows.slice(0, 300).map((row) => `<tr><td>${row.phase}</td><td>${row.rank}</td><td>${row.host || "-"}</td><td>${row.node_name || "-"}</td><td>${row.target}</td><td>${row.scope}</td><td>${formatMs(row.est_ms)}</td><td>${formatMs(row.measured_ms)}</td><td>${formatMs(row.abs_err_ms)}</td><td>${formatPct(row.rel_err_pct)}</td><td>${row.calls}</td><td>${row.match_method} (${Number(row.match_confidence || 0).toFixed(2)})</td></tr>`).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderMeasuredPending(run) {
  measuredMetaEl.textContent = run.status === "estimate_ready"
    ? `Estimate 已完成，系统正在执行 measured 的 prefill/decode/request 校验。`
    : `Measured timings 来自 estimate 完成后的多次 CUDA 同步实测。`;
  measuredCardsEl.innerHTML = `
    <article class="pipeline-card">
      <div class="metric-label">Validation 状态</div>
      <div class="metric-value">进行中</div>
      <div>estimate 已在上方显示</div>
      <div>当前阶段 ${statusLabel(run.status)}</div>
    </article>
  `;
  measuredEl.innerHTML = '<div class="muted">Measured timings 仍在采集中。</div>';
}

function renderProfilesPending() {
  profileEl.innerHTML = '<div class="muted">Profiler 表格会在 measurement 阶段完成后出现。</div>';
}

function renderRun(run) {
  if (!run) {
    titleEl.textContent = "尚未选择运行任务";
    metaEl.textContent = "启动一次运行后，这里会显示分析结果。";
    pipelineMetaEl.textContent = "这里会展示在进入 measured 阶段之前，为构建 estimate 预先花费了多少分析时间。";
    pipelineCardsEl.innerHTML = "";
    pipelineStepsEl.innerHTML = "";
    estimateCardsEl.innerHTML = "";
    executionCardsEl.innerHTML = "";
    measuredMetaEl.textContent = "Measured timings 来自 estimate 完成之后的多次 CUDA 同步实测。";
    measuredCardsEl.innerHTML = "";
    estimateEl.innerHTML = "";
    measuredEl.innerHTML = "";
    calibrationEl.innerHTML = "";
    profileEl.innerHTML = "";
    rankTimingsEl.innerHTML = "";
    commTablesEl.innerHTML = "";
    operatorCompareEl.innerHTML = "";
    topOpsEl.innerHTML = "";
    logsEl.textContent = "";
    graphSelectEl.innerHTML = "";
    graphLinksEl.innerHTML = "";
    graphViewerEl.innerHTML = '<div class="muted">尚未选择图产物。</div>';
    graphMetaEl.textContent = "每次运行可按需开启 graph export，完成后会显示在这里。";
    setBanner("已就绪，可启动分析。", "idle");
    return;
  }
  const execution = executionInfo(run.report, run.request);
  titleEl.textContent = run.request.model_path.split("/").slice(-1)[0];
  metaEl.textContent = `${statusLabel(run.status)} · ${execution.parallel_mode} · ${placementSummary(execution)} · 分析总耗时 ${formatSeconds(run.timings?.total_wall_time_s)} · 输出 ${run.output_dir || "待生成"}`;
  logsEl.textContent = [run.stdout, run.stderr].filter(Boolean).join("\n\n");
  setBanner(
    run.error || `${statusLabel(run.status)} · ${durationText(run)}`,
    run.status === "completed" ? "completed" : run.status === "failed" ? "failed" : isActiveStatus(run.status) ? "running" : "idle",
  );
  if (!run.report) {
    pipelineCardsEl.innerHTML = "";
    pipelineStepsEl.innerHTML = "";
    estimateCardsEl.innerHTML = "";
    executionCardsEl.innerHTML = "";
    measuredCardsEl.innerHTML = "";
    estimateEl.innerHTML = "";
    measuredEl.innerHTML = "";
    calibrationEl.innerHTML = "";
    profileEl.innerHTML = "";
    rankTimingsEl.innerHTML = '<div class="muted">暂时还没有 rank timing 数据。</div>';
    commTablesEl.innerHTML = '<div class="muted">暂时还没有 communication 数据。</div>';
    operatorCompareEl.innerHTML = '<div class="muted">暂时还没有算子 compare 数据。</div>';
    topOpsEl.innerHTML = "";
    profileEl.innerHTML = '<div class="muted">暂时还没有 profiler 数据。</div>';
    renderGraphViewer(run);
    return;
  }
  renderPipeline(run);
  renderExecutionOverview(run.report, run.request);
  if (reportHasEstimate(run.report)) {
    renderEstimateOverview(run.report);
    renderEstimateBreakdowns(run.report);
    renderTopOps(run.report);
    renderCalibration(run.report);
  } else {
    estimateCardsEl.innerHTML = '<div class="muted">Estimate 仍在生成中。</div>';
    estimateEl.innerHTML = '<div class="muted">Estimate breakdown 会在 estimation 阶段完成后显示。</div>';
    topOpsEl.innerHTML = '<div class="muted">主要贡献项会随 estimate 一起显示。</div>';
    calibrationEl.innerHTML = '<div class="muted">Calibration 数据会随 estimate 一起显示。</div>';
  }
  if (reportHasMeasured(run.report)) {
    renderMeasuredOverview(run.report);
    renderMeasured(run.report);
  } else {
    renderMeasuredPending(run);
  }
  if (reportHasProfiles(run.report)) {
    renderProfiles(run.report);
  } else {
    renderProfilesPending();
  }
  renderRankTimings(run.report);
  renderComm(run.report);
  renderOperatorCompare(run.report);
  renderGraphViewer(run);
}

async function refreshRuns() {
  const payload = await fetchJson("/api/runs");
  state.runs = payload.runs;
  if (!state.activeRunId && state.runs.length) {
    state.activeRunId = state.runs[0].run_id;
  }
  renderRunList();
  const active = state.runs.find((run) => run.run_id === state.activeRunId);
  renderRun(active || null);
  if (active && isActiveStatus(active.status)) {
    schedulePoll(active.run_id);
  }
}

async function selectRun(runId) {
  state.activeRunId = runId;
  const run = await fetchJson(`/api/runs/${runId}`);
  const index = state.runs.findIndex((item) => item.run_id === runId);
  if (index >= 0) {
    state.runs[index] = run;
  } else {
    state.runs.unshift(run);
  }
  renderRunList();
  renderRun(run);
  if (isActiveStatus(run.status)) {
    schedulePoll(runId);
  }
}

function schedulePoll(runId) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(() => selectRun(runId).catch((error) => setBanner(error.message, "failed")), 2500);
}

async function loadConfig() {
  const payload = await fetchJson("/api/config");
  state.config = payload;
  state.environment = payload.environment || null;
  fillForm(payload.defaults);
  refreshDeploymentControls();
  refreshPlacementControls();
  refreshRunnerControls();
  renderEnvironmentStatus();
  const datalist = document.getElementById("model-options");
  datalist.innerHTML = payload.models.map((path) => `<option value="${path}"></option>`).join("");
}

runButton.addEventListener("click", async () => {
  if (!managedEnvironment()?.ready) {
    setBanner("环境尚未准备好，请先等待自动准备完成或手动点击“准备环境”。", "failed");
    return;
  }
  runButton.disabled = true;
  try {
    setBanner("正在启动运行任务...", "running");
    const response = await fetchJson("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });
    state.activeRunId = response.run_id;
    await refreshRuns();
    await selectRun(response.run_id);
  } catch (error) {
    setBanner(error.message, "failed");
  } finally {
    runButton.disabled = false;
  }
});

prepareButton.addEventListener("click", async () => {
  prepareButton.disabled = true;
  try {
    setBanner("正在准备环境...", "running");
    state.environment = await fetchJson("/api/environment/prepare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formPayload()),
    });
  } catch (error) {
    setBanner(error.message, "failed");
  } finally {
    renderEnvironmentStatus();
  }
});

stopEnvButton.addEventListener("click", async () => {
  stopEnvButton.disabled = true;
  try {
    state.environment = await fetchJson("/api/environment/stop", { method: "POST" });
    setBanner("环境已停止。", "idle");
  } catch (error) {
    setBanner(error.message, "failed");
  } finally {
    renderEnvironmentStatus();
  }
});

refreshButton.addEventListener("click", () => refreshRuns().catch((error) => setBanner(error.message, "failed")));
graphSelectEl.addEventListener("change", (event) => updateGraphPreview(event.target.value));
localGpuCountEl.addEventListener("change", () => refreshPlacementControls({ updatePhysicalDevices: true }));
if (deploymentModeEl) {
  deploymentModeEl.addEventListener("change", () => {
    const mode = deploymentModeEl.value;
    if (mode === "single_host") {
      namedField("nnodes").value = "1";
      if (Number(namedField("nproc_per_node").value || 0) <= 0) {
        namedField("nproc_per_node").value = String(Math.max(1, Number(localGpuCountEl.value || 1)));
      }
    } else {
      if (Number(namedField("nnodes").value || 1) <= 1) {
        namedField("nnodes").value = "2";
      }
      namedField("parallel_mode").value = "tp";
    }
    refreshPlacementControls({ updatePhysicalDevices: singleHostRequested() });
  });
}
[namedField("physical_devices"), namedField("nnodes"), namedField("parallel_mode")].forEach((element) => {
  element.addEventListener("change", () => {
    if (element === namedField("physical_devices") && singleHostRequested()) {
      const devices = parsePhysicalDevices(namedField("physical_devices").value);
      if (devices.length) {
        localGpuCountEl.value = String(devices.length);
      }
      refreshPlacementControls({ updatePhysicalDevices: true });
      return;
    }
    if (element === namedField("parallel_mode") && singleHostRequested()) {
      localGpuCountEl.value = namedField("parallel_mode").value === "single"
        ? "1"
        : String(Math.max(2, Number(localGpuCountEl.value || 2)));
      refreshPlacementControls({ updatePhysicalDevices: true });
      return;
    }
    refreshDeploymentControls();
  });
});
namedField("runner").addEventListener("change", () => refreshRunnerControls());
[operatorPhaseFilterEl, operatorRankFilterEl, operatorSortEl, operatorMinErrorEl].forEach((element) => {
  element.addEventListener("change", () => {
    const active = state.runs.find((run) => run.run_id === state.activeRunId);
    if (active?.report) renderOperatorCompare(active.report);
  });
});

Promise.all([loadConfig(), refreshRuns(), fetchEnvironment()]).catch((error) => setBanner(error.message, "failed"));
