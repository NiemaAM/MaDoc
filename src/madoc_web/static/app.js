/* MaDoc: the interface (Alpine.js component). One page, three views: the acts, the wizard of one act, the
   settings and the monitoring. The server keeps everything: this file only shows it and sends the changes. */
(function () {
  const LANGS = ["en", "fr", "ar"];
  const LONG = ["TEXTEDECLA", "TEXTEMENTIONS", "TEXTEMENTIONSFR", "ADRESSEPARENT"];

  async function api(method, path, body, raw) {
    const options = { method, headers: {} };
    if (body instanceof FormData) options.body = body;
    else if (body !== undefined) { options.body = JSON.stringify(body); options.headers["Content-Type"] = "application/json"; }
    let answer;
    try { answer = await fetch(path, options); } catch (_) { throw new Error("network"); }
    if (!answer.ok) {
      let detail = answer.statusText;
      try { detail = (await answer.json()).detail || detail; } catch (_) { /* not JSON */ }
      const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      error.status = answer.status;
      throw error;
    }
    return raw ? answer : answer.json();
  }

  window.madoc = function () {
    let ed = null;          // the LineEditor: kept out of Alpine's reactive state (it holds DOM nodes)
    return {
      // ---- state ----
      lang: "en", view: "docs", loading: false, toasts: [],
      docs: [], retention: 1, meta: null,
      doc: null, step: 1, page: "2", follow: 0, mode: "select", selected: null, canUndo: false,
      saving: false, uploading: null, accepted: [],
      settings: null, released: "",
      mon: null, minutes: 15,
      camera: { open: false, page: "2", ready: false, error: "" },
      timer: null, monTimer: null, stream: null, pending: null, saveTimer: null,

      // ---- start, routes ----
      async init() {
        let saved = null;
        try { saved = localStorage.getItem("madoc-lang"); } catch (_) { /* storage blocked */ }
        const wanted = saved || (navigator.language || "en").slice(0, 2);
        this.setLang(LANGS.includes(wanted) ? wanted : "en");
        window.addEventListener("hashchange", () => this.route());
        window.addEventListener("resize", () => { if (ed) ed.render(); });
        // the keys of the line editor (Delete, Escape, Ctrl+Z) work wherever the focus is, but in a field
        window.addEventListener("keydown", (e) => {
          const field = e.target && ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName);
          if (ed && this.view === "doc" && this.step === 2 && !this.running() && !field && !this.camera.open) ed.key(e);
        });
        try { this.meta = await api("GET", "/api/meta"); } catch (e) { this.fail(e); }
        try { this.settings = await api("GET", "/api/settings"); } catch (e) { this.fail(e); }
        this.route();
      },
      setLang(lang) {
        this.lang = lang;
        document.documentElement.lang = lang;
        document.documentElement.dir = lang === "ar" ? "rtl" : "ltr";
        try { localStorage.setItem("madoc-lang", lang); } catch (_) { /* storage blocked */ }
      },
      t(key, values) {
        const entry = window.MADOC_TEXT[key];
        if (!entry) return key;
        let text = entry[LANGS.indexOf(this.lang)] || entry[0];
        for (const name in values || {}) text = text.replaceAll("{" + name + "}", values[name]);
        return text;
      },
      route() {
        const parts = location.hash.replace(/^#\/?/, "").split("/");
        clearInterval(this.monTimer); this.monTimer = null;
        if (parts[0] === "doc" && parts[1]) { this.view = "doc"; this.openDoc(parts[1]); return; }
        this.stopPolling(); this.closeCamera();
        if (parts[0] === "settings") { this.view = "settings"; this.loadSettings(); return; }
        if (parts[0] === "monitor") {
          this.view = "monitor"; this.loadMonitor();
          this.monTimer = setInterval(() => { if (!document.hidden) this.loadMonitor(); }, 5000);
          return;
        }
        this.view = "docs"; this.loadDocs();
      },
      toast(text, kind) {
        const id = Date.now() + Math.random();
        this.toasts.push({ id, text, kind: kind || "" });
        setTimeout(() => { this.toasts = this.toasts.filter((x) => x.id !== id); }, kind === "bad" ? 9000 : 3500);
      },
      fail(error) { this.toast(error.message === "network" ? this.t("err_network") : error.message, "bad"); },

      // ---- the list ----
      async loadDocs() {
        this.loading = true;
        try { const answer = await api("GET", "/api/docs"); this.docs = answer.docs; this.retention = answer.retention_hours; }
        catch (e) { this.fail(e); }
        this.loading = false;
      },
      async newDoc() {
        try { const doc = await api("POST", "/api/docs"); location.hash = "#/doc/" + doc.id; } catch (e) { this.fail(e); }
      },
      async removeDoc(d) {
        if (!confirm(this.t("confirm_delete"))) return;
        try { await api("DELETE", "/api/docs/" + d.id); this.loadDocs(); } catch (e) { this.fail(e); }
      },
      docState(d) {
        if (d.job === "running" || d.job === "queued") return this.t("st_running");
        if (d.job === "error") return this.t("st_error");
        if (!d.pages.length) return this.t("st_empty");
        if (d.valid === true) return this.t("st_valid");
        if (d.valid === false) return this.t("st_invalid");
        return this.t("st_draft");
      },
      docChip(d) {
        if (d.job === "error") return "bad";
        if (d.job === "running" || d.job === "queued") return "warn";
        return d.valid === true ? "ok" : (d.valid === false ? "warn" : "");
      },
      ago(at) {
        const minutes = Math.max(0, Math.round((Date.now() / 1000 - at) / 60));
        if (minutes < 1) return this.t("ago_now");
        return minutes < 60 ? this.t("ago_min", { n: minutes }) : this.t("ago_hour", { n: Math.round(minutes / 60) });
      },

      // ---- one act ----
      async openDoc(id) {
        if (this.doc && this.doc.id === id) return;
        this.stopPolling();
        this.doc = null; ed = null; this.selected = null; this.accepted = []; this.pending = null;
        try {
          const doc = await api("GET", "/api/docs/" + id);
          this.page = "2"; this.follow = 0; this.mode = "select";
          this.setDoc(doc, true);
          this.step = this.doc.record ? 4 : this.bestStep();
          this.$nextTick(() => { this.mountEditor(); if (this.running()) this.poll(); });
        } catch (e) { this.fail(e); location.hash = "#/"; }
      },
      setDoc(doc, first) {
        const before = this.doc;
        this.doc = doc;
        if (!this.has(this.page)) this.page = this.presentPages()[0] || "2";
        if (doc.spelling && (!before || !before.spelling || before.spelling.base !== doc.spelling.base)) {
          this.accepted = doc.spelling.changes.map((c, i) => (c.accepted ? i : -1)).filter((i) => i >= 0);
        }
        if (!doc.spelling) this.accepted = [];
        if (this.follow) this.step = Math.min(this.follow, this.bestStep());
        if (!first) this.syncEditor();
      },
      mountEditor() {
        if (!this.$refs.canvas || !this.doc) return;
        ed = new window.LineEditor(this.$refs.canvas, {
          onChange: (lines) => this.saveLines(lines),
          onSelect: (id) => { this.selected = id; if (id && this.step === 3) { const input = document.getElementById("line-" + id); if (input) input.focus({ preventScroll: false }); } },
          onHover: () => {},
          onState: () => { this.canUndo = ed.canUndo(); this.selected = ed.selected; },
        });
        this.syncEditor(true);
      },
      syncEditor(reset) {
        if (!ed || !this.doc) return;
        const state = this.doc.pages[this.page];
        ed.threshold = this.settings ? this.settings.values.confidence_threshold : 0.8;
        if (state.state === "empty") { ed.setLines([], false); return; }
        ed.setImage(this.imageUrl(this.page), state.width, state.height);
        ed.setMode(this.step === 2 && !this.running() ? this.mode : "view");
        ed.setLines(state.lines, !reset);
        if (ed.selectAfter !== undefined && ed.selectAfter !== null) {
          const line = state.lines[ed.selectAfter];
          ed.selectAfter = null;
          if (line) ed.setSelected(line.id);
        }
      },
      has(p) { return !!this.doc && this.doc.pages[p].state !== "empty"; },
      presentPages() { return this.doc ? ["2", "1"].filter((p) => this.has(p)) : []; },
      lines() { return this.doc ? this.doc.pages[this.page].lines : []; },
      dirtyCount() { return this.lines().filter((l) => l.dirty).length; },
      // rev changes with every scan stored: a new scan is a new address, never the previous one from the cache
      imageUrl(p) { const s = this.doc.pages[p]; return "/api/docs/" + this.doc.id + "/pages/" + p + "/image?v=" + encodeURIComponent(s.rev || s.image + s.width); },
      running() { return !!this.doc && !!this.doc.job && (this.doc.job.state === "running" || this.doc.job.state === "queued"); },
      allIn(states) { const pages = this.presentPages(); return pages.length > 0 && pages.every((p) => states.includes(this.doc.pages[p].state)); },
      stepDone(n) {
        if (!this.doc) return false;
        if (n === 1) return this.has("2");
        if (n === 2) return this.has("2") && this.allIn(["segmented", "recognized"]);
        if (n === 3) return this.has("2") && this.allIn(["recognized"]) && !this.presentPages().some((p) => this.doc.pages[p].lines.some((l) => l.dirty));
        return !!this.doc.record && !this.doc.stale;
      },
      stepOpen(n) { return n === 1 || this.stepDone(n - 1) || (n === 4 && !!this.doc && !!this.doc.record); },
      bestStep() { for (const n of [4, 3, 2]) if (this.stepDone(n - 1) && (n < 4 || (this.doc.record && !this.doc.stale))) return n; return 1; },
      goStep(n) {
        if (n < 1 || n > 4 || !this.stepOpen(n)) return;
        this.follow = 0; this.step = n;
        this.$nextTick(() => this.syncEditor());
      },
      setPage(p) { this.page = p; this.selected = null; if (ed) ed.selected = null; this.syncEditor(true); },
      setMode(mode) { this.mode = mode; this.syncEditor(); },
      zoom(step) { if (ed) ed.zoom(step); },
      boxLine() { if (ed) ed.boxSelected(); },
      deleteLine(id) { if (ed) { if (id) ed.deleteLine(id); else ed.deleteSelected(); } },
      undoEdit() { if (ed) ed.undo(); },
      select(id, quiet) { this.selected = id; if (ed) ed.setSelected(id, !quiet); },
      hover(id) { if (ed) ed.setHover(id); },
      hoverField(row) {
        if (!ed || !this.doc) return;
        const source = this.doc.field_lines[row.ar] || (row.fr && this.doc.field_lines[row.fr]);
        if (!source || !source.lines.length) { ed.setHighlight([]); return; }
        if (source.page !== this.page && this.has(source.page)) this.setPage(source.page);
        ed.setHighlight(source.lines);
      },

      // the wizard's button
      needsRead() { return this.presentPages().some((p) => this.doc.pages[p].state !== "recognized" || this.doc.pages[p].lines.some((l) => l.dirty)); },
      canNext() {
        if (!this.doc) return false;
        if (this.step === 1) return this.has("2") && !this.uploading;
        if (this.step === 2) return this.doc.pages["2"].lines.length > 0;
        if (this.step === 3) return this.doc.pages["2"].lines.some((l) => l.text);
        return false;
      },
      nextLabel() {
        if (!this.doc) return "";
        const auto = this.settings && this.settings.values.auto_run;
        if (this.step === 1) return this.allIn(["segmented", "recognized"]) ? this.t("to_lines") : (auto ? this.t("start") : this.t("find_lines"));
        if (this.step === 2) return this.needsRead() ? this.t("read_text") : this.t("to_text");
        return !this.doc.record || this.doc.stale ? this.t("extract") : this.t("to_record");
      },
      async next() {
        if (this.step === 1) {
          if (this.allIn(["segmented", "recognized"])) return this.goStep(2);
          const auto = this.settings && this.settings.values.auto_run;
          return this.run(auto ? "auto" : "segment", {}, auto ? 4 : 2);
        }
        if (this.step === 2) return this.needsRead() ? this.run("recognize", { only_dirty: true }, 3) : this.goStep(3);
        if (this.step === 3) return !this.doc.record || this.doc.stale ? this.run("extract", {}, 4) : this.goStep(4);
      },

      // ---- runs ----
      async run(kind, options, follow) {
        if (kind === "segment" && options && options.pages && this.doc.pages[options.pages[0]].lines.length && !confirm(this.t("confirm_segment"))) return;
        try {
          await this.flush();
          this.follow = follow || 0;
          this.setDoc(await api("POST", "/api/docs/" + this.doc.id + "/run", { kind, ...(options || {}) }));
          if (this.running()) this.poll();
          else if (follow) this.step = Math.min(follow, this.bestStep());
        } catch (e) { this.fail(e); }
      },
      poll() {
        this.stopPolling();
        const id = this.doc.id;
        let version = this.doc.version;
        this.timer = setInterval(async () => {
          if (!this.doc || this.doc.id !== id) return this.stopPolling();
          try {
            const status = await api("GET", "/api/docs/" + id + "/status");
            const done = !status.job || (status.job.state !== "running" && status.job.state !== "queued");
            if (status.version !== version || done) {
              version = status.version;
              const doc = await api("GET", "/api/docs/" + id);
              if (!this.doc || this.doc.id !== id) return;
              if (doc.job && doc.job.state !== "running" && doc.job.state !== "queued") this.stopPolling();
              this.setDoc(doc);
              if (!this.running()) { this.follow = 0; this.syncEditor(); }
            }
          } catch (e) { if (e.status === 404) { this.stopPolling(); location.hash = "#/"; } else if (e.message !== "network") console.error(e); }
        }, 800);
      },
      stopPolling() { clearInterval(this.timer); this.timer = null; },
      jobLabel() {
        const job = this.doc && this.doc.job;
        if (!job) return "";
        if (job.state === "queued") return this.t("job_queued");
        const [name, page] = (job.label || "").split(" ");
        return this.t("job_" + name) + (page ? " — " + this.t("of_page", { n: page }) : "") + "  (" + (job.index + 1) + "/" + job.steps.length + ")";
      },
      jobPercent() { const job = this.doc && this.doc.job; return job && job.steps.length ? Math.round(((job.index + 0.5) / job.steps.length) * 100) : 0; },

      // ---- pages ----
      async upload(p, file) {
        if (!file || !this.doc) return;
        const limit = this.meta ? this.meta.max_upload_mb : 25;
        if (file.size > limit * 1024 * 1024) return this.toast("> " + limit + " MB", "bad");
        this.uploading = p;
        try {
          const form = new FormData();
          form.append("file", file, file.name || "photo.jpg");
          this.setDoc(await api("PUT", "/api/docs/" + this.doc.id + "/pages/" + p, form));
        } catch (e) { this.fail(e); }
        this.uploading = null;
      },
      async removePage(p) {
        try { this.setDoc(await api("DELETE", "/api/docs/" + this.doc.id + "/pages/" + p)); } catch (e) { this.fail(e); }
      },
      async openCamera(p) {
        this.camera = { open: true, page: p, ready: false, error: "" };
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
          this.camera.error = this.t("camera_denied", { why: window.isSecureContext ? "no camera API" : "HTTPS" });
          return;
        }
        try {
          this.stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" }, width: { ideal: 3840 }, height: { ideal: 2160 } }, audio: false });
          if (!this.camera.open) return this.closeCamera();
          this.$refs.video.srcObject = this.stream;
          await this.$refs.video.play();
          this.camera.ready = true;
        } catch (e) { this.camera.error = this.t("camera_denied", { why: e.name || "error" }); }
      },
      closeCamera() {
        if (this.stream) this.stream.getTracks().forEach((track) => track.stop());
        this.stream = null;
        if (this.$refs.video) this.$refs.video.srcObject = null;
        this.camera.open = false; this.camera.ready = false;
      },
      shoot() {
        const video = this.$refs.video;
        if (!video.videoWidth) return;
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth; canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);
        const p = this.camera.page;
        canvas.toBlob((blob) => { if (blob) this.upload(p, new File([blob], "photo_page" + p + ".jpg", { type: "image/jpeg" })); }, "image/jpeg", 0.92);
        this.closeCamera();
      },

      // ---- lines, text ----
      saveLines(lines) {
        this.pending = { page: this.page, lines: lines.map((l) => ({ id: l.id, polygon: l.polygon, baseline: l.baseline })) };
        this.doc.pages[this.page].lines = lines;      // shown at once; the server's answer replaces it
        this.saving = true;
        clearTimeout(this.saveTimer);
        this.saveTimer = setTimeout(() => this.flush().catch((e) => this.fail(e)), 350);
      },
      async flush() {                       // send the last edit of the lines now (before a run, before leaving)
        clearTimeout(this.saveTimer);
        const pending = this.pending;
        if (!pending) { this.saving = false; return; }
        this.pending = null;
        try {
          const doc = await api("PUT", "/api/docs/" + this.doc.id + "/pages/" + pending.page + "/lines", { lines: pending.lines });
          if (!this.pending) this.setDoc(doc);
        } finally { if (!this.pending) this.saving = false; }
      },
      async saveText(line, text) {
        if (text.trim() === line.text) return;
        try { this.setDoc(await api("PATCH", "/api/docs/" + this.doc.id + "/pages/" + this.page + "/lines/" + line.id, { text })); }
        catch (e) { this.fail(e); }
      },
      isLow(l) { return l.confidence !== null && l.confidence !== undefined && this.settings && l.confidence < this.settings.values.confidence_threshold; },
      conf(l) { return l.confidence === null || l.confidence === undefined ? "" : Math.round(l.confidence * 100) + " %"; },
      toggleChange(i) { this.accepted = this.accepted.includes(i) ? this.accepted.filter((x) => x !== i) : this.accepted.concat([i]); },
      async applySpelling() {
        try { this.setDoc(await api("POST", "/api/docs/" + this.doc.id + "/spelling/apply", { accepted: this.accepted })); }
        catch (e) { this.fail(e); }
      },
      async dropSpelling() {
        try { this.setDoc(await api("DELETE", "/api/docs/" + this.doc.id + "/spelling")); } catch (e) { this.fail(e); }
      },

      // ---- record ----
      groups() {
        const out = [];
        for (const [group, label, ar, fr] of (this.meta ? this.meta.rows : [])) {
          let entry = out.find((g) => g.name === group);
          if (!entry) { entry = { name: group, rows: [] }; out.push(entry); }
          entry.rows.push({ label, ar, fr });
        }
        return out;
      },
      isLong(field) { return LONG.includes(field); },
      isEdited(row) { const edits = (this.doc && this.doc.record_edits) || {}; return row.ar in edits || (row.fr && row.fr in edits); },
      rowClass(row) {
        const issues = (this.doc && this.doc.report && this.doc.report.issues) || [];
        const mine = issues.filter((i) => i.field === row.ar || i.field === row.fr);
        return mine.some((i) => i.level === "error") ? "error" : (mine.length ? "warning" : "");
      },
      fieldLabel(field) {
        for (const [, label, ar, fr] of (this.meta ? this.meta.rows : [])) {
          if (field === ar || field === fr) return this.t("f_" + label) + (field === fr ? " (FR)" : "") + " · " + field;
        }
        return field;
      },
      async saveField(field, value) {
        if (String(this.doc.record[field]) === String(value)) return;
        try { this.setDoc(await api("PATCH", "/api/docs/" + this.doc.id + "/record", { [field]: value })); } catch (e) { this.fail(e); }
      },
      dl(kind) { return this.doc ? "/api/docs/" + this.doc.id + "/download/" + kind : "#"; },

      // ---- settings ----
      async loadSettings() { try { this.settings = await api("GET", "/api/settings"); } catch (e) { this.fail(e); } },
      async saveSettings() {
        this.loading = true;
        try { this.settings = await api("PUT", "/api/settings", this.settings.values); this.toast(this.t("saved"), "ok"); }
        catch (e) { this.fail(e); }
        this.loading = false;
      },
      async releaseModels() {
        try {
          const answer = await api("POST", "/api/settings/release");
          this.released = this.t("set_released", { k: answer.kraken, o: answer.ollama ? "ok" : "—" });
        } catch (e) { this.fail(e); }
      },

      // ---- monitoring ----
      async loadMonitor() { try { this.mon = await api("GET", "/api/monitor?minutes=" + this.minutes); } catch (e) { /* the next tick tries again */ } },
      secs(v) { return v === null || v === undefined ? "—" : (v < 10 ? v.toFixed(2) : v.toFixed(1)) + " s"; },
      clock(at) { return new Date(at * 1000).toLocaleTimeString(); },
      duration(seconds) {
        const h = Math.floor(seconds / 3600), m = Math.floor((seconds % 3600) / 60);
        return h ? h + " h " + m + " min" : m + " min";
      },
      gb(bytes) { return (bytes / 1073741824).toFixed(1); },
      spark(points, value, max, color) {
        const since = this.mon.now - this.minutes * 60, W = 300, H = 70;
        const usable = points.map((p) => [p.at, value(p)]).filter((p) => p[1] !== null && p[1] !== undefined);
        if (!usable.length) return "";
        const top = max || Math.max(1, ...usable.map((p) => p[1])) * 1.15;
        const xy = usable.map(([at, v]) => [Math.max(0, ((at - since) / (this.minutes * 60)) * W), H - 2 - (Math.min(v, top) / top) * (H - 6)]);
        const path = xy.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
        const area = path + ` L${xy[xy.length - 1][0].toFixed(1)} ${H} L${xy[0][0].toFixed(1)} ${H} Z`;
        return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img"><line x1="0" y1="${H - 1}" x2="${W}" y2="${H - 1}" stroke="currentColor" stroke-opacity=".15"/>` +
          `<path d="${area}" fill="${color}" fill-opacity=".15"/><path d="${path}" fill="none" stroke="${color}" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>`;
      },
      tiles() {
        const points = this.mon.samples, last = points[points.length - 1] || {};
        const gpu = (last.gpus && last.gpus[0]) || null;
        const g = (key) => (p) => (p.gpus && p.gpus[0] ? p.gpus[0][key] : null);
        const tiles = [
          { key: "cpu", title: this.t("mon_cpu"), value: last.cpu === null || last.cpu === undefined ? "—" : Math.round(last.cpu) + " %",
            chart: this.spark(points, (p) => p.cpu, 100, "#2456c9"), note: this.mon.cpu_count + " threads" },
          { key: "mem", title: this.t("mon_memory"), value: last.memory_used ? this.gb(last.memory_used) + " / " + this.gb(last.memory_total) + " GB" : "—",
            chart: this.spark(points, (p) => p.memory_used, last.memory_total, "#7a4fd0"),
            note: last.process_memory ? this.t("mon_process") + ": " + this.gb(last.process_memory) + " GB" : "" },
          { key: "gpu", title: this.t("mon_gpu"), value: gpu && gpu.util !== null ? Math.round(gpu.util) + " %" : "—",
            chart: this.spark(points, g("util"), 100, "#1a7f4b"), note: gpu ? gpu.name : this.t("mon_no_gpu") },
          { key: "gmem", title: this.t("mon_gpu_memory"), value: gpu && gpu.memory_used !== null ? (gpu.memory_used / 1024).toFixed(1) + " / " + (gpu.memory_total / 1024).toFixed(1) + " GB" : "—",
            chart: this.spark(points, g("memory_used"), gpu ? gpu.memory_total : null, "#c2571a"), note: gpu && gpu.power !== null ? Math.round(gpu.power) + " W" : "" },
          { key: "temp", title: this.t("mon_gpu_temp"), value: gpu && gpu.temperature !== null ? Math.round(gpu.temperature) + " °C" : "—",
            chart: this.spark(points, g("temperature"), 100, "#c0392b"), note: "" },
          { key: "queue", title: this.t("mon_queue"), value: String(last.queue === undefined ? 0 : last.queue),
            chart: this.spark(points, (p) => p.queue, null, "#667085"),
            note: last.disk_used ? this.t("mon_disk") + ": " + this.gb(last.disk_used) + " / " + this.gb(last.disk_total) + " GB" : "" },
        ];
        return gpu ? tiles : tiles.filter((tile) => tile.key !== "gmem" && tile.key !== "temp");
      },
    };
  };
})();