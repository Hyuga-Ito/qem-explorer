// QEM Explorer drag-and-drop composer (Streamlit components v2).
// The component never edits the circuit itself: it sends validated-on-the-server *actions*
// ({type: "add"|"move"|"retarget"|"delete"|"param"|"flip"|"duplicate"|"clear"|"qubits"|"undo"|"redo", ...})
// through setTriggerValue("edit", ...). Python checks every action and sends back the new circuit.
// No user text is ever inserted as HTML: labels are set with textContent.

const CELL = 54, ROW = 58, LEFT0 = 74, TOP = 14, IOW = 40;
let LEFT = LEFT0;
const SVGNS = "http://www.w3.org/2000/svg";

function el(tag, attrs, parent) {
  const e = document.createElement(tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v; else if (k === "cls") e.className = v; else e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}
function sv(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) {
    if (k === "text") e.textContent = v; else e.setAttribute(k, v);
  }
  if (parent) parent.appendChild(e);
  return e;
}

export default function (component) {
  const { data, parentElement, setTriggerValue } = component;
  let root = parentElement.querySelector(".qx-root");
  if (!root) {
    root = el("div", { cls: "qx-root", tabindex: "0" }, parentElement);
    root.__state = { sel: null, drag: null, version: null, scroll: 0 };
    setup(root);
  }
  root.__data = data;
  root.__send = (payload) => {
    payload.v = root.__data.version;
    setTriggerValue("edit", payload);
  };
  render(root);
  return () => {};
}

function setup(root) {
  root.addEventListener("keydown", (ev) => {
    const st = root.__state;
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT") return;
    if ((ev.key === "Delete" || ev.key === "Backspace") && st.sel !== null) {
      root.__send({ type: "delete", idx: st.sel }); st.sel = null; ev.preventDefault();
    } else if ((ev.ctrlKey || ev.metaKey) && (ev.key === "z" || ev.key === "Z")) {
      root.__send({ type: ev.shiftKey ? "redo" : "undo" }); ev.preventDefault();
    } else if ((ev.ctrlKey || ev.metaKey) && (ev.key === "y" || ev.key === "Y")) {
      root.__send({ type: "redo" }); ev.preventDefault();
    } else if (ev.key === "Escape") {
      st.sel = null; render(root);
    }
  });
  root.addEventListener("pointermove", (ev) => onMove(root, ev));
  root.addEventListener("pointerup", (ev) => onUp(root, ev));
  root.addEventListener("pointercancel", () => endDrag(root));
}

function colors(data, fam) {
  const c = data.colors[fam] || data.colors["pauli"];
  return { fill: c[0], text: c[1] };
}

function render(root) {
  const data = root.__data, st = root.__state;
  if (st.version !== data.version) {
    if (st.keepSel === undefined || st.keepSel === null) st.sel = null; else st.sel = st.keepSel;
    st.keepSel = null;
    st.version = data.version;
  }
  const oldCanvas = root.querySelector(".qx-canvas");
  if (oldCanvas) st.scroll = oldCanvas.scrollLeft;
  root.textContent = "";
  root.classList.toggle("qx-dark", !!data.dark);

  // ---------------- palette
  const pal = el("div", { cls: "qx-palette" }, root);
  const groups = {};
  for (const t of data.palette) (groups[t.group] = groups[t.group] || []).push(t);
  for (const [g, tiles] of Object.entries(groups)) {
    const box = el("div", { cls: "qx-group" }, pal);
    el("div", { cls: "qx-group-title", text: g }, box);
    const row = el("div", { cls: "qx-tiles" }, box);
    for (const t of tiles) {
      const c = colors(data, t.fam);
      const tile = el("div", { cls: "qx-tile" + (t.kind === "noise" ? " qx-noise" : ""), title: t.help || t.name }, row);
      tile.style.background = c.fill; tile.style.color = c.text;
      tile.textContent = t.label;
      tile.dataset.name = t.name;
      tile.addEventListener("pointerdown", (ev) => startDrag(root, ev, { kind: "new", name: t.name, label: t.label, fill: c.fill, text: c.text }));
    }
  }
  const tools = el("div", { cls: "qx-tools" }, pal);
  const btn = (label, title, payload, disabled) => {
    const b = el("button", { cls: "qx-btn", title: title, type: "button" }, tools);
    b.textContent = label; if (disabled) b.disabled = true;
    b.addEventListener("click", () => root.__send(payload));
    return b;
  };
  btn("↶ Undo", "Undo (Ctrl+Z)", { type: "undo" }, !data.can_undo);
  btn("↷ Redo", "Redo (Ctrl+Shift+Z)", { type: "redo" }, !data.can_redo);
  btn("+ qubit", "Add a qubit", { type: "qubits", n: data.n + 1 });
  btn("− qubit", "Remove the last qubit (and its operations)", { type: "qubits", n: data.n - 1 }, data.n <= 1);
  btn("Clear", "Remove every operation", { type: "clear" }, !data.ops.length);

  // ---------------- canvas
  const ncol = Math.max(0, ...data.ops.map((o) => o.col + 1));
  const cols = Math.max(ncol + 3, 12);
  // input preparation and basis change (set in "Inputs and measurement"): locked boxes around the editable grid
  const prep = data.prep || [], post = data.post || [];
  const npre = Math.max(0, ...prep.map((p) => p.length)), npost = Math.max(0, ...post.map((p) => p.length));
  const preW = npre ? npre * IOW + 10 : 0, postW = npost ? npost * IOW + 4 : 0;
  LEFT = LEFT0 + preW;
  const W = LEFT + cols * CELL + 60 + postW, H = TOP + data.n * ROW + 8;
  const canvas = el("div", { cls: "qx-canvas" }, root);
  const svg = sv("svg", { width: W, height: H, class: "qx-svg" }, canvas);
  root.__geom = { svg, cols, n: data.n };
  // grid cells (drop targets are computed from coordinates; cells are visual)
  for (let q = 0; q < data.n; q++) {
    const y = TOP + q * ROW + ROW / 2;
    sv("line", { x1: LEFT0 - 8, x2: W - 30, y1: y, y2: y, class: "qx-wire" }, svg);
    const lab = sv("text", { x: 8, y: y + 5, class: "qx-qlabel" }, svg);
    lab.textContent = "q" + q;
    const inp = (data.inputs && data.inputs[q]) || "0";
    const il = sv("text", { x: 34, y: y + 5, class: "qx-input" }, svg);
    il.textContent = "|" + inp + "⟩";
    (prep[q] || []).forEach((lab, k) => {
      const px = LEFT0 - 4 + k * IOW;
      const r = sv("rect", { x: px, y: y - 13, width: IOW - 6, height: 26, rx: 4, class: "qx-io" }, svg);
      const t = sv("text", { x: px + (IOW - 6) / 2, y: y + 4, class: "qx-io-t" + (lab.length > 3 ? " qx-io-small" : "") }, svg);
      t.textContent = lab;
      const tip = sv("title", {}, r); tip.textContent = "input preparation (set in Inputs and measurement)";
    });
    const mx0 = LEFT + cols * CELL + 6;
    (post[q] || []).forEach((lab, k) => {
      const px = mx0 + k * IOW;
      const r = sv("rect", { x: px, y: y - 13, width: IOW - 6, height: 26, rx: 4, class: "qx-io" }, svg);
      const t = sv("text", { x: px + (IOW - 6) / 2, y: y + 4, class: "qx-io-t" }, svg);
      t.textContent = lab;
      const tip = sv("title", {}, r); tip.textContent = "measurement-basis change (set in Inputs and measurement)";
    });
    if (data.measure) {
      const mx = mx0 + postW;
      sv("rect", { x: mx, y: y - 15, width: 30, height: 30, rx: 3, class: "qx-meas" }, svg);
      const mt = sv("text", { x: mx + 15, y: y + 5, class: "qx-meas-t" }, svg);
      const b = (data.basis && data.basis[q]) || "Z";
      mt.textContent = b === "Z" ? "M" : "M" + b.toLowerCase();
    }
  }
  for (let c = 0; c <= cols; c++) sv("line", { x1: LEFT + c * CELL, x2: LEFT + c * CELL, y1: TOP, y2: H - 8, class: "qx-gridline" }, svg);
  const hl = sv("rect", { x: 0, y: 0, width: CELL - 4, height: ROW - 6, rx: 6, class: "qx-drop", visibility: "hidden" }, svg);
  root.__hl = hl;

  // ---------------- operations
  data.ops.forEach((o) => drawOp(root, svg, o));
  canvas.scrollLeft = st.scroll || 0;

  // ---------------- trash + hint
  const foot = el("div", { cls: "qx-foot" }, root);
  const trash = el("div", { cls: "qx-trash", text: "🗑  Drop here to delete" }, foot);
  root.__trash = trash;
  el("div", { cls: "qx-hint", text: "Drag a tile onto a wire to add it · drag a gate to move it · drag the small end of a two-qubit gate to another wire · click to select, Delete to remove · Ctrl+Z undo" }, foot);

  // ---------------- inspector
  drawInspector(root);
  if (data.message) el("div", { cls: "qx-msg", text: data.message }, root);
}

function cx(o) { return LEFT + o.col * CELL + CELL / 2; }
function cy(q) { return TOP + q * ROW + ROW / 2; }

function drawOp(root, svg, o) {
  const data = root.__data, st = root.__state;
  const g = sv("g", { class: "qx-op" + (st.sel === o.i ? " qx-sel" : "") + (o.tag ? " qx-tag-" + o.tag : "") }, svg);
  const x = cx(o);
  const c = colors(data, o.fam);
  const box = (q, label, par, cls) => {
    const y = cy(q);
    const r = sv("rect", { x: x - 21, y: y - 20, width: 42, height: 40, rx: 4, class: cls || "qx-gate" }, g);
    r.style.fill = c.fill;
    const t = sv("text", { x: x, y: par ? y + 1 : y + 5, class: "qx-glabel" }, g);
    t.textContent = label; t.style.fill = c.text;
    if (par) { const p = sv("text", { x: x, y: y + 14, class: "qx-gpar" }, g); p.textContent = par; p.style.fill = c.text; }
    return r;
  };
  if (o.kind === "barrier") {
    const r = sv("rect", { x: x - 5, y: TOP + 4, width: 10, height: data.n * ROW - 8, class: "qx-barrier" }, g);
    r.addEventListener("pointerdown", (ev) => startDrag(root, ev, { kind: "move", idx: o.i, label: "barrier", fill: "#A2A9B0", text: "#000" }));
    return;
  }
  if (o.q.length === 1) {
    const r = box(o.q[0], o.label, o.par, o.kind === "noise" ? "qx-gate qx-noisebox" : "qx-gate");
    g.addEventListener("pointerdown", (ev) => startDrag(root, ev, { kind: "move", idx: o.i, label: o.label, fill: c.fill, text: c.text }));
    g.addEventListener("dblclick", () => focusParam(root, o.i));
    return;
  }
  if (o.kind === "noise") {                               // multi-qubit noise channel: one tall box
    const q0 = Math.min(...o.q), q1 = Math.max(...o.q);
    const r = sv("rect", { x: x - 21, y: cy(q0) - 20, width: 42, height: cy(q1) - cy(q0) + 40, rx: 4, class: "qx-gate qx-noisebox" }, g);
    r.style.fill = c.fill;
    const t = sv("text", { x: x, y: (cy(q0) + cy(q1)) / 2 + 5, class: "qx-glabel" }, g); t.textContent = o.label; t.style.fill = c.text;
    g.addEventListener("pointerdown", (ev) => startDrag(root, ev, { kind: "move", idx: o.i, label: o.label, fill: c.fill, text: c.text }));
    return;
  }
  const [a, b] = o.q;
  sv("line", { x1: x, x2: x, y1: cy(a), y2: cy(b), class: "qx-link" }, g);
  if (o.name === "cx") {
    sv("circle", { cx: x, cy: cy(a), r: 6, class: "qx-ctrl" }, g);
    sv("circle", { cx: x, cy: cy(b), r: 13, class: "qx-targ" }, g);
    sv("line", { x1: x - 13, x2: x + 13, y1: cy(b), y2: cy(b), class: "qx-targ-l" }, g);
    sv("line", { x1: x, x2: x, y1: cy(b) - 13, y2: cy(b) + 13, class: "qx-targ-l" }, g);
  } else if (o.name === "cz") {
    sv("circle", { cx: x, cy: cy(a), r: 6, class: "qx-ctrl" }, g);
    sv("circle", { cx: x, cy: cy(b), r: 6, class: "qx-ctrl" }, g);
  } else if (o.name === "swap") {
    for (const q of [a, b]) {
      sv("line", { x1: x - 9, x2: x + 9, y1: cy(q) - 9, y2: cy(q) + 9, class: "qx-swap" }, g);
      sv("line", { x1: x - 9, x2: x + 9, y1: cy(q) + 9, y2: cy(q) - 9, class: "qx-swap" }, g);
    }
  } else if (o.name === "cp") {
    sv("circle", { cx: x, cy: cy(a), r: 6, class: "qx-ctrl" }, g);
    box(b, "P", o.par);
  } else {                                                 // rzz, ecr: boxes on both ends
    box(a, o.label, o.par); box(b, o.label, "");
  }
  // drag the whole gate from its first end, retarget from the second end
  const hitA = sv("rect", { x: x - 22, y: cy(a) - 22, width: 44, height: 44, class: "qx-hit" }, g);
  hitA.addEventListener("pointerdown", (ev) => startDrag(root, ev, { kind: "move", idx: o.i, label: o.label, fill: c.fill, text: c.text }));
  // the move area first, the retarget handle last: SVG paints (and hit-tests) later siblings on top
  const hitB2 = sv("rect", { x: x - 22, y: cy(b) - 22, width: 44, height: 44, class: "qx-hit" }, g);
  hitB2.addEventListener("pointerdown", (ev) => startDrag(root, ev, { kind: "move", idx: o.i, label: o.label, fill: c.fill, text: c.text }));
  const hitB = sv("circle", { cx: x + 17, cy: cy(b) + 15, r: 7, class: "qx-handle" }, g);
  const tb = sv("title", {}, hitB); tb.textContent = "drag to another wire";
  hitB.addEventListener("pointerdown", (ev) => { ev.stopPropagation(); startDrag(root, ev, { kind: "retarget", idx: o.i, label: "⟶", fill: "#6929C4", text: "#fff" }); });
  g.addEventListener("dblclick", () => focusParam(root, o.i));
}

function focusParam(root, i) {
  root.__state.sel = i; render(root);
  const inp = root.querySelector(".qx-param");
  if (inp) { inp.focus(); inp.select(); }
}

function drawInspector(root) {
  const data = root.__data, st = root.__state;
  const box = el("div", { cls: "qx-inspector" }, root);
  if (st.sel === null || st.sel >= data.ops.length) {
    el("span", { cls: "qx-muted", text: "Select an operation to edit its parameter, flip, duplicate or delete it." }, box);
    return;
  }
  const o = data.ops[st.sel];
  el("b", { text: (o.label || o.name) + "  " }, box);
  el("span", { cls: "qx-muted", text: "on " + o.q.map((q) => "q" + q).join(", ") + "  " }, box);
  if (o.npar) {
    const inp = el("input", { cls: "qx-param", type: "text", value: o.ptext, title: "a number or an expression such as pi/4, 2*pi/3, sqrt(2)/2", maxlength: "200" }, box);
    const apply = el("button", { cls: "qx-btn", type: "button", text: "Set" }, box);
    const go = () => { st.keepSel = st.sel; root.__send({ type: "param", idx: st.sel, text: String(inp.value).slice(0, 200) }); };
    apply.addEventListener("click", go);
    inp.addEventListener("keydown", (ev) => { if (ev.key === "Enter") { go(); ev.preventDefault(); } ev.stopPropagation(); });
  }
  if (o.q.length === 2) {
    const f = el("button", { cls: "qx-btn", type: "button", text: "⇅ Flip" }, box);
    f.addEventListener("click", () => { st.keepSel = st.sel; root.__send({ type: "flip", idx: st.sel }); });
  }
  const d = el("button", { cls: "qx-btn", type: "button", text: "⧉ Duplicate" }, box);
  d.addEventListener("click", () => root.__send({ type: "duplicate", idx: st.sel }));
  const del = el("button", { cls: "qx-btn qx-danger", type: "button", text: "✕ Delete" }, box);
  del.addEventListener("click", () => { root.__send({ type: "delete", idx: st.sel }); st.sel = null; });
}

// ---------------- dragging
function startDrag(root, ev, info) {
  if (ev.button !== undefined && ev.button !== 0) return;
  ev.preventDefault(); ev.stopPropagation();
  root.focus({ preventScroll: true });
  const ghost = el("div", { cls: "qx-ghost" }, document.body);
  Object.assign(ghost.style, { position: "fixed", zIndex: "100000", pointerEvents: "none", padding: "8px 11px",
    borderRadius: "5px", font: "600 14px 'IBM Plex Sans', 'Source Sans Pro', sans-serif", opacity: "0.92",
    boxShadow: "0 4px 14px rgba(0,0,0,0.25)", minWidth: "22px", textAlign: "center" });
  ghost.textContent = info.label || "";
  ghost.style.background = info.fill || "#002D9C"; ghost.style.color = info.text || "#fff";
  ghost.style.left = ev.clientX + 8 + "px"; ghost.style.top = ev.clientY + 8 + "px";
  root.__state.drag = { ...info, x0: ev.clientX, y0: ev.clientY, moved: false, ghost, target: null };
  try { root.setPointerCapture(ev.pointerId); } catch (e) { /* ignore */ }
}

function cellAt(root, ev) {
  const g = root.__geom; if (!g) return null;
  const r = g.svg.getBoundingClientRect();
  const x = ev.clientX - r.left, y = ev.clientY - r.top;
  const col = Math.floor((x - LEFT) / CELL), q = Math.floor((y - TOP) / ROW);
  if (x < LEFT - 30 || q < 0 || q >= g.n || y < 0) return null;
  return { q, col: Math.max(0, col) };
}

function overTrash(root, ev) {
  const t = root.__trash; if (!t) return false;
  const r = t.getBoundingClientRect();
  return ev.clientX >= r.left && ev.clientX <= r.right && ev.clientY >= r.top && ev.clientY <= r.bottom;
}

function onMove(root, ev) {
  const d = root.__state.drag; if (!d) return;
  if (Math.abs(ev.clientX - d.x0) + Math.abs(ev.clientY - d.y0) > 4) d.moved = true;
  d.ghost.style.left = ev.clientX + 8 + "px"; d.ghost.style.top = ev.clientY + 8 + "px";
  const cell = cellAt(root, ev);
  const hl = root.__hl;
  const trash = overTrash(root, ev);
  root.__trash.classList.toggle("qx-trash-on", trash && d.kind !== "new");
  if (cell && !trash) {
    hl.setAttribute("x", LEFT + cell.col * CELL + 2); hl.setAttribute("y", TOP + cell.q * ROW + 3);
    hl.setAttribute("visibility", "visible");
  } else hl.setAttribute("visibility", "hidden");
  d.target = cell; d.trash = trash;
}

function endDrag(root) {
  const d = root.__state.drag; if (!d) return;
  if (d.ghost && d.ghost.parentNode) d.ghost.parentNode.removeChild(d.ghost);
  if (root.__hl) root.__hl.setAttribute("visibility", "hidden");
  if (root.__trash) root.__trash.classList.remove("qx-trash-on");
  root.__state.drag = null;
}

function onUp(root, ev) {
  const d = root.__state.drag; if (!d) return;
  const st = root.__state;
  const target = cellAt(root, ev), trash = overTrash(root, ev);
  endDrag(root);
  if (!d.moved) {                                     // a click
    if (d.kind === "move" || d.kind === "retarget") { st.sel = d.idx; render(root); }
    else if (d.kind === "new") {                        // click on a tile: append on the selected op's qubit
      const q = st.sel !== null && root.__data.ops[st.sel] ? root.__data.ops[st.sel].q[0] : 0;
      root.__send({ type: "add", name: d.name, q: q, col: 1000000 });
    }
    return;
  }
  if (trash && d.kind !== "new") { root.__send({ type: "delete", idx: d.idx }); st.sel = null; return; }
  if (!target) return;
  if (d.kind === "new") root.__send({ type: "add", name: d.name, q: target.q, col: target.col });
  else if (d.kind === "move") { root.__send({ type: "move", idx: d.idx, q: target.q, col: target.col }); st.sel = null; }
  else if (d.kind === "retarget") { st.keepSel = d.idx; root.__send({ type: "retarget", idx: d.idx, which: 1, q: target.q }); }
}
