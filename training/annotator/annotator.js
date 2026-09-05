(() => {
  "use strict";

  const $ = (selector) => document.querySelector(selector);
  const canvas = $("#canvas");
  const ctx = canvas.getContext("2d");
  const state = {
    classes: [],
    image: null,
    imageName: "",
    boxes: [],
    selected: -1,
    expected: {},
    zoom: 1,
    action: null,
  };
  const sessionKey = "clashcompare-annotator-v1";

  function setStatus(message, ok = false) {
    const element = $("#status");
    element.textContent = message;
    element.style.color = ok ? "var(--green)" : "var(--amber)";
  }

  function classIndex(id) {
    return state.classes.findIndex((item) => item.id === id);
  }

  function selectedClassId() {
    return Number($("#class-select").value || 0);
  }

  async function readJson(file) {
    return JSON.parse(await file.text());
  }

  function download(name, text, type = "application/json") {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const link = document.createElement("a");
    link.href = url;
    link.download = name;
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function normalizeBox(box) {
    const x1 = Math.min(box.x1, box.x2);
    const y1 = Math.min(box.y1, box.y2);
    const x2 = Math.max(box.x1, box.x2);
    const y2 = Math.max(box.y1, box.y2);
    return {
      ...box,
      x1: clamp(x1, 0, state.image.naturalWidth),
      y1: clamp(y1, 0, state.image.naturalHeight),
      x2: clamp(x2, 0, state.image.naturalWidth),
      y2: clamp(y2, 0, state.image.naturalHeight),
    };
  }

  function point(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * canvas.width / rect.width / state.zoom,
      y: (event.clientY - rect.top) * canvas.height / rect.height / state.zoom,
    };
  }

  function hitTest(x, y) {
    return state.boxes
      .map((box, index) => ({ box, index, area: (box.x2 - box.x1) * (box.y2 - box.y1) }))
      .filter(({ box }) => x >= box.x1 && x <= box.x2 && y >= box.y1 && y <= box.y2)
      .sort((a, b) => a.area - b.area)[0]?.index ?? -1;
  }

  function draw() {
    if (!state.image) return;
    const width = Math.round(state.image.naturalWidth * state.zoom);
    const height = Math.round(state.image.naturalHeight * state.zoom);
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.drawImage(state.image, 0, 0, width, height);
    ctx.font = `${Math.max(11, 12 * state.zoom)}px system-ui`;
    ctx.lineWidth = Math.max(2, 2 * state.zoom);

    state.boxes.forEach((box, index) => {
      const selected = index === state.selected;
      const pending = box.status !== "accepted";
      ctx.strokeStyle = selected ? "#ffffff" : pending ? "#ffbd59" : "#63da78";
      ctx.fillStyle = ctx.strokeStyle;
      const x = box.x1 * state.zoom;
      const y = box.y1 * state.zoom;
      const w = (box.x2 - box.x1) * state.zoom;
      const h = (box.y2 - box.y1) * state.zoom;
      ctx.strokeRect(x, y, w, h);
      const label = `${state.classes[box.classId]?.id || "classe?"}${pending ? " ?" : ""}`;
      const textWidth = ctx.measureText(label).width + 8;
      ctx.fillRect(x, Math.max(0, y - 17 * state.zoom), textWidth, 17 * state.zoom);
      ctx.fillStyle = "#071008";
      ctx.fillText(label, x + 4, Math.max(11, y - 4 * state.zoom));
    });
    updateSummary();
  }

  function acceptedCounts() {
    const counts = {};
    for (const box of state.boxes.filter((item) => item.status === "accepted")) {
      const id = state.classes[box.classId]?.id;
      if (id) counts[id] = (counts[id] || 0) + 1;
    }
    return counts;
  }

  function updateSummary() {
    const accepted = state.boxes.filter((box) => box.status === "accepted").length;
    const pending = state.boxes.length - accepted;
    $("#accepted-count").textContent = accepted;
    $("#pending-count").textContent = pending;
    const box = state.boxes[state.selected];
    $("#selected-label").textContent = box
      ? `${state.classes[box.classId]?.name || "Classe inconnue"} · ${box.status}`
      : "—";

    const actual = acceptedCounts();
    const entries = Object.entries(state.expected);
    $("#count-table").innerHTML = entries.length
      ? entries.map(([id, expected]) => {
          const found = actual[id] || 0;
          return `<div class="count-row ${found === expected ? "" : "mismatch"}">` +
            `<span>${id}</span><strong>${found}</strong><span>/ ${expected}</span></div>`;
        }).join("")
      : '<p class="hint">Charge un résumé d’export pour afficher les contrôles.</p>';
  }

  function snapshot() {
    return {
      version: 1,
      image: state.image ? {
        name: state.imageName,
        width: state.image.naturalWidth,
        height: state.image.naturalHeight,
      } : null,
      metadata: {
        village_group: $("#village-group").value.trim(),
        source: $("#source").value.trim(),
        license: $("#license").value.trim(),
        exhaustive: $("#exhaustive").checked,
      },
      expected_counts: state.expected,
      boxes: state.boxes.map((box) => ({
        class_id: box.classId,
        class_name: state.classes[box.classId]?.id || null,
        x1: Math.round(box.x1 * 100) / 100,
        y1: Math.round(box.y1 * 100) / 100,
        x2: Math.round(box.x2 * 100) / 100,
        y2: Math.round(box.y2 * 100) / 100,
        status: box.status,
      })),
    };
  }

  function autosave() {
    try {
      localStorage.setItem(sessionKey, JSON.stringify(snapshot()));
    } catch {
      setStatus("Autosauvegarde locale impossible.");
    }
  }

  function applySession(payload) {
    const metadata = payload.metadata || {};
    $("#village-group").value = metadata.village_group || "";
    $("#source").value = metadata.source || "";
    $("#license").value = metadata.license || "";
    $("#exhaustive").checked = metadata.exhaustive === true;
    state.expected = payload.expected_counts || {};
    state.boxes = (payload.boxes || []).map((box) => ({
      classId: Number.isInteger(box.class_id) ? box.class_id : classIndex(box.class_name),
      x1: Number(box.x1),
      y1: Number(box.y1),
      x2: Number(box.x2),
      y2: Number(box.y2),
      status: box.status === "accepted" ? "accepted" : "pending",
    })).filter((box) => box.classId >= 0 && [box.x1, box.y1, box.x2, box.y2].every(Number.isFinite));
    state.selected = -1;
    draw();
    setStatus(`${state.boxes.length} boîte(s) chargée(s). Vérifie chaque suggestion.`, true);
  }

  async function loadImage(file) {
    const url = URL.createObjectURL(file);
    const image = new Image();
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = reject;
      image.src = url;
    });
    state.image = image;
    state.imageName = file.name;
    state.boxes = [];
    state.selected = -1;
    $("#empty-state").hidden = true;
    const saved = JSON.parse(localStorage.getItem(sessionKey) || "null");
    if (saved?.image?.name === file.name &&
        saved.image.width === image.naturalWidth &&
        saved.image.height === image.naturalHeight) {
      applySession(saved);
      setStatus("Image chargée et session locale restaurée.", true);
    } else {
      draw();
      setStatus(`${file.name} · ${image.naturalWidth}×${image.naturalHeight}`, true);
    }
  }

  function validateForExport() {
    if (!state.image) return "Capture manquante.";
    if (!$("#village-group").value.trim()) return "Groupe village manquant.";
    if (!$("#source").value.trim()) return "Source/auteur manquant.";
    if (!$("#license").value.trim()) return "Licence ou consentement manquant.";
    if (!$("#exhaustive").checked) return "La validation exhaustive n’est pas cochée.";
    if (state.boxes.some((box) => box.status !== "accepted")) {
      return "Des suggestions sont encore en attente.";
    }
    if (!state.boxes.length) return "Aucune annotation.";
    for (const box of state.boxes) {
      if (box.x2 - box.x1 < 2 || box.y2 - box.y1 < 2) return "Une boîte est trop petite.";
    }
    const actual = acceptedCounts();
    const mismatches = Object.entries(state.expected)
      .filter(([id, expected]) => classIndex(id) >= 0 && (actual[id] || 0) !== expected);
    if (mismatches.length) {
      return `Quantités incompatibles avec l’export : ${mismatches.slice(0, 4).map(([id]) => id).join(", ")}.`;
    }
    return null;
  }

  canvas.addEventListener("pointerdown", (event) => {
    if (!state.image) return;
    canvas.setPointerCapture(event.pointerId);
    const start = point(event);
    const hit = hitTest(start.x, start.y);
    if (hit >= 0) {
      state.selected = hit;
      const box = state.boxes[hit];
      state.action = { type: "move", start, original: { ...box } };
      $("#class-select").value = box.classId;
    } else {
      const box = {
        classId: selectedClassId(),
        x1: start.x, y1: start.y, x2: start.x, y2: start.y,
        status: "accepted",
      };
      state.boxes.push(box);
      state.selected = state.boxes.length - 1;
      state.action = { type: "draw", start };
    }
    draw();
  });

  canvas.addEventListener("pointermove", (event) => {
    if (!state.action || state.selected < 0) return;
    const current = point(event);
    const box = state.boxes[state.selected];
    if (state.action.type === "draw") {
      box.x2 = current.x;
      box.y2 = current.y;
    } else {
      const dx = current.x - state.action.start.x;
      const dy = current.y - state.action.start.y;
      const width = state.action.original.x2 - state.action.original.x1;
      const height = state.action.original.y2 - state.action.original.y1;
      box.x1 = clamp(state.action.original.x1 + dx, 0, state.image.naturalWidth - width);
      box.y1 = clamp(state.action.original.y1 + dy, 0, state.image.naturalHeight - height);
      box.x2 = box.x1 + width;
      box.y2 = box.y1 + height;
    }
    draw();
  });

  canvas.addEventListener("pointerup", () => {
    if (!state.action || state.selected < 0) return;
    state.boxes[state.selected] = normalizeBox(state.boxes[state.selected]);
    const box = state.boxes[state.selected];
    if (box.x2 - box.x1 < 2 || box.y2 - box.y1 < 2) {
      state.boxes.splice(state.selected, 1);
      state.selected = -1;
    }
    state.action = null;
    autosave();
    draw();
  });

  $("#class-select").addEventListener("change", () => {
    if (state.selected >= 0) {
      state.boxes[state.selected].classId = selectedClassId();
      autosave();
      draw();
    }
  });
  $("#zoom").addEventListener("input", (event) => {
    state.zoom = Number(event.target.value) / 100;
    $("#zoom-value").textContent = `${event.target.value} %`;
    draw();
  });
  $("#accept").addEventListener("click", () => {
    if (state.selected >= 0) {
      state.boxes[state.selected].status = "accepted";
      autosave();
      draw();
    }
  });
  function deleteSelected() {
    if (state.selected < 0) return;
    state.boxes.splice(state.selected, 1);
    state.selected = -1;
    autosave();
    draw();
  }
  $("#delete").addEventListener("click", deleteSelected);
  document.addEventListener("keydown", (event) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (event.key === "Delete" || event.key === "Backspace") deleteSelected();
    if (event.key.toLowerCase() === "a" && state.selected >= 0) {
      state.boxes[state.selected].status = "accepted";
      autosave();
      draw();
    }
    if (event.key === "Escape") {
      state.selected = -1;
      draw();
    }
  });

  $("#image-file").addEventListener("change", (event) => event.target.files[0] && loadImage(event.target.files[0]));
  $("#session-file").addEventListener("change", async (event) => {
    if (event.target.files[0]) applySession(await readJson(event.target.files[0]));
  });
  $("#candidates-file").addEventListener("change", async (event) => {
    if (!state.image) return setStatus("Charge d’abord la capture correspondante.");
    const payload = await readJson(event.target.files[0]);
    state.boxes = (payload.detections || []).map((item) => {
      const [x1, y1, x2, y2] = item.box;
      return { classId: classIndex(item.building), x1, y1, x2, y2, status: "pending" };
    }).filter((box) => box.classId >= 0);
    state.selected = -1;
    autosave();
    draw();
    setStatus(`${state.boxes.length} suggestions importées en attente.`);
  });
  $("#ground-file").addEventListener("change", async (event) => {
    const payload = await readJson(event.target.files[0]);
    const inventory = payload.inventory || {};
    state.expected = Object.fromEntries(
      Object.entries(inventory).map(([id, slot]) => [id, Number(slot.total || 0)])
    );
    updateSummary();
    autosave();
    setStatus(`${Object.keys(state.expected).length} quantités de référence chargées.`, true);
  });
  ["village-group", "source", "license", "exhaustive"].forEach((id) => {
    $(`#${id}`).addEventListener("change", autosave);
  });

  $("#save-session").addEventListener("click", () => {
    const base = (state.imageName || "annotation").replace(/\.[^.]+$/, "");
    download(`${base}.annotation-session.json`, JSON.stringify(snapshot(), null, 2));
    setStatus("Session JSON exportée.", true);
  });
  $("#export-yolo").addEventListener("click", () => {
    const error = validateForExport();
    if (error) return setStatus(error);
    const width = state.image.naturalWidth;
    const height = state.image.naturalHeight;
    const lines = state.boxes.map((box) => {
      const x = ((box.x1 + box.x2) / 2) / width;
      const y = ((box.y1 + box.y2) / 2) / height;
      const w = (box.x2 - box.x1) / width;
      const h = (box.y2 - box.y1) / height;
      return `${box.classId} ${x.toFixed(6)} ${y.toFixed(6)} ${w.toFixed(6)} ${h.toFixed(6)}`;
    });
    const base = state.imageName.replace(/\.[^.]+$/, "");
    download(`${base}.txt`, `${lines.join("\n")}\n`, "text/plain");
    setStatus(`${lines.length} labels YOLO exportés.`, true);
  });

  fetch("../classes.json")
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.json();
    })
    .then((payload) => {
      state.classes = payload.detector_classes || [];
      $("#class-select").innerHTML = state.classes.map(
        (item, index) => `<option value="${index}">${index} — ${item.name} (${item.id})</option>`
      ).join("");
      updateSummary();
    })
    .catch((error) => setStatus(`Chargement classes impossible : ${error.message}`));
})();
