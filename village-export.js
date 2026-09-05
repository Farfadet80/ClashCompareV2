/**
 * Parseur local de l'export village Clash of Clans (Settings → More Settings → Export).
 * Ne fait aucune estimation : IDs inconnus → Non détecté / unresolved.
 */
(function (global) {
  "use strict";

  const DEFAULT_SECTIONS = ["buildings", "traps"];

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  }

  function asArray(value) {
    return Array.isArray(value) ? value : [];
  }

  function toInt(value) {
    if (typeof value === "number" && Number.isFinite(value)) return Math.trunc(value);
    if (typeof value === "string" && value.trim() !== "" && !Number.isNaN(Number(value))) {
      return Math.trunc(Number(value));
    }
    return null;
  }

  function emptySlot() {
    return { total: 0, levels: {} };
  }

  function addCount(inventory, buildingId, levelKey, count) {
    if (!inventory[buildingId]) inventory[buildingId] = emptySlot();
    const slot = inventory[buildingId];
    slot.total += count;
    slot.levels[levelKey] = (slot.levels[levelKey] || 0) + count;
  }

  /**
   * @param {string|object} raw
   * @param {{dataIdToBuilding?: Record<string,string>, homeSections?: string[]}|null} mapping
   */
  function parseVillageExport(raw, mapping) {
    const map = asObject(mapping) || {};
    const dataIdToBuilding = asObject(map.dataIdToBuilding) || {};
    const sections = asArray(map.homeSections).length
      ? asArray(map.homeSections)
      : DEFAULT_SECTIONS;

    let text = "";
    let root;
    if (typeof raw === "string") {
      text = raw.trim();
      if (!text) {
        return { ok: false, error: "JSON vide" };
      }
      try {
        root = JSON.parse(text);
      } catch (err) {
        return { ok: false, error: "JSON invalide : " + (err && err.message ? err.message : err) };
      }
    } else {
      root = raw;
      try {
        text = JSON.stringify(raw);
      } catch {
        text = "";
      }
    }

    const data = asObject(root);
    if (!data) {
      return { ok: false, error: "L'export doit être un objet JSON" };
    }

    const inventory = {};
    const unresolved = [];
    let mappedItems = 0;
    let townHallLevel = null;

    for (const section of sections) {
      const items = asArray(data[section]);
      for (const item of items) {
        const row = asObject(item);
        if (!row) continue;
        const dataId = toInt(row.data);
        const level = toInt(row.lvl);
        const count = toInt(row.cnt);
        const qty = count === null ? 1 : count;
        if (dataId === null || qty <= 0) continue;

        const buildingId = dataIdToBuilding[String(dataId)];
        const levelKey = level === null ? "inconnu" : String(level);
        const upgrading = toInt(row.timer) !== null && toInt(row.timer) > 0;

        if (!buildingId) {
          unresolved.push({
            section,
            dataId,
            level: level === null ? null : level,
            count: qty,
            upgrading,
            status: "Non détecté",
          });
          continue;
        }

        addCount(inventory, buildingId, levelKey, qty);
        mappedItems += qty;
        if (buildingId === "town-hall" && level !== null) {
          townHallLevel = Math.max(townHallLevel || 0, level);
        }
      }
    }

    const tag = typeof data.tag === "string" ? data.tag : null;
    const buildingCount = Object.values(inventory).reduce((n, slot) => n + slot.total, 0);

    return {
      ok: true,
      source: "export",
      tag,
      townHallLevel,
      inventory,
      unresolved,
      mappedItems,
      buildingCount,
      sectionsUsed: sections,
      raw: text,
      importedAt: new Date().toISOString(),
      note:
        unresolved.length > 0
          ? `${buildingCount} bâtiments/pièges mappés ; ${unresolved.length} entrée(s) Non détectée(s).`
          : `${buildingCount} bâtiments/pièges importés depuis l'export officiel.`,
    };
  }

  /**
   * Export prioritaire ; YOLO complète uniquement les ids absents de l'export.
   * Jamais d'écrasement des niveaux export par YOLO.
   */
  function mergeInventories(exportInventory, yoloInventory) {
    const exp = asObject(exportInventory) || null;
    const yolo = asObject(yoloInventory) || {};
    if (!exp) {
      return {
        inventory: yolo,
        source: Object.keys(yolo).length ? "yolo" : "none",
        complementIds: [],
      };
    }
    const inventory = {};
    for (const [id, slot] of Object.entries(exp)) {
      inventory[id] = {
        total: slot.total || 0,
        levels: { ...(slot.levels || {}) },
      };
    }
    const complementIds = [];
    for (const [id, slot] of Object.entries(yolo)) {
      if (inventory[id]) continue;
      inventory[id] = {
        total: slot.total || 0,
        levels: { ...(slot.levels || {}) },
      };
      complementIds.push(id);
    }
    return {
      inventory,
      source: complementIds.length ? "mixed" : "export",
      complementIds,
    };
  }

  global.ClashCompareVillageExport = {
    parseVillageExport,
    mergeInventories,
  };
})(typeof window !== "undefined" ? window : globalThis);
