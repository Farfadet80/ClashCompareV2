/**
 * % maxage HDV + restes à améliorer — inventaire JSON uniquement (pas YOLO).
 * Données absentes / niveaux inconnus → Non disponible (jamais 0 % inventé).
 */
(function (global) {
  "use strict";

  const CATEGORY_GROUPS = {
    defense: "defenses",
    "merged-defense": "defenses",
    "town-hall-guardian": "defenses",
    "crafted-defense": "defenses",
    wall: "walls",
    resource: "ressources",
    army: "armee",
    hero: "armee",
    core: "core",
    utility: "utilitaires",
    trap: "pieges",
  };

  const GROUP_LABELS = {
    global: "Global",
    defenses: "Défenses",
    walls: "Murs",
    ressources: "Ressources",
    armee: "Armée",
    pieges: "Pièges",
    utilitaires: "Utilitaires",
    core: "Cœur",
  };

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  }

  function catalogById(catalog) {
    const map = {};
    const buildings = asObject(catalog)?.buildings;
    if (!Array.isArray(buildings)) return map;
    for (const row of buildings) {
      if (row && row.id) map[row.id] = row;
    }
    return map;
  }

  function capsForTh(thCaps, townHallLevel) {
    const byTh = asObject(thCaps)?.by_th || {};
    return asObject(byTh[String(townHallLevel)]) || null;
  }

  function maxForBuilding(buildingId, caps, catalogMap) {
    if (!caps || caps[buildingId] == null) return null;
    const max = Number(caps[buildingId]);
    if (!Number.isFinite(max) || max <= 0) return null;
    // Crafted defense : niveaux variables / modules — exclus du score v1
    const meta = catalogMap[buildingId];
    if (meta?.category === "crafted-defense") return null;
    return max;
  }

  function groupFor(buildingId, catalogMap) {
    const cat = catalogMap[buildingId]?.category || "defense";
    return CATEGORY_GROUPS[cat] || "defenses";
  }

  /**
   * @param {object} inventory merge inventaire { id: { total, levels } }
   * @param {number|null} townHallLevel
   * @param {object} thCaps data/th-caps.json
   * @param {object} catalog data/buildings.json
   */
  function computeMaxage(inventory, townHallLevel, thCaps, catalog) {
    const inv = asObject(inventory) || {};
    const catalogMap = catalogById(catalog);

    if (townHallLevel == null || !Number.isFinite(Number(townHallLevel))) {
      return {
        ok: false,
        status: "Non disponible",
        reason: "HDV inconnu (importe un export JSON).",
        townHallLevel: null,
        percent: null,
        byGroup: {},
        remaining: [],
        scoredUnits: 0,
        skippedUnknownLevels: 0,
        skippedNoCap: 0,
      };
    }

    const th = Math.trunc(Number(townHallLevel));
    const caps = capsForTh(thCaps, th);
    if (!caps) {
      return {
        ok: false,
        status: "Non disponible",
        reason: `Plafonds HDV ${th} absents du catalogue (supportés : ${(asObject(thCaps)?.supported_town_halls || []).join(", ") || "—"}).`,
        townHallLevel: th,
        percent: null,
        byGroup: {},
        remaining: [],
        scoredUnits: 0,
        skippedUnknownLevels: 0,
        skippedNoCap: 0,
      };
    }

    const groups = {};
    const remaining = [];
    let scoredUnits = 0;
    let skippedUnknownLevels = 0;
    let skippedNoCap = 0;
    let sumRatio = 0;
    let sumWeight = 0;

    function touchGroup(key) {
      if (!groups[key]) {
        groups[key] = { sumRatio: 0, weight: 0, remainingCount: 0 };
      }
      return groups[key];
    }

    for (const [buildingId, slot] of Object.entries(inv)) {
      const maxLvl = maxForBuilding(buildingId, caps, catalogMap);
      if (maxLvl == null) {
        skippedNoCap += slot?.total || 0;
        continue;
      }
      const levels = asObject(slot?.levels) || {};
      const groupKey = groupFor(buildingId, catalogMap);
      const g = touchGroup(groupKey);
      const name = catalogMap[buildingId]?.name || buildingId;

      for (const [levelKey, countRaw] of Object.entries(levels)) {
        const count = Number(countRaw) || 0;
        if (count <= 0) continue;
        if (levelKey === "inconnu" || levelKey === "" || Number.isNaN(Number(levelKey))) {
          skippedUnknownLevels += count;
          continue;
        }
        const level = Math.trunc(Number(levelKey));
        // Merged defenses can appear at lvl 0 in export before upgrade — treat as level 0 / non max
        const clamped = Math.max(0, level);
        const ratio = Math.min(clamped, maxLvl) / maxLvl;
        sumRatio += ratio * count;
        sumWeight += count;
        g.sumRatio += ratio * count;
        g.weight += count;
        scoredUnits += count;

        if (clamped < maxLvl) {
          const deficit = maxLvl - clamped;
          remaining.push({
            id: buildingId,
            name,
            group: groupKey,
            level: clamped,
            max: maxLvl,
            count,
            levelsShort: deficit,
          });
          g.remainingCount += count;
        }
      }
    }

    if (sumWeight === 0) {
      return {
        ok: false,
        status: "Non disponible",
        reason:
          skippedUnknownLevels > 0
            ? "Niveaux inconnus dans l'inventaire — impossible de scorer."
            : "Aucun bâtiment scorablе pour ce HDV.",
        townHallLevel: th,
        percent: null,
        byGroup: {},
        remaining: [],
        scoredUnits: 0,
        skippedUnknownLevels,
        skippedNoCap,
      };
    }

    const percent = Math.round((1000 * sumRatio) / sumWeight) / 10;
    const byGroup = {};
    for (const [key, g] of Object.entries(groups)) {
      if (g.weight <= 0) continue;
      byGroup[key] = {
        label: GROUP_LABELS[key] || key,
        percent: Math.round((1000 * g.sumRatio) / g.weight) / 10,
        units: g.weight,
        remainingCount: g.remainingCount,
      };
    }

    remaining.sort((a, b) => {
      const gap = b.levelsShort * b.count - a.levelsShort * a.count;
      if (gap !== 0) return gap;
      return a.name.localeCompare(b.name, "fr");
    });

    return {
      ok: true,
      status: "ok",
      reason: null,
      townHallLevel: th,
      percent,
      byGroup,
      remaining,
      scoredUnits,
      skippedUnknownLevels,
      skippedNoCap,
    };
  }

  function formatPercent(value) {
    if (value == null || Number.isNaN(Number(value))) return "Non disponible";
    const n = Number(value);
    return Number.isInteger(n) ? `${n} %` : `${n.toFixed(1)} %`;
  }

  global.ClashCompareMaxage = {
    computeMaxage,
    formatPercent,
    GROUP_LABELS,
  };
})(typeof window !== "undefined" ? window : globalThis);
