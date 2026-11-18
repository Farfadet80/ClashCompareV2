/**
 * Stub client profil API Clash of Clans (tag #).
 * Ne remplace PAS l'export JSON pour les bâtiments / murs / pièges.
 * Sans CLASHCOMPARE_API configuré → Non disponible (pas d'erreur bloquante).
 */
(function (global) {
  "use strict";

  function normalizeTag(tag) {
    let s = String(tag || "").trim().toUpperCase().replace(/\s+/g, "");
    if (s && !s.startsWith("#")) s = "#" + s;
    return s;
  }

  /**
   * @param {string} tag
   * @param {{baseUrl?: string}|null} options
   * @returns {Promise<{ok:boolean, status:string, profile?:object|null, error?:string}>}
   */
  async function fetchPlayerProfile(tag, options) {
    const base =
      (options && options.baseUrl) ||
      (typeof global.CLASHCOMPARE_API === "string" ? global.CLASHCOMPARE_API : "") ||
      "";
    const normalized = normalizeTag(tag);
    if (!normalized || normalized === "#") {
      return { ok: false, status: "Non disponible", error: "Tag joueur manquant.", profile: null };
    }
    if (!base.trim()) {
      return {
        ok: false,
        status: "Non disponible",
        error:
          "API CoC non configurée (CLASHCOMPARE_API vide). L'export JSON suffit pour l'inventaire.",
        profile: null,
      };
    }

    const encoded = encodeURIComponent(normalized.replace(/^#/, ""));
    const url = base.replace(/\/$/, "") + "/players/%23" + encoded;
    try {
      const response = await fetch(url);
      if (!response.ok) {
        return {
          ok: false,
          status: "Non disponible",
          error: `API profil HTTP ${response.status}`,
          profile: null,
        };
      }
      const profile = await response.json();
      return { ok: true, status: "ok", profile, error: null };
    } catch (err) {
      return {
        ok: false,
        status: "Non disponible",
        error: String(err && err.message ? err.message : err),
        profile: null,
      };
    }
  }

  function summarizeProfile(profile) {
    if (!profile || typeof profile !== "object") return null;
    return {
      name: profile.name || null,
      tag: profile.tag || null,
      townHallLevel: profile.townHallLevel ?? null,
      trophies: profile.trophies ?? null,
      expLevel: profile.expLevel ?? null,
      league: profile.league?.name || null,
      heroes: Array.isArray(profile.heroes) ? profile.heroes : [],
      troops: Array.isArray(profile.troops) ? profile.troops : [],
      spells: Array.isArray(profile.spells) ? profile.spells : [],
    };
  }

  global.ClashCompareCocApi = {
    fetchPlayerProfile,
    summarizeProfile,
    normalizeTag,
  };
})(typeof window !== "undefined" ? window : globalThis);
