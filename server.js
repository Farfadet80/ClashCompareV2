import express from "express";

const app = express();
const PORT = process.env.PORT || 3000;

app.use((req, res, next) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  next();
});

app.get("/", (req, res) => {
  res.json({
    ok: true,
    service: "ClashCompare API"
  });
});

app.get("/player", async (req, res) => {
  const tag = (req.query.tag || "").trim().toUpperCase();

  if (!tag.startsWith("#")) {
    return res.status(400).json({ message: "Tag joueur invalide." });
  }

  if (!process.env.COC_API_TOKEN) {
    return res.status(500).json({
      message: "Clé API Clash of Clans non configurée."
    });
  }

  try {
    const url =
      "https://api.clashofclans.com/v1/players/" +
      encodeURIComponent(tag);

    const response = await fetch(url, {
      headers: {
        Authorization: `Bearer ${process.env.COC_API_TOKEN}`,
        Accept: "application/json"
      }
    });

    const data = await response.json();

    res.status(response.status).json(data);

  } catch (error) {
    res.status(500).json({
      message: "Erreur de connexion à Clash of Clans."
    });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`ClashCompare API démarrée sur le port ${PORT}`);
});
