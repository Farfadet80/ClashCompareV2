const CACHE="clashcompare-v3-12-th19-maxage";
const ASSETS=[
  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./village-export.js",
  "./maxage.js",
  "./coc-api.js",
  "./config.js",
  "./manifest.json",
  "./icon-192.png",
  "./icon-512.png",
  "./apple-touch-icon.png",
  "./data/buildings.json",
  "./data/coc-export-mapping.json",
  "./data/th-caps.json"
];
self.addEventListener("install",e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS))));
self.addEventListener("activate",e=>e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k))))));
self.addEventListener("fetch",e=>{
  const url=new URL(e.request.url);
  if(e.request.method!=="GET" || url.pathname.startsWith("/api/")){
    e.respondWith(fetch(e.request));
    return;
  }
  e.respondWith(caches.match(e.request).then(r=>r||fetch(e.request)));
});
