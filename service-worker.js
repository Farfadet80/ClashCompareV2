const CACHE="clashcompare-v3-11-export-52";
const ASSETS=[
  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./village-export.js",
  "./config.js",
  "./manifest.json",
  "./data/buildings.json",
  "./data/coc-export-mapping.json"
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
