import http from "http";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = 8080;

const MIME_TYPES = {
  ".html": "text/html",
  ".js": "text/javascript",
  ".css": "text/css",
  ".json": "application/json",
  ".png": "image/png",
  ".jpg": "image/jpg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".svg": "image/svg+xml",
  ".webp": "image/webp",
  ".wasm": "application/wasm",
};

const server = http.createServer((req, res) => {
  console.log(`${req.method} ${req.url}`);

  // Default to the background blur demo
  let filePath = req.url === "/" ? "/examples/background-blur/background-blur-demo.html" : req.url;

  // Remove query strings or hashes if present
  filePath = filePath.split("?")[0].split("#")[0];

  // Ensure we don't try to access files outside the directory
  // Remove leading slash for path.join to behave consistently
  const safePath = path.normalize(filePath).replace(/^[\/\\]+/, "");
  let fullPath = path.join(__dirname, safePath);

  console.log(`Serving: ${fullPath}`);

  const extname = path.extname(fullPath);
  let contentType = MIME_TYPES[extname] || "application/octet-stream";

  fs.readFile(fullPath, (error, content) => {
    if (error) {
      if (error.code === "ENOENT") {
        res.writeHead(404);
        res.end("File not found");
      } else {
        res.writeHead(500);
        res.end(`Server error: ${error.code}`);
      }
    } else {
      res.writeHead(200, { "Content-Type": contentType });
      res.end(content, "utf-8");
    }
  });
});

server.listen(PORT, () => {
  console.log(`Server running at http://localhost:${PORT}/`);
  console.log(`Demo page: http://localhost:${PORT}/examples/background-blur/background-blur-demo.html`);
});
