from __future__ import annotations

import json
import mimetypes
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from app import Article, SelectionResult, load_inventory, prepare_selection, resolve_project_path
from llm_assistant import build_pharmacist_context
from vision import VisionResult, process_image


APP_NAME = "Gestion Stock Vision"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000


def article_to_payload(article: Article) -> dict[str, object]:
    return {
        "id_unique": article.id_unique,
        "nom": article.nom,
        "classe_therapeutique": article.classe_therapeutique,
        "emplacement_rayon": article.emplacement_rayon,
        "quantite_stock": article.quantite_stock,
        "prix": article.prix,
        "prix_formate": article.prix_formate,
        "image_path": article.image_path.as_posix(),
        "image_url": image_url_for(article.image_path),
    }


def vision_to_payload(result: VisionResult) -> dict[str, object]:
    return {
        "enabled": result.enabled,
        "ok": result.ok,
        "message": result.message,
        "detections": list(result.detections),
    }


def selection_to_payload(
    result: SelectionResult,
    vision_result: VisionResult | None = None,
) -> dict[str, object]:
    return {
        "ok": result.ok,
        "code": result.code,
        "messages": list(result.messages),
        "article": article_to_payload(result.article) if result.article else None,
        "vision": vision_to_payload(vision_result) if vision_result else None,
    }


def image_url_for(image_path: Path) -> str:
    return "/" + quote(image_path.as_posix(), safe="/")


def parse_quantity(value: object, default: int = 1) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default


def parse_request_payload(content_type: str, body: bytes) -> dict[str, object]:
    if "application/json" in content_type:
        try:
            data = json.loads(body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    parsed = parse_qs(body.decode("utf-8"), keep_blank_values=True)
    return {key: values[-1] for key, values in parsed.items()}


def run_selection(query: str, quantity: int) -> tuple[SelectionResult, VisionResult | None]:
    inventory = load_inventory()
    result = prepare_selection(inventory, query, quantity)
    vision_result = None

    if result.ok and result.article is not None:
        image_path = resolve_project_path(result.article.image_path)
        vision_result = process_image(image_path)

    return result, vision_result


def render_inventory(inventory: list[Article]) -> str:
    rows = []
    for article in inventory:
        stock_class = "stock-ok" if article.quantite_stock > 20 else "stock-low"
        form_id = f"select-{article.id_unique}"
        rows.append(
            "<tr>"
            f"<td><strong>{escape(article.id_unique)}</strong></td>"
            f"<td>{escape(article.nom)}</td>"
            f"<td>{escape(article.classe_therapeutique)}</td>"
            f"<td>{escape(article.emplacement_rayon)}</td>"
            f'<td class="{stock_class}">{article.quantite_stock}</td>'
            f"<td>{escape(article.prix_formate)}</td>"
            "<td>"
            f'<button type="submit" name="query" value="{escape(article.id_unique)}" '
            f'form="{escape(form_id)}">Choisir</button>'
            f'<form id="{escape(form_id)}" method="post" action="/select">'
            '<input type="hidden" name="quantity" value="1">'
            "</form>"
            "</td>"
            "</tr>"
        )

    return "\n".join(rows)


def render_selection_result(
    result: SelectionResult | None,
    vision_result: VisionResult | None,
) -> str:
    if result is None:
        return (
            '<p class="muted">Aucune demande traitee pour le moment.</p>'
            '<div class="vision-slot">Point vision: en attente de selection.</div>'
        )

    status_class = "success" if result.ok else "error"
    messages = "".join(f"<li>{escape(message)}</li>" for message in result.messages)
    image_preview = ""
    if result.ok and result.article is not None:
        image_preview = (
            '<div class="image-preview">'
            f'<img src="{escape(image_url_for(result.article.image_path))}" '
            f'alt="{escape(result.article.nom)}">'
            "</div>"
        )

    vision_html = ""
    if vision_result is not None:
        vision_html = (
            '<div class="vision-slot">'
            "<strong>Point d'integration vision</strong>"
            "<span>cloud_app.py appelle vision.process_image(image_path).</span>"
            f"<span>{escape(vision_result.message)}</span>"
            "</div>"
        )

    return (
        f'<div class="status {status_class}">{escape(result.code)}</div>'
        f"<ul>{messages}</ul>"
        f"{image_preview}"
        f"{vision_html}"
    )


def render_assistant_output(text: str | None) -> str:
    if not text:
        return '<p class="muted">Le contexte LLM apparaitra ici.</p>'
    return f"<pre>{escape(text)}</pre>"


def render_page(
    *,
    selection_result: SelectionResult | None = None,
    vision_result: VisionResult | None = None,
    query: str = "",
    quantity: int = 1,
    symptoms: str = "",
    assistant_context: str | None = None,
) -> str:
    inventory = load_inventory()
    inventory_rows = render_inventory(inventory)
    selection_html = render_selection_result(selection_result, vision_result)
    assistant_html = render_assistant_output(assistant_context)

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{APP_NAME}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #637083;
      --line: #d8e0ea;
      --blue: #1d4ed8;
      --green: #15803d;
      --amber: #b45309;
      --red: #b91c1c;
      --teal: #0f766e;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header {{
      background: #101827;
      color: white;
      padding: 18px 28px;
      border-bottom: 4px solid var(--teal);
    }}
    header h1 {{
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }}
    main {{
      display: grid;
      grid-template-columns: minmax(520px, 1.2fr) minmax(360px, .8fr);
      gap: 18px;
      padding: 18px;
      max-width: 1440px;
      margin: 0 auto;
    }}
    section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-width: 0;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: 0;
    }}
    label {{
      display: block;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    input, textarea {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      background: white;
      color: var(--ink);
    }}
    textarea {{
      min-height: 122px;
      resize: vertical;
    }}
    button {{
      border: 0;
      border-radius: 6px;
      padding: 10px 14px;
      font: inherit;
      font-weight: 700;
      background: var(--blue);
      color: white;
      cursor: pointer;
      white-space: nowrap;
    }}
    button:hover {{ filter: brightness(.95); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: middle;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .request-grid {{
      display: grid;
      grid-template-columns: 1fr 120px;
      gap: 12px;
      align-items: end;
    }}
    .action-row {{
      margin-top: 12px;
    }}
    .status {{
      display: inline-block;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 10px;
    }}
    .success {{ background: #dcfce7; color: var(--green); }}
    .error {{ background: #fee2e2; color: var(--red); }}
    .stock-ok {{ color: var(--green); font-weight: 700; }}
    .stock-low {{ color: var(--amber); font-weight: 700; }}
    .muted {{ color: var(--muted); }}
    .image-preview {{
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 12px;
      overflow: hidden;
      background: #f8fafc;
    }}
    .image-preview img {{
      display: block;
      width: 100%;
      max-height: 210px;
      object-fit: contain;
    }}
    .vision-slot {{
      display: grid;
      gap: 6px;
      margin-top: 12px;
      padding: 12px;
      border: 1px solid #99f6e4;
      border-radius: 8px;
      background: #f0fdfa;
      color: #134e4a;
    }}
    pre {{
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      max-height: 360px;
      overflow: auto;
    }}
    .stack {{
      display: grid;
      gap: 18px;
    }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; padding: 12px; }}
      .request-grid {{ grid-template-columns: 1fr; }}
      table {{ font-size: 13px; }}
      th:nth-child(3), td:nth-child(3), th:nth-child(6), td:nth-child(6) {{
        display: none;
      }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{APP_NAME}</h1>
  </header>
  <main>
    <div class="stack">
      <section>
        <h2>Demande client</h2>
        <form method="post" action="/select">
          <div class="request-grid">
            <div>
              <label for="query">Nom ou ID article</label>
              <input id="query" name="query" value="{escape(query)}" placeholder="MED-001 ou amoxicilline">
            </div>
            <div>
              <label for="quantity">Quantite</label>
              <input id="quantity" name="quantity" value="{quantity}" inputmode="numeric">
            </div>
          </div>
          <div class="action-row">
            <button type="submit">Simuler la selection</button>
          </div>
        </form>
      </section>

      <section>
        <h2>Inventaire</h2>
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Article</th>
              <th>Classe</th>
              <th>Rayon</th>
              <th>Stock</th>
              <th>Prix</th>
              <th></th>
            </tr>
          </thead>
          <tbody>{inventory_rows}</tbody>
        </table>
      </section>
    </div>

    <div class="stack">
      <section>
        <h2>Resultat</h2>
        {selection_html}
      </section>

      <section>
        <h2>Assistant pharmacien LLM</h2>
        <form method="post" action="/assistant">
          <label for="symptoms">Symptomes decrits</label>
          <textarea id="symptoms" name="symptoms">{escape(symptoms)}</textarea>
          <div class="action-row">
            <button type="submit">Preparer le contexte</button>
          </div>
        </form>
        {assistant_html}
      </section>
    </div>
  </main>
</body>
</html>"""


def resolve_image_file(url_path: str) -> Path | None:
    images_root = resolve_project_path("images").resolve()
    relative_url = unquote(url_path.removeprefix("/images/"))
    candidate = (images_root / relative_url).resolve()

    try:
        candidate.relative_to(images_root)
    except ValueError:
        return None

    if candidate.is_file():
        return candidate
    return None


class CloudRequestHandler(BaseHTTPRequestHandler):
    server_version = "GestionStockVision/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.send_html(render_page())
            return

        if parsed.path == "/health":
            self.send_json({"status": "ok"})
            return

        if parsed.path == "/api/inventory":
            inventory = load_inventory()
            self.send_json(
                {"items": [article_to_payload(article) for article in inventory]}
            )
            return

        if parsed.path.startswith("/images/"):
            self.send_image(parsed.path)
            return

        self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        payload = self.read_payload()

        if parsed.path == "/select":
            query = str(payload.get("query", "")).strip()
            quantity = parse_quantity(payload.get("quantity"), default=0)
            result, vision_result = run_selection(query, quantity)
            self.send_html(
                render_page(
                    selection_result=result,
                    vision_result=vision_result,
                    query=query,
                    quantity=max(quantity, 1),
                )
            )
            return

        if parsed.path == "/assistant":
            symptoms = str(payload.get("symptoms", "")).strip()
            context = build_pharmacist_context(symptoms, load_inventory())
            self.send_html(
                render_page(
                    symptoms=symptoms,
                    assistant_context=context.as_text(),
                )
            )
            return

        if parsed.path == "/api/select":
            query = str(payload.get("query", "")).strip()
            quantity = parse_quantity(payload.get("quantity"), default=0)
            result, vision_result = run_selection(query, quantity)
            status = HTTPStatus.OK if result.ok else HTTPStatus.BAD_REQUEST
            self.send_json(selection_to_payload(result, vision_result), status=status)
            return

        if parsed.path == "/api/assistant":
            symptoms = str(payload.get("symptoms", "")).strip()
            context = build_pharmacist_context(symptoms, load_inventory())
            self.send_json(
                {
                    "system_prompt": context.system_prompt,
                    "user_context": context.user_context,
                    "text": context.as_text(),
                }
            )
            return

        self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)

    def read_payload(self) -> dict[str, object]:
        content_length = int(self.headers.get("Content-Length", "0") or "0")
        body = self.rfile.read(content_length) if content_length else b""
        return parse_request_payload(self.headers.get("Content-Type", ""), body)

    def send_html(self, html: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(
        self,
        payload: dict[str, object],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_image(self, url_path: str) -> None:
        image_file = resolve_image_file(url_path)
        if image_file is None:
            self.send_json({"error": "Image not found"}, status=HTTPStatus.NOT_FOUND)
            return

        body = image_file.read_bytes()
        content_type = mimetypes.guess_type(str(image_file))[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    server = ThreadingHTTPServer((host, port), CloudRequestHandler)
    print(f"{APP_NAME} disponible sur http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArret du serveur.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
