import json
import mimetypes
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from app import Article, SelectionResult, load_inventory, prepare_selection, resolve_project_path
from llm_assistant import (
    AssistantReply,
    build_pharmacist_context,
    call_remote_llm,
    configured_llm_api_url,
    normalize_llm_api_url,
)
from pill_dataset import discover_dataset_images, find_dataset_dir, resolve_dataset_image
from vision import VisionResult, process_image


APP_NAME = "Gestion Stock Vision"
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
LOW_STOCK_THRESHOLD = 50


def format_ariary(value: float) -> str:
    return f"{value:,.0f} Ar".replace(",", " ")


def stock_class_for(quantity: int) -> str:
    if quantity <= 0:
        return "stock-out"
    if quantity <= LOW_STOCK_THRESHOLD:
        return "stock-low"
    return "stock-ok"


def stock_label_for(quantity: int) -> str:
    if quantity <= 0:
        return "Rupture"
    if quantity <= LOW_STOCK_THRESHOLD:
        return "A surveiller"
    return "Disponible"


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
        "model_path": result.model_path,
        "image_path": result.image_path,
        "detections": [
            {
                "label": detection.label,
                "confidence": detection.confidence,
                "box_xyxy": list(detection.box_xyxy),
            }
            for detection in result.detections
        ],
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


def dataset_image_url_for(relative_path: str) -> str:
    return "/dataset-images/" + quote(relative_path, safe="/")


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
        stock_class = stock_class_for(article.quantite_stock)
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
            f'form="{escape(form_id)}">Selectionner</button>'
            f'<form id="{escape(form_id)}" method="post" action="/select">'
            '<input type="hidden" name="quantity" value="1">'
            "</form>"
            "</td>"
            "</tr>"
        )

    return "\n".join(rows)


def render_article_options(inventory: list[Article], selected_query: str = "") -> str:
    selected = selected_query.casefold().strip()
    options = [
        '<option value="">Selectionner un medicament du stock</option>'
    ]

    for article in inventory:
        is_selected = selected in {
            article.id_unique.casefold(),
            article.nom.casefold(),
        }
        label = (
            f"{article.id_unique} - {article.nom} | "
            f"{article.quantite_stock} en stock | {article.emplacement_rayon}"
        )
        options.append(
            f'<option value="{escape(article.id_unique)}" '
            f'{"selected" if is_selected else ""}>'
            f"{escape(label)}"
            "</option>"
        )

    return "\n".join(options)


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
            f"{render_detection_list(vision_result)}"
            "</div>"
        )

    return (
        f'<div class="status {status_class}">{escape(result.code)}</div>'
        f"<ul>{messages}</ul>"
        f"{image_preview}"
        f"{vision_html}"
    )


def render_detection_list(vision_result: VisionResult) -> str:
    if not vision_result.detections:
        return ""

    items = "".join(
        "<li>"
        f"{escape(detection.label)} - {detection.confidence:.2f} "
        f"({', '.join(f'{value:.0f}' for value in detection.box_xyxy)})"
        "</li>"
        for detection in vision_result.detections
    )
    return f'<ul class="detections">{items}</ul>'


def render_dataset_vision_test(
    vision_result: VisionResult | None = None,
    selected_dataset_image: str = "",
    dataset_vision_error: str = "",
) -> str:
    dataset_dir = find_dataset_dir()
    samples = discover_dataset_images(limit=30)

    if dataset_dir is None:
        dataset_status = (
            '<p class="muted">'
            "Dataset pilules introuvable. Definissez PILL_DATASET_DIR ou placez "
            "le dataset dans pill_data/pill."
            "</p>"
        )
    else:
        dataset_status = f'<p class="muted">Dataset detecte: {escape(str(dataset_dir))}</p>'

    if samples:
        options = "".join(
            f'<option value="{escape(sample.relative_path)}" '
            f'{"selected" if sample.relative_path == selected_dataset_image else ""}>'
            f"{escape(sample.relative_path)}"
            "</option>"
            for sample in samples
        )
        form = (
            '<form method="post" action="/vision-test">'
            '<label for="dataset_image">Image du dataset</label>'
            f'<select id="dataset_image" name="dataset_image">{options}</select>'
            '<div class="action-row">'
            '<button type="submit">Tester la vision</button>'
            "</div>"
            "</form>"
        )
    else:
        form = '<p class="muted">Aucune image JPG/PNG trouvee dans le dataset.</p>'

    preview = ""
    if selected_dataset_image:
        preview = (
            '<div class="image-preview">'
            f'<img src="{escape(dataset_image_url_for(selected_dataset_image))}" '
            f'alt="{escape(selected_dataset_image)}">'
            "</div>"
        )

    result_html = ""
    if dataset_vision_error:
        result_html = f'<div class="status error">{escape(dataset_vision_error)}</div>'
    elif vision_result is not None:
        status_class = "success" if vision_result.ok else "error"
        result_html = (
            f'<div class="status {status_class}">{escape(vision_result.message)}</div>'
            f"{render_detection_list(vision_result)}"
        )

    return dataset_status + form + preview + result_html


def assistant_reply_to_payload(
    context,
    reply: AssistantReply,
) -> dict[str, object]:
    return {
        "ok": reply.ok,
        "answer": reply.answer if reply.ok else None,
        "error": reply.error if not reply.ok else None,
        "system_prompt": context.system_prompt,
        "user_context": context.user_context,
        "upstream_status_code": reply.status_code,
    }


def render_assistant_output(text: str | None, ok: bool | None = None) -> str:
    if not text:
        return '<p class="muted">La reponse du LLM apparaitra ici.</p>'

    status = ""
    if ok is not None:
        status_class = "success" if ok else "error"
        label = "REPONSE_LLM" if ok else "LLM_INDISPONIBLE"
        status = f'<div class="status {status_class}">{label}</div>'

    return f"{status}<pre>{escape(text)}</pre>"


def render_base_styles() -> str:
    return """
    :root {
      color-scheme: light;
      --bg: #f3f6f4;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #637083;
      --line: #d8e0ea;
      --blue: #1d4ed8;
      --green: #15803d;
      --amber: #b45309;
      --red: #b91c1c;
      --teal: #0f766e;
      --violet: #6d28d9;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }
    header {
      background: #101827;
      color: white;
      padding: 16px 24px;
      border-bottom: 4px solid var(--teal);
    }
    .topbar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      max-width: 1440px;
      margin: 0 auto;
    }
    header h1 {
      margin: 0;
      font-size: 24px;
      letter-spacing: 0;
    }
    nav {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    nav a {
      color: white;
      text-decoration: none;
      border: 1px solid rgba(255, 255, 255, .24);
      border-radius: 6px;
      padding: 8px 12px;
      font-weight: 700;
    }
    nav a.active {
      background: white;
      color: #101827;
    }
    main {
      padding: 18px;
      max-width: 1440px;
      margin: 0 auto;
    }
    .workspace-main {
      display: grid;
      grid-template-columns: minmax(520px, 1.2fr) minmax(360px, .8fr);
      gap: 18px;
    }
    .dashboard-main {
      display: grid;
      gap: 18px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      min-width: 0;
    }
    h2 {
      margin: 0 0 14px;
      font-size: 18px;
      letter-spacing: 0;
    }
    h3 {
      margin: 0 0 10px;
      font-size: 15px;
      letter-spacing: 0;
    }
    label {
      display: block;
      font-weight: 700;
      margin-bottom: 6px;
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font: inherit;
      background: white;
      color: var(--ink);
    }
    textarea {
      min-height: 122px;
      resize: vertical;
    }
    button {
      border: 0;
      border-radius: 6px;
      padding: 10px 14px;
      font: inherit;
      font-weight: 700;
      background: var(--blue);
      color: white;
      cursor: pointer;
      white-space: nowrap;
    }
    button:hover { filter: brightness(.95); }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 10px 8px;
      text-align: left;
      vertical-align: middle;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .table-wrap {
      overflow-x: auto;
    }
    .request-grid {
      display: grid;
      grid-template-columns: 1fr 120px;
      gap: 12px;
      align-items: end;
    }
    .action-row {
      margin-top: 12px;
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
    }
    .status {
      display: inline-block;
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 10px;
    }
    .success { background: #dcfce7; color: var(--green); }
    .error { background: #fee2e2; color: var(--red); }
    .stock-ok { color: var(--green); font-weight: 700; }
    .stock-low { color: var(--amber); font-weight: 700; }
    .stock-out { color: var(--red); font-weight: 700; }
    .muted { color: var(--muted); }
    .image-preview {
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-top: 12px;
      overflow: hidden;
      background: #f8fafc;
    }
    .image-preview img {
      display: block;
      width: 100%;
      max-height: 210px;
      object-fit: contain;
    }
    .vision-slot {
      display: grid;
      gap: 6px;
      margin-top: 12px;
      padding: 12px;
      border: 1px solid #99f6e4;
      border-radius: 8px;
      background: #f0fdfa;
      color: #134e4a;
    }
    .detections {
      margin: 8px 0 0;
      padding-left: 20px;
    }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      max-height: 360px;
      overflow: auto;
    }
    .stack {
      display: grid;
      gap: 18px;
    }
    .dashboard-heading {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
    }
    .dashboard-heading h2 {
      margin-bottom: 4px;
      font-size: 24px;
    }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
    }
    .metric-card {
      display: grid;
      gap: 6px;
      border-top: 4px solid var(--teal);
    }
    .metric-card.warning { border-top-color: var(--amber); }
    .metric-card.good { border-top-color: var(--green); }
    .metric-card.value { border-top-color: var(--violet); }
    .metric-label {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }
    .metric-value {
      font-size: 26px;
      font-weight: 800;
    }
    .dashboard-grid {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr);
      gap: 18px;
    }
    .bar-list {
      display: grid;
      gap: 12px;
    }
    .bar-row {
      display: grid;
      gap: 5px;
    }
    .bar-top {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: 14px;
    }
    .bar-track {
      height: 10px;
      border-radius: 999px;
      background: #e8edf3;
      overflow: hidden;
    }
    .bar-fill {
      display: block;
      height: 100%;
      min-width: 6px;
      border-radius: 999px;
      background: var(--teal);
    }
    .bar-fill.value { background: var(--violet); }
    @media (max-width: 980px) {
      .topbar { align-items: flex-start; flex-direction: column; }
      main { padding: 12px; }
      .workspace-main,
      .dashboard-grid,
      .metric-grid { grid-template-columns: 1fr; }
      .request-grid { grid-template-columns: 1fr; }
      table { font-size: 13px; }
      .hide-mobile { display: none; }
      th:nth-child(3), td:nth-child(3), th:nth-child(6), td:nth-child(6) {
        display: none;
      }
    }
    """


def render_shell(active_page: str, main_html: str, main_class: str) -> str:
    home_active = "active" if active_page == "home" else ""
    dashboard_active = "active" if active_page == "dashboard" else ""

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{APP_NAME}</title>
  <style>{render_base_styles()}</style>
</head>
<body>
  <header>
    <div class="topbar">
      <h1>{APP_NAME}</h1>
      <nav aria-label="Navigation principale">
        <a class="{home_active}" href="/">Stock</a>
        <a class="{dashboard_active}" href="/dashboard">Tableau de bord</a>
      </nav>
    </div>
  </header>
  <main class="{escape(main_class)}">
    {main_html}
  </main>
</body>
</html>"""


def build_dashboard_payload(inventory: list[Article]) -> dict[str, object]:
    stock_by_class: dict[str, int] = {}
    value_by_location: dict[str, float] = {}

    for article in inventory:
        stock_by_class[article.classe_therapeutique] = (
            stock_by_class.get(article.classe_therapeutique, 0)
            + article.quantite_stock
        )
        value_by_location[article.emplacement_rayon] = (
            value_by_location.get(article.emplacement_rayon, 0.0)
            + article.quantite_stock * article.prix
        )

    low_stock_items = sorted(
        (
            article
            for article in inventory
            if article.quantite_stock <= LOW_STOCK_THRESHOLD
        ),
        key=lambda article: article.quantite_stock,
    )
    out_of_stock_items = [
        article for article in inventory if article.quantite_stock <= 0
    ]
    total_units = sum(article.quantite_stock for article in inventory)
    total_value = sum(article.quantite_stock * article.prix for article in inventory)

    return {
        "low_stock_threshold": LOW_STOCK_THRESHOLD,
        "total_references": len(inventory),
        "total_units": total_units,
        "total_value": total_value,
        "total_value_formatted": format_ariary(total_value),
        "low_stock_count": len(low_stock_items),
        "out_of_stock_count": len(out_of_stock_items),
        "stock_by_class": [
            {"label": label, "units": units}
            for label, units in sorted(stock_by_class.items())
        ],
        "value_by_location": [
            {
                "label": label,
                "value": value,
                "value_formatted": format_ariary(value),
            }
            for label, value in sorted(value_by_location.items())
        ],
        "low_stock_items": [
            article_to_payload(article) for article in low_stock_items
        ],
    }


def render_bar_list(
    items: list[dict[str, object]],
    *,
    value_key: str,
    label_key: str,
    display_key: str,
    bar_class: str = "",
) -> str:
    numeric_values = [
        float(item.get(value_key, 0) or 0)
        for item in items
    ]
    max_value = max(numeric_values, default=0.0)
    if max_value <= 0:
        return '<p class="muted">Aucune donnee disponible.</p>'

    rows = []
    for item in items:
        value = float(item.get(value_key, 0) or 0)
        width = max(3.0, min(100.0, (value / max_value) * 100))
        label = str(item.get(label_key, ""))
        display = str(item.get(display_key, value))
        rows.append(
            '<div class="bar-row">'
            '<div class="bar-top">'
            f"<span>{escape(label)}</span>"
            f"<strong>{escape(display)}</strong>"
            "</div>"
            '<div class="bar-track">'
            f'<span class="bar-fill {escape(bar_class)}" style="width: {width:.1f}%"></span>'
            "</div>"
            "</div>"
        )

    return '<div class="bar-list">' + "".join(rows) + "</div>"


def render_low_stock_table(inventory: list[Article]) -> str:
    low_stock_items = sorted(
        (
            article
            for article in inventory
            if article.quantite_stock <= LOW_STOCK_THRESHOLD
        ),
        key=lambda article: article.quantite_stock,
    )

    if not low_stock_items:
        return '<p class="muted">Aucun stock faible selon le seuil actuel.</p>'

    rows = []
    for article in low_stock_items:
        stock_class = stock_class_for(article.quantite_stock)
        rows.append(
            "<tr>"
            f"<td><strong>{escape(article.id_unique)}</strong></td>"
            f"<td>{escape(article.nom)}</td>"
            f"<td>{escape(article.emplacement_rayon)}</td>"
            f'<td class="{stock_class}">{article.quantite_stock}</td>'
            f"<td>{escape(stock_label_for(article.quantite_stock))}</td>"
            "</tr>"
        )

    return (
        '<div class="table-wrap"><table>'
        "<thead><tr><th>ID</th><th>Article</th><th>Rayon</th>"
        "<th>Stock</th><th>Statut</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_dashboard_inventory_table(inventory: list[Article]) -> str:
    rows = []
    for article in inventory:
        stock_value = article.quantite_stock * article.prix
        stock_class = stock_class_for(article.quantite_stock)
        rows.append(
            "<tr>"
            f"<td><strong>{escape(article.id_unique)}</strong></td>"
            f"<td>{escape(article.nom)}</td>"
            f"<td>{escape(article.classe_therapeutique)}</td>"
            f"<td>{escape(article.emplacement_rayon)}</td>"
            f'<td class="{stock_class}">{article.quantite_stock}</td>'
            f"<td>{escape(format_ariary(stock_value))}</td>"
            f"<td>{escape(stock_label_for(article.quantite_stock))}</td>"
            "</tr>"
        )

    return (
        '<div class="table-wrap"><table>'
        "<thead><tr><th>ID</th><th>Article</th><th>Classe</th><th>Rayon</th>"
        "<th>Stock</th><th>Valeur</th><th>Statut</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def render_dashboard_page() -> str:
    inventory = load_inventory()
    payload = build_dashboard_payload(inventory)
    stock_by_class = payload["stock_by_class"]
    value_by_location = payload["value_by_location"]
    assert isinstance(stock_by_class, list)
    assert isinstance(value_by_location, list)

    class_bars = render_bar_list(
        stock_by_class,
        value_key="units",
        label_key="label",
        display_key="units",
    )
    location_bars = render_bar_list(
        value_by_location,
        value_key="value",
        label_key="label",
        display_key="value_formatted",
        bar_class="value",
    )

    body = f"""
    <div class="dashboard-heading">
      <div>
        <h2>Tableau de bord stock pharmacie</h2>
        <p class="muted">Vue centree sur les references, les quantites, les rayons et la valeur du stock.</p>
      </div>
      <a href="/" class="muted">Retour a la selection</a>
    </div>

    <div class="metric-grid">
      <section class="metric-card good">
        <span class="metric-label">References</span>
        <span class="metric-value">{payload["total_references"]}</span>
        <span class="muted">Medicaments suivis</span>
      </section>
      <section class="metric-card">
        <span class="metric-label">Unites en stock</span>
        <span class="metric-value">{payload["total_units"]}</span>
        <span class="muted">Toutes references confondues</span>
      </section>
      <section class="metric-card warning">
        <span class="metric-label">Stocks faibles</span>
        <span class="metric-value">{payload["low_stock_count"]}</span>
        <span class="muted">Seuil: {LOW_STOCK_THRESHOLD} unites</span>
      </section>
      <section class="metric-card value">
        <span class="metric-label">Valeur stock</span>
        <span class="metric-value">{escape(str(payload["total_value_formatted"]))}</span>
        <span class="muted">Valorisation indicative</span>
      </section>
    </div>

    <div class="dashboard-grid">
      <section>
        <h2>Stock par classe</h2>
        {class_bars}
      </section>
      <section>
        <h2>Valeur par rayon</h2>
        {location_bars}
      </section>
    </div>

    <div class="dashboard-grid">
      <section>
        <h2>Articles a surveiller</h2>
        {render_low_stock_table(inventory)}
      </section>
      <section>
        <h2>Inventaire valorise</h2>
        {render_dashboard_inventory_table(inventory)}
      </section>
    </div>
    """

    return render_shell("dashboard", body, "dashboard-main")


def render_page(
    *,
    selection_result: SelectionResult | None = None,
    vision_result: VisionResult | None = None,
    dataset_vision_result: VisionResult | None = None,
    selected_dataset_image: str = "",
    dataset_vision_error: str = "",
    query: str = "",
    quantity: int = 1,
    symptoms: str = "",
    assistant_output: str | None = None,
    assistant_ok: bool | None = None,
) -> str:
    inventory = load_inventory()
    inventory_rows = render_inventory(inventory)
    article_options = render_article_options(inventory, query)
    selection_html = render_selection_result(selection_result, vision_result)
    dataset_vision_html = render_dataset_vision_test(
        vision_result=dataset_vision_result,
        selected_dataset_image=selected_dataset_image,
        dataset_vision_error=dataset_vision_error,
    )
    assistant_html = render_assistant_output(assistant_output, assistant_ok)
    llm_url = normalize_llm_api_url(configured_llm_api_url(use_default=True))

    body = f"""
    <div class="stack">
      <section>
        <h2>Sortie stock</h2>
        <form method="post" action="/select">
          <div class="request-grid">
            <div>
              <label for="query">Medicament</label>
              <select id="query" name="query" required>
                {article_options}
              </select>
            </div>
            <div>
              <label for="quantity">Quantite</label>
              <input id="quantity" name="quantity" value="{quantity}" type="number" min="1" step="1" inputmode="numeric">
            </div>
          </div>
          <div class="action-row">
            <button type="submit">Verifier le stock</button>
            <a href="/dashboard" class="muted">Ouvrir le tableau de bord</a>
          </div>
        </form>
      </section>

      <section>
        <h2>Inventaire</h2>
        <div class="table-wrap">
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
        </div>
      </section>
    </div>

    <div class="stack">
      <section>
        <h2>Resultat</h2>
        {selection_html}
      </section>

      <section>
        <h2>Controle vision dataset</h2>
        {dataset_vision_html}
      </section>

      <section>
        <h2>Assistant stock LLM</h2>
        <p class="muted">Serveur LLM: {escape(llm_url)}</p>
        <form method="post" action="/assistant">
          <label for="symptoms">Question de gestion de stock</label>
          <textarea id="symptoms" name="symptoms" placeholder="Exemple: quelles references dois-je reapprovisionner en priorite ?">{escape(symptoms)}</textarea>
          <div class="action-row">
            <button type="submit">Envoyer au LLM</button>
          </div>
        </form>
        {assistant_html}
      </section>
    </div>
    """

    return render_shell("home", body, "workspace-main")


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


def resolve_dataset_image_file(url_path: str) -> Path | None:
    relative_url = unquote(url_path.removeprefix("/dataset-images/"))
    return resolve_dataset_image(relative_url)


class CloudRequestHandler(BaseHTTPRequestHandler):
    server_version = "GestionStockVision/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/":
            self.send_html(render_page())
            return

        if parsed.path == "/dashboard":
            self.send_html(render_dashboard_page())
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

        if parsed.path == "/api/dashboard":
            inventory = load_inventory()
            self.send_json(build_dashboard_payload(inventory))
            return

        if parsed.path == "/api/dataset/samples":
            samples = discover_dataset_images(limit=30)
            self.send_json(
                {
                    "dataset_dir": str(find_dataset_dir()) if find_dataset_dir() else None,
                    "items": [
                        {
                            "relative_path": sample.relative_path,
                            "split": sample.split,
                            "image_url": dataset_image_url_for(sample.relative_path),
                        }
                        for sample in samples
                    ],
                }
            )
            return

        if parsed.path.startswith("/images/"):
            self.send_image(parsed.path)
            return

        if parsed.path.startswith("/dataset-images/"):
            self.send_dataset_image(parsed.path)
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
            reply = call_remote_llm(
                context,
                api_url=configured_llm_api_url(use_default=True),
            )
            self.send_html(
                render_page(
                    symptoms=symptoms,
                    assistant_output=reply.display_text(context),
                    assistant_ok=reply.ok,
                )
            )
            return

        if parsed.path == "/vision-test":
            selected_image = str(payload.get("dataset_image", "")).strip()
            image_path = resolve_dataset_image(selected_image)
            if image_path is None:
                self.send_html(
                    render_page(
                        selected_dataset_image=selected_image,
                        dataset_vision_error="Image dataset introuvable ou non autorisee.",
                    )
                )
                return
            vision_result = process_image(image_path)
            self.send_html(
                render_page(
                    dataset_vision_result=vision_result,
                    selected_dataset_image=selected_image,
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
            reply = call_remote_llm(
                context,
                api_url=configured_llm_api_url(use_default=True),
            )
            status = HTTPStatus.OK if reply.ok else HTTPStatus.BAD_GATEWAY
            self.send_json(assistant_reply_to_payload(context, reply), status=status)
            return

        if parsed.path == "/api/vision-test":
            selected_image = str(payload.get("image_path", "")).strip()
            image_path = resolve_dataset_image(selected_image)
            if image_path is None:
                self.send_json(
                    {"ok": False, "error": "Image dataset introuvable ou non autorisee."},
                    status=HTTPStatus.BAD_REQUEST,
                )
                return
            vision_result = process_image(image_path)
            status = HTTPStatus.OK if vision_result.ok else HTTPStatus.BAD_REQUEST
            self.send_json(vision_to_payload(vision_result), status=status)
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

    def send_dataset_image(self, url_path: str) -> None:
        image_file = resolve_dataset_image_file(url_path)
        if image_file is None:
            self.send_json({"error": "Dataset image not found"}, status=HTTPStatus.NOT_FOUND)
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
    llm_url = normalize_llm_api_url(configured_llm_api_url(use_default=True))
    print(f"LLM distant configure sur {llm_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArret du serveur.")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
