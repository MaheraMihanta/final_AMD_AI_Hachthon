from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st

from app import Article, load_inventory, prepare_selection, resolve_project_path
from llm_assistant import (
    build_pharmacist_context,
    call_remote_llm,
    configured_llm_api_url,
    normalize_llm_api_url,
)
from pill_dataset import DatasetImage, discover_dataset_images
from vision import VisionResult, process_image


APP_TITLE = "Gestion Stock Vision"
DEFAULT_LOW_STOCK_THRESHOLD = 50


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=None,
    layout="wide",
)


def format_ariary(value: float) -> str:
    return f"{value:,.0f} Ar".replace(",", " ")


def stock_status(quantity: int, threshold: int) -> str:
    if quantity <= 0:
        return "Rupture"
    if quantity <= threshold:
        return "A surveiller"
    return "Disponible"


@st.cache_data(ttl=30)
def cached_inventory() -> list[Article]:
    return load_inventory()


@st.cache_data(ttl=30)
def cached_dataset_images(limit: int = 60) -> list[DatasetImage]:
    return discover_dataset_images(limit=limit)


def inventory_dataframe(
    inventory: list[Article],
    low_stock_threshold: int,
) -> pd.DataFrame:
    rows = []
    for article in inventory:
        rows.append(
            {
                "ID": article.id_unique,
                "Medicament": article.nom,
                "Classe": article.classe_therapeutique,
                "Rayon": article.emplacement_rayon,
                "Stock": article.quantite_stock,
                "Statut": stock_status(article.quantite_stock, low_stock_threshold),
                "Prix": article.prix,
                "Valeur": article.quantite_stock * article.prix,
                "Image": article.image_path.as_posix(),
            }
        )
    return pd.DataFrame(rows)


def format_article_option(article: Article) -> str:
    return (
        f"{article.id_unique} - {article.nom} | "
        f"{article.quantite_stock} en stock | {article.emplacement_rayon}"
    )


def article_by_id(inventory: list[Article], article_id: str) -> Article:
    for article in inventory:
        if article.id_unique == article_id:
            return article
    return inventory[0]


def show_article_image(article: Article) -> None:
    image_path = resolve_project_path(article.image_path)
    if not image_path.exists():
        st.warning(f"Image introuvable: {article.image_path}")
        return

    try:
        st.image(str(image_path), caption=article.nom, width="stretch")
    except Exception as exc:
        st.warning(f"Impossible d'afficher l'image: {exc}")


def show_dataset_image(sample: DatasetImage) -> None:
    try:
        st.image(str(sample.path), caption=sample.relative_path, width="stretch")
    except Exception as exc:
        st.warning(f"Impossible d'afficher l'image dataset: {exc}")


def show_vision_result(result: VisionResult) -> None:
    if result.ok:
        st.success(result.message)
    elif result.enabled:
        st.warning(result.message)
    else:
        st.info(result.message)

    if not result.detections:
        return

    detections = pd.DataFrame(
        [
            {
                "Label": detection.label,
                "Confiance": detection.confidence,
                "X1": detection.box_xyxy[0],
                "Y1": detection.box_xyxy[1],
                "X2": detection.box_xyxy[2],
                "Y2": detection.box_xyxy[3],
            }
            for detection in result.detections
        ]
    )
    st.dataframe(
        detections,
        hide_index=True,
        width="stretch",
        column_config={
            "Confiance": st.column_config.NumberColumn(format="%.2f"),
            "X1": st.column_config.NumberColumn(format="%.1f"),
            "Y1": st.column_config.NumberColumn(format="%.1f"),
            "X2": st.column_config.NumberColumn(format="%.1f"),
            "Y2": st.column_config.NumberColumn(format="%.1f"),
        },
    )


def secret_value(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except Exception:
        value = ""
    return str(value or "").strip()


def configured_streamlit_llm_url() -> str:
    configured = (
        os.getenv("LLM_API_URL", "").strip()
        or secret_value("LLM_API_URL")
        or configured_llm_api_url(use_default=True)
    )
    return normalize_llm_api_url(configured)


def configured_streamlit_llm_key() -> str:
    return os.getenv("LLM_API_KEY", "").strip() or secret_value("LLM_API_KEY")


def render_dashboard(inventory: list[Article], low_stock_threshold: int) -> None:
    df = inventory_dataframe(inventory, low_stock_threshold)
    total_units = int(df["Stock"].sum()) if not df.empty else 0
    total_value = float(df["Valeur"].sum()) if not df.empty else 0.0
    low_count = int((df["Stock"] <= low_stock_threshold).sum()) if not df.empty else 0
    out_count = int((df["Stock"] <= 0).sum()) if not df.empty else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("References", len(inventory))
    col2.metric("Unites", total_units)
    col3.metric("Stocks faibles", low_count)
    col4.metric("Valeur stock", format_ariary(total_value))

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Stock par classe")
        class_df = (
            df.groupby("Classe", as_index=True)["Stock"].sum().sort_values(ascending=False)
            if not df.empty
            else pd.Series(dtype="int64")
        )
        st.bar_chart(class_df)

    with chart_right:
        st.subheader("Valeur par rayon")
        location_df = (
            df.groupby("Rayon", as_index=True)["Valeur"].sum().sort_values(ascending=False)
            if not df.empty
            else pd.Series(dtype="float64")
        )
        st.bar_chart(location_df)

    st.subheader("Inventaire")
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Prix": st.column_config.NumberColumn(format="%d Ar"),
            "Valeur": st.column_config.NumberColumn(format="%d Ar"),
            "Image": None,
        },
    )


def render_stock_selection(inventory: list[Article]) -> None:
    options = [article.id_unique for article in inventory]
    selected_id = st.selectbox(
        "Medicament",
        options=options,
        format_func=lambda item: format_article_option(article_by_id(inventory, item)),
    )
    quantity = st.number_input("Quantite", min_value=1, max_value=999, value=1, step=1)

    selected_article = article_by_id(inventory, selected_id)
    preview, result_area = st.columns([0.85, 1.15])

    with preview:
        show_article_image(selected_article)
        st.write(f"Rayon: {selected_article.emplacement_rayon}")
        st.write(f"Prix: {selected_article.prix_formate}")

    with result_area:
        if st.button("Verifier et selectionner", type="primary"):
            result = prepare_selection(inventory, selected_id, int(quantity))
            if result.ok:
                st.success(result.code)
            else:
                st.error(result.code)

            for message in result.messages:
                st.write(message)

            if result.ok and result.article is not None:
                with st.spinner("Analyse vision"):
                    vision_result = process_image(resolve_project_path(result.article.image_path))
                show_vision_result(vision_result)
        else:
            st.info("Selection en attente.")


def render_dataset_vision() -> None:
    samples = cached_dataset_images()
    if not samples:
        st.info("Aucune image dataset YOLO trouvee.")
        return

    selected_path = st.selectbox(
        "Image dataset",
        options=[sample.relative_path for sample in samples],
    )
    sample = next(item for item in samples if item.relative_path == selected_path)

    preview, result_area = st.columns([0.85, 1.15])
    with preview:
        show_dataset_image(sample)
        st.write(f"Split: {sample.split}")
        st.write(Path(sample.path).name)

    with result_area:
        if st.button("Tester la vision", type="primary"):
            with st.spinner("Inference YOLO"):
                result = process_image(sample.path)
            show_vision_result(result)
        else:
            st.info("Test vision en attente.")


def render_assistant(inventory: list[Article], llm_url: str, llm_api_key: str) -> None:
    question = st.text_area(
        "Question de gestion de stock",
        value="Quelles references dois-je reapprovisionner en priorite ?",
        height=120,
    )

    if st.button("Envoyer au LLM", type="primary"):
        context = build_pharmacist_context(question, inventory)
        with st.spinner("Appel du serveur LLM"):
            reply = call_remote_llm(
                context,
                api_url=llm_url,
                api_key=llm_api_key,
            )

        if reply.ok:
            st.success("Reponse recue")
            st.write(reply.answer)
        else:
            st.error(reply.error or "Impossible de joindre le LLM distant.")
            with st.expander("Contexte prepare"):
                st.code(context.as_text())


def main() -> None:
    st.title(APP_TITLE)

    inventory = cached_inventory()
    llm_url = configured_streamlit_llm_url()
    llm_api_key = configured_streamlit_llm_key()

    with st.sidebar:
        st.header("Parametres")
        low_stock_threshold = st.number_input(
            "Seuil stock faible",
            min_value=1,
            max_value=500,
            value=DEFAULT_LOW_STOCK_THRESHOLD,
            step=1,
        )
        llm_url = st.text_input("Serveur LLM", value=llm_url)
        llm_api_key = st.text_input("Cle LLM", value=llm_api_key, type="password")

    dashboard_tab, selection_tab, vision_tab, assistant_tab = st.tabs(
        ["Tableau de bord", "Sortie stock", "Vision dataset", "Assistant LLM"]
    )

    with dashboard_tab:
        render_dashboard(inventory, int(low_stock_threshold))

    with selection_tab:
        render_stock_selection(inventory)

    with vision_tab:
        render_dataset_vision()

    with assistant_tab:
        render_assistant(inventory, llm_url, llm_api_key)


if __name__ == "__main__":
    main()
