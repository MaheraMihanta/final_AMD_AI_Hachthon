from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import tkinter as tk
    import customtkinter as ctk
except ImportError:  # pragma: no cover - depends on local environment
    tk = None
    ctk = None

from app import Article, load_inventory, prepare_selection, resolve_project_path
from llm_assistant import build_pharmacist_context
from vision import process_image


APP_TITLE = "Gestion Stock Vision"


def open_path(path: Path) -> None:
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
        return
    subprocess.Popen(["xdg-open", str(path)])


if ctk is not None and tk is not None:

    class StockVisionApp(ctk.CTk):
        def __init__(self) -> None:
            super().__init__()

            self.inventory: list[Article] = load_inventory()
            self.last_selected_article: Article | None = None
            self.last_image_path: Path | None = None
            self.quantity_var = tk.StringVar(value="1")

            ctk.set_appearance_mode("system")
            ctk.set_default_color_theme("blue")

            self.title(APP_TITLE)
            self.geometry("1180x760")
            self.minsize(980, 640)

            self.grid_columnconfigure(0, weight=0, minsize=330)
            self.grid_columnconfigure(1, weight=1)
            self.grid_rowconfigure(0, weight=1)

            self._build_inventory_panel()
            self._build_work_panel()
            self._refresh_inventory_rows()

        def _build_inventory_panel(self) -> None:
            self.inventory_panel = ctk.CTkFrame(self, corner_radius=0)
            self.inventory_panel.grid(row=0, column=0, sticky="nsew")
            self.inventory_panel.grid_rowconfigure(2, weight=1)
            self.inventory_panel.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                self.inventory_panel,
                text="Inventaire",
                font=ctk.CTkFont(size=22, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=18, pady=(18, 6))

            ctk.CTkLabel(
                self.inventory_panel,
                text="Cliquez sur un article pour remplir la demande.",
                anchor="w",
                text_color=("gray35", "gray70"),
            ).grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 12))

            self.inventory_rows = ctk.CTkScrollableFrame(
                self.inventory_panel,
                corner_radius=8,
            )
            self.inventory_rows.grid(row=2, column=0, sticky="nsew", padx=14, pady=10)
            self.inventory_rows.grid_columnconfigure(0, weight=1)

            ctk.CTkButton(
                self.inventory_panel,
                text="Recharger le stock",
                command=self._reload_inventory,
            ).grid(row=3, column=0, sticky="ew", padx=18, pady=(8, 18))

        def _build_work_panel(self) -> None:
            self.work_panel = ctk.CTkFrame(self, fg_color="transparent")
            self.work_panel.grid(row=0, column=1, sticky="nsew", padx=22, pady=22)
            self.work_panel.grid_columnconfigure(0, weight=1)
            self.work_panel.grid_columnconfigure(1, weight=1)
            self.work_panel.grid_rowconfigure(2, weight=1)

            ctk.CTkLabel(
                self.work_panel,
                text="Poste pharmacien",
                font=ctk.CTkFont(size=26, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 16))

            self._build_request_frame()
            self._build_result_frame()
            self._build_assistant_frame()

        def _build_request_frame(self) -> None:
            request_frame = ctk.CTkFrame(self.work_panel, corner_radius=8)
            request_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(0, 16))
            request_frame.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                request_frame,
                text="Demande client",
                font=ctk.CTkFont(size=18, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, columnspan=4, sticky="ew", padx=16, pady=(16, 10))

            ctk.CTkLabel(request_frame, text="Nom ou ID article", anchor="w").grid(
                row=1, column=0, columnspan=4, sticky="ew", padx=16
            )
            self.query_entry = ctk.CTkEntry(
                request_frame,
                placeholder_text="Ex: MED-001 ou amoxicilline",
            )
            self.query_entry.grid(
                row=2, column=0, columnspan=4, sticky="ew", padx=16, pady=(6, 12)
            )
            self.query_entry.bind("<Return>", lambda _event: self._simulate_selection())

            ctk.CTkLabel(request_frame, text="Quantite", anchor="w").grid(
                row=3, column=0, sticky="w", padx=16
            )
            ctk.CTkButton(
                request_frame,
                text="-",
                width=42,
                command=lambda: self._change_quantity(-1),
            ).grid(row=4, column=0, sticky="w", padx=(16, 4), pady=(6, 16))
            self.quantity_entry = ctk.CTkEntry(
                request_frame,
                width=74,
                textvariable=self.quantity_var,
                justify="center",
            )
            self.quantity_entry.grid(row=4, column=1, sticky="w", padx=4, pady=(6, 16))
            ctk.CTkButton(
                request_frame,
                text="+",
                width=42,
                command=lambda: self._change_quantity(1),
            ).grid(row=4, column=2, sticky="w", padx=4, pady=(6, 16))

            ctk.CTkButton(
                request_frame,
                text="Simuler la selection",
                command=self._simulate_selection,
            ).grid(row=5, column=0, columnspan=4, sticky="ew", padx=16, pady=(0, 16))

        def _build_result_frame(self) -> None:
            result_frame = ctk.CTkFrame(self.work_panel, corner_radius=8)
            result_frame.grid(row=1, column=1, sticky="nsew", padx=(10, 0), pady=(0, 16))
            result_frame.grid_columnconfigure(0, weight=1)
            result_frame.grid_rowconfigure(2, weight=1)

            ctk.CTkLabel(
                result_frame,
                text="Resultat et point vision",
                font=ctk.CTkFont(size=18, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=16, pady=(16, 10))

            self.status_label = ctk.CTkLabel(
                result_frame,
                text="En attente d'une demande",
                anchor="w",
                text_color=("gray30", "gray75"),
            )
            self.status_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 8))

            self.result_text = ctk.CTkTextbox(result_frame, height=210, wrap="word")
            self.result_text.grid(row=2, column=0, sticky="nsew", padx=16, pady=(0, 12))
            self._set_text(
                self.result_text,
                "Selectionnez un article ou saisissez une demande pour lancer la simulation.",
            )

            self.open_image_button = ctk.CTkButton(
                result_frame,
                text="Ouvrir l'image selectionnee",
                command=self._open_selected_image,
                state="disabled",
            )
            self.open_image_button.grid(
                row=3, column=0, sticky="ew", padx=16, pady=(0, 16)
            )

        def _build_assistant_frame(self) -> None:
            assistant_frame = ctk.CTkFrame(self.work_panel, corner_radius=8)
            assistant_frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
            assistant_frame.grid_columnconfigure(0, weight=1)
            assistant_frame.grid_columnconfigure(1, weight=1)
            assistant_frame.grid_rowconfigure(1, weight=1)

            ctk.CTkLabel(
                assistant_frame,
                text="Assistant pharmacien LLM - futur",
                font=ctk.CTkFont(size=18, weight="bold"),
                anchor="w",
            ).grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(16, 10))

            self.symptoms_text = ctk.CTkTextbox(
                assistant_frame,
                height=180,
                wrap="word",
            )
            self.symptoms_text.grid(
                row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 12)
            )
            self.symptoms_text.insert(
                "1.0",
                "Decrire ici les symptomes du client quand il n'a pas d'ordonnance.",
            )

            self.assistant_output = ctk.CTkTextbox(
                assistant_frame,
                height=180,
                wrap="word",
            )
            self.assistant_output.grid(
                row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 12)
            )
            self._set_text(
                self.assistant_output,
                "Cette zone preparera le contexte a envoyer a un LLM. "
                "La recommandation finale reste cote pharmacien.",
            )

            ctk.CTkButton(
                assistant_frame,
                text="Preparer le contexte LLM",
                command=self._prepare_llm_context,
            ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=16, pady=(0, 16))

        def _refresh_inventory_rows(self) -> None:
            for child in self.inventory_rows.winfo_children():
                child.destroy()

            for row_index, article in enumerate(self.inventory):
                row = ctk.CTkFrame(self.inventory_rows, corner_radius=8)
                row.grid(row=row_index, column=0, sticky="ew", pady=5)
                row.grid_columnconfigure(0, weight=1)

                stock_color = "#166534" if article.quantite_stock > 20 else "#b45309"
                title = f"{article.id_unique} - {article.nom}"
                details = (
                    f"{article.classe_therapeutique} | {article.emplacement_rayon} | "
                    f"stock {article.quantite_stock}"
                )

                ctk.CTkLabel(
                    row,
                    text=title,
                    anchor="w",
                    font=ctk.CTkFont(size=14, weight="bold"),
                ).grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 2))
                ctk.CTkLabel(
                    row,
                    text=details,
                    anchor="w",
                    text_color=stock_color,
                ).grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
                ctk.CTkButton(
                    row,
                    text="Choisir",
                    width=78,
                    command=lambda selected=article: self._fill_article(selected),
                ).grid(row=0, column=1, rowspan=2, padx=10, pady=10)

        def _reload_inventory(self) -> None:
            self.inventory = load_inventory()
            self._refresh_inventory_rows()
            self.status_label.configure(text="Stock recharge depuis data/inventory.csv")

        def _fill_article(self, article: Article) -> None:
            self.query_entry.delete(0, "end")
            self.query_entry.insert(0, article.id_unique)
            self.quantity_var.set("1")
            self.status_label.configure(text=f"Article selectionne: {article.nom}")

        def _change_quantity(self, delta: int) -> None:
            try:
                current = int(self.quantity_var.get())
            except ValueError:
                current = 1
            self.quantity_var.set(str(max(1, current + delta)))

        def _simulate_selection(self) -> None:
            query = self.query_entry.get().strip()
            try:
                quantity = int(self.quantity_var.get())
            except ValueError:
                quantity = 0

            result = prepare_selection(self.inventory, query, quantity)
            self.last_selected_article = result.article
            self.last_image_path = None
            self.open_image_button.configure(state="disabled")

            lines = list(result.messages)
            if result.ok and result.article is not None:
                image_path = resolve_project_path(result.article.image_path)
                self.last_image_path = image_path
                self.open_image_button.configure(state="normal")

                vision_result = process_image(image_path)
                lines.extend(
                    (
                        "",
                        "POINT D'INTEGRATION VISION",
                        "Appel prevu: vision.process_image(image_path)",
                        f"Statut actuel: {vision_result.message}",
                    )
                )

            self.status_label.configure(
                text="Selection validee" if result.ok else f"Erreur: {result.code}"
            )
            self._set_text(self.result_text, "\n".join(lines))

        def _open_selected_image(self) -> None:
            if self.last_image_path is None:
                return
            open_path(self.last_image_path)

        def _prepare_llm_context(self) -> None:
            symptoms = self.symptoms_text.get("1.0", "end").strip()
            context = build_pharmacist_context(symptoms, self.inventory)
            self._set_text(self.assistant_output, context.as_text())

        @staticmethod
        def _set_text(widget: "ctk.CTkTextbox", text: str) -> None:
            widget.configure(state="normal")
            widget.delete("1.0", "end")
            widget.insert("1.0", text)
            widget.configure(state="disabled")


def main() -> int:
    if ctk is None:
        print(
            "Interface desktop indisponible: Tkinter ou CustomTkinter manque. "
            "Sur le cloud, lancez plutot: python start_cloud.py. "
            "En local desktop: python -m pip install -r requirements-desktop.txt"
        )
        return 1

    app = StockVisionApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
