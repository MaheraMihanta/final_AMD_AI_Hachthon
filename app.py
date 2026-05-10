from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INVENTORY_PATH = BASE_DIR / "data" / "inventory.csv"


@dataclass(frozen=True)
class Article:
    id_unique: str
    nom: str
    classe_therapeutique: str
    emplacement_rayon: str
    quantite_stock: int
    prix: float
    image_path: Path

    @property
    def prix_formate(self) -> str:
        return f"{self.prix:,.0f} Ar".replace(",", " ")


@dataclass(frozen=True)
class SelectionResult:
    ok: bool
    code: str
    messages: tuple[str, ...]
    article: Article | None = None


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def normalize(value: str) -> str:
    return " ".join(value.casefold().strip().split())


def load_inventory(path: str | Path = DEFAULT_INVENTORY_PATH) -> list[Article]:
    inventory_path = resolve_project_path(path)
    articles: list[Article] = []

    with inventory_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_fields = {
            "id_unique",
            "nom",
            "classe_therapeutique",
            "emplacement_rayon",
            "quantite_stock",
            "prix",
            "image_path",
        }
        missing_fields = required_fields.difference(reader.fieldnames or [])
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Inventaire invalide: colonnes manquantes: {missing}")

        for line_number, row in enumerate(reader, start=2):
            try:
                articles.append(
                    Article(
                        id_unique=row["id_unique"].strip(),
                        nom=row["nom"].strip(),
                        classe_therapeutique=row["classe_therapeutique"].strip(),
                        emplacement_rayon=row["emplacement_rayon"].strip(),
                        quantite_stock=int(row["quantite_stock"]),
                        prix=float(row["prix"]),
                        image_path=Path(row["image_path"].strip()),
                    )
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Inventaire invalide a la ligne {line_number}: {exc}"
                ) from exc

    return articles


def find_article(inventory: list[Article], query: str) -> Article | None:
    normalized_query = normalize(query)
    if not normalized_query:
        return None

    for article in inventory:
        if normalize(article.id_unique) == normalized_query:
            return article

    for article in inventory:
        if normalize(article.nom) == normalized_query:
            return article

    partial_matches = [
        article for article in inventory if normalized_query in normalize(article.nom)
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]

    return None


def prepare_selection(
    inventory: list[Article], query: str, quantity: int = 1
) -> SelectionResult:
    if quantity <= 0:
        return SelectionResult(
            ok=False,
            code="QUANTITE_INVALIDE",
            messages=("La quantite demandee doit etre superieure a zero.",),
        )

    article = find_article(inventory, query)
    if article is None:
        return SelectionResult(
            ok=False,
            code="ARTICLE_INTROUVABLE",
            messages=(f"Aucun article ne correspond a la requete: {query}",),
        )

    if article.quantite_stock <= 0:
        return SelectionResult(
            ok=False,
            code="RUPTURE_STOCK",
            article=article,
            messages=(
                f"Article trouve: {article.nom} ({article.id_unique})",
                "Statut: rupture de stock.",
            ),
        )

    if article.quantite_stock < quantity:
        return SelectionResult(
            ok=False,
            code="STOCK_INSUFFISANT",
            article=article,
            messages=(
                f"Article trouve: {article.nom} ({article.id_unique})",
                f"Stock insuffisant: {article.quantite_stock} disponible(s), "
                f"{quantity} demande(s).",
            ),
        )

    absolute_image_path = resolve_project_path(article.image_path)
    if not absolute_image_path.exists():
        return SelectionResult(
            ok=False,
            code="IMAGE_MANQUANTE",
            article=article,
            messages=(
                f"Article trouve: {article.nom} ({article.id_unique})",
                f"Image associee introuvable: {article.image_path}",
            ),
        )

    return SelectionResult(
        ok=True,
        code="SELECTION_OK",
        article=article,
        messages=(
            f"Article trouve: {article.nom} ({article.id_unique})",
            f"Classe therapeutique: {article.classe_therapeutique}",
            f"Quantite demandee: {quantity}",
            f"Stock actuel: {article.quantite_stock}",
            f"Prix unitaire: {article.prix_formate}",
            f"Image selectionnee: {article.image_path}",
            f"Robot allant au compartiment: {article.emplacement_rayon}",
            "Etape 1 terminee: pret pour le module 3 (vision par ordinateur).",
        ),
    )


def print_inventory(inventory: list[Article]) -> None:
    print("Inventaire disponible")
    print("-" * 80)
    for article in inventory:
        print(
            f"{article.id_unique:8} | {article.nom:28} | "
            f"{article.emplacement_rayon:8} | stock: {article.quantite_stock:3} | "
            f"{article.prix_formate}"
        )


def run_cli() -> int:
    parser = argparse.ArgumentParser(
        description="Simulation des modules 1 et 2: stock + selection d'image."
    )
    parser.add_argument(
        "-q",
        "--query",
        help="Nom ou id_unique de l'article a rechercher.",
    )
    parser.add_argument(
        "-n",
        "--quantity",
        type=int,
        default=1,
        help="Quantite demandee pour la simulation.",
    )
    parser.add_argument(
        "--inventory",
        default=str(DEFAULT_INVENTORY_PATH),
        help="Chemin du fichier CSV d'inventaire.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Afficher l'inventaire puis quitter.",
    )
    args = parser.parse_args()

    inventory = load_inventory(args.inventory)

    if args.list:
        print_inventory(inventory)
        return 0

    query = args.query
    if not query:
        query = input("Nom ou ID de l'article: ").strip()

    result = prepare_selection(inventory, query, args.quantity)
    print("\n".join(result.messages))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(run_cli())
