import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

from app import find_article, load_inventory, prepare_selection
from cloud_app import (
    article_to_payload,
    parse_request_payload,
    render_page,
    run_selection,
    selection_to_payload,
)
from llm_assistant import build_pharmacist_context
from vision import VisionResult, process_image


class StockSimulationTest(unittest.TestCase):
    def setUp(self):
        self.inventory = load_inventory()

    def test_find_article_by_id(self):
        article = find_article(self.inventory, "MED-001")

        self.assertIsNotNone(article)
        self.assertEqual(article.nom, "Paracetamol 500mg")

    def test_find_article_by_unique_partial_name(self):
        article = find_article(self.inventory, "amoxicilline")

        self.assertIsNotNone(article)
        self.assertEqual(article.id_unique, "MED-003")

    def test_prepare_selection_success(self):
        result = prepare_selection(self.inventory, "MED-002", 2)

        self.assertTrue(result.ok)
        self.assertEqual(result.code, "SELECTION_OK")
        self.assertIn("Robot allant au compartiment: Rayon A2", result.messages)

    def test_prepare_selection_rejects_unknown_article(self):
        result = prepare_selection(self.inventory, "Produit inexistant", 1)

        self.assertFalse(result.ok)
        self.assertEqual(result.code, "ARTICLE_INTROUVABLE")

    def test_prepare_selection_rejects_insufficient_stock(self):
        result = prepare_selection(self.inventory, "MED-003", 999)

        self.assertFalse(result.ok)
        self.assertEqual(result.code, "STOCK_INSUFFISANT")

    def test_prepare_selection_rejects_invalid_quantity(self):
        result = prepare_selection(self.inventory, "MED-003", 0)

        self.assertFalse(result.ok)
        self.assertEqual(result.code, "QUANTITE_INVALIDE")

    def test_vision_rejects_unsupported_image_format_before_model_load(self):
        result = process_image("images/paracetamol_500mg.svg")

        self.assertTrue(result.enabled)
        self.assertFalse(result.ok)
        self.assertIn("Format non compatible", result.message)

    def test_llm_context_keeps_pharmacist_validation(self):
        context = build_pharmacist_context("fievre et douleur", self.inventory)
        text = context.as_text()

        self.assertIn("assistant pour pharmacien", text)
        self.assertIn("fievre et douleur", text)
        self.assertIn("Ne pas delivrer automatiquement", text)

    def test_gui_can_fail_gracefully_without_tkinter(self):
        import gui

        with mock.patch.object(gui, "ctk", None), redirect_stdout(StringIO()):
            self.assertEqual(gui.main(), 1)

    def test_cloud_page_renders_without_desktop_gui(self):
        html = render_page()

        self.assertIn("Gestion Stock Vision", html)
        self.assertIn("Assistant pharmacien LLM", html)
        self.assertIn("MED-001", html)

    def test_cloud_api_selection_payload(self):
        stub_vision = VisionResult(
            enabled=True,
            ok=True,
            message="Vision OK test.",
        )
        with mock.patch("cloud_app.process_image", return_value=stub_vision):
            result, vision_result = run_selection("MED-001", 2)
        payload = selection_to_payload(result, vision_result)

        self.assertTrue(payload["ok"])
        self.assertEqual(payload["code"], "SELECTION_OK")
        self.assertIsNotNone(payload["vision"])

    def test_cloud_payload_parses_json_and_form(self):
        json_payload = parse_request_payload(
            "application/json",
            b'{"query":"MED-001","quantity":2}',
        )
        form_payload = parse_request_payload(
            "application/x-www-form-urlencoded",
            b"query=MED-002&quantity=3",
        )

        self.assertEqual(json_payload["query"], "MED-001")
        self.assertEqual(form_payload["quantity"], "3")

    def test_cloud_inventory_payload_has_image_url(self):
        payload = article_to_payload(self.inventory[0])

        self.assertTrue(str(payload["image_url"]).startswith("/dataset-images/train/images/"))


if __name__ == "__main__":
    unittest.main()
