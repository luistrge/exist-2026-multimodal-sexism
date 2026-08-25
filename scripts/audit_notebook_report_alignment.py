"""Verify that curated notebook evidence agrees with the final technical report.

This audit intentionally checks both executable source and preserved text outputs.
It also rejects the known superseded Task 2.2 cascade distribution.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks"


def load(name: str) -> dict:
    return json.loads((NOTEBOOK_DIR / name).read_text(encoding="utf-8"))


def source_text(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def output_text(notebook: dict) -> str:
    chunks: list[str] = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            chunks.extend(output.get("text", []))
            data = output.get("data", {})
            chunks.extend(data.get("text/plain", []))
    return "\n".join(chunks)


def require(haystack: str, needle: str, claim: str, passes: list[str]) -> None:
    if needle not in haystack:
        raise AssertionError(f"Missing notebook evidence for {claim!r}: {needle!r}")
    passes.append(claim)


def require_absent(haystack: str, needle: str, claim: str, passes: list[str]) -> None:
    if needle in haystack:
        raise AssertionError(f"Superseded evidence reintroduced for {claim!r}: {needle!r}")
    passes.append(claim)


def main() -> None:
    overview = load("00_project_overview.ipynb")
    eda = load("01_eda_and_sensor_analysis.ipynb")
    task21 = load("02_task21_binary_gate.ipynb")
    task22 = load("03_task22_source_intention.ipynb")
    task23 = load("04_task23_sexism_facets.ipynb")

    overview_source = source_text(overview)
    eda_all = source_text(eda) + output_text(eda)
    task21_source, task21_output = source_text(task21), output_text(task21)
    task22_source, task22_output = source_text(task22), output_text(task22)
    task23_source, task23_output = source_text(task23), output_text(task23)
    passes: list[str] = []

    for token, claim in [
        ("3,984", "training size"),
        ("1,053", "test size"),
        ("2,005 English", "English training count"),
        ("1,979 Spanish", "Spanish training count"),
        ("608 `YES`, 445 `NO`", "Task 2.1 final distribution"),
        ("396 `DIRECT`, 212 `JUDGEMENTAL`", "Task 2.2 final distribution"),
        ("420 stereotyping/dominance", "Task 2.3 final facet distribution"),
    ]:
        require(overview_source, token, claim, passes)

    for token, claim in [
        ("annotator disagreement", "EDA disagreement analysis"),
        ("physiological sensor block", "EDA physiological analysis"),
        ("Paper-guided sensor analysis", "EDA interpretable sensor axes"),
        ("Cross-modal relationships", "EDA cross-modal audit"),
    ]:
        require(eda_all, token, claim, passes)

    task21_members = [
        "text+image_intfloat_multilingual_e5_base_openai_clip_vit_base_patch32_lr_c1",
        "text+image_intfloat_multilingual_e5_base_openai_clip_vit_base_patch32_xgboost",
        "transformer_en_cardiffnlp_twitter_roberta_base_offensive",
        "text+image+sensor_intfloat_multilingual_e5_base_openai_clip_vit_base_patch32_lightgbm",
        "text+image_intfloat_multilingual_e5_base_facebook_dinov2_base_lr_c3",
        "text+image_intfloat_multilingual_e5_base_openai_clip_vit_base_patch32_lightgbm",
        "text+image_intfloat_multilingual_e5_base_facebook_dinov2_base_lr_c1",
        "text+image_intfloat_multilingual_e5_base_openai_clip_vit_base_patch32_lr_c3",
    ]
    for index, member in enumerate(task21_members, 1):
        require(task21_output, member, f"Task 2.1 retained member {index}", passes)
    for token, claim in [
        ("Train/dev: (2696, 976) (674, 976)", "Task 2.1 split"),
        ("0.709011", "Task 2.1 macro-F1"),
        ("0.815217", "Task 2.1 YES F1"),
        ("0.747774", "Task 2.1 accuracy"),
        ("0.371642", "Task 2.1 threshold"),
        ("YES       0.72      0.94      0.82       401", "Task 2.1 YES recall"),
        ("gold_NO       129       144", "Task 2.1 NO confusion row"),
        ("gold_YES       26       375", "Task 2.1 YES confusion row"),
    ]:
        require(task21_output, token, claim, passes)
    require(task21_source, "expected_task21", "Task 2.1 report-distribution guard", passes)

    for token, claim in [
        ("fit:   n=1111", "Task 2.2 fit size"),
        ("calib: n= 314", "Task 2.2 calibration size"),
        ("dev:   n= 357", "Task 2.2 development size"),
        ("Candidatos totales:   37", "Task 2.2 candidate count"),
        ("Candidatos robustos:  31", "Task 2.2 stable-pool count"),
        ("geom_mean_top5", "Task 2.2 selected ensemble"),
        ("0.636674", "Task 2.2 conditional macro-F1"),
        ("0.468750", "Task 2.2 JUDGEMENTAL F1"),
        ("0.484744", "Task 2.2 routed diagnostic"),
        ("n_lost_sexist_to_NO", "Task 2.2 routing-loss field"),
        ("114", "Task 2.2 lost sexist examples"),
        ("VLM_WordChar_LR", "Task 2.2 VLM LR member"),
        ("TransformerV2_xlm-roberta-base_qwen_rich_e4_L256_lr1.5e-05_soft", "Task 2.2 XLM-R/Qwen member"),
        ("VLM_WordChar_ComplementNB", "Task 2.2 VLM ComplementNB member"),
        ("TransformerV2_distilbert-base-multilingual-cased_ocr_lang_e4_L256_lr3e-05_soft", "Task 2.2 DistilBERT member"),
        ("Text_WordChar_ComplementNB_a0.2", "Task 2.2 OCR anchor"),
    ]:
        require(task22_output, token, claim, passes)
    require(task22_source, "expected_task22", "Task 2.2 report-distribution guard", passes)
    require(
        task22_source,
        "This is not the selected Task 2.1 submission system.",
        "Task 2.2 auxiliary-gate scope disclosure",
        passes,
    )
    require_absent(task22_output, "hard={'DIRECT': 493, 'NO': 465", "superseded Task 2.2 distribution", passes)
    require_absent(task22_output, "Task 2.1 dev macro-F1: 0.7779", "superseded auxiliary-gate headline", passes)

    task23_members = [
        "text+image_intfloat_multilingual_e5_base_openai_clip_vit_base_patch32_mlp",
        "transformer_en_cardiffnlp_twitter_roberta_base_offensive",
        "text+image_intfloat_multilingual_e5_base_openai_clip_vit_base_patch32_ovr_lr_c1",
        "transformer_es_dccuchile_bert_base_spanish_wwm_cased",
        "text_intfloat_multilingual_e5_base_no_image_ovr_lr_c3",
        "text+image_intfloat_multilingual_e5_base_openai_clip_vit_base_patch32_ovr_lr_c3",
        "text+image_intfloat_multilingual_e5_base_openai_clip_vit_base_patch32_chain_lr_c1",
        "text_intfloat_multilingual_e5_base_no_image_ovr_lr_c1",
    ]
    for index, member in enumerate(task23_members, 1):
        require(task23_output, member, f"Task 2.3 retained member {index}", passes)
    for token, claim in [
        ("Filas Task 2.3 v2 con al menos una faceta: 1990 de 3984", "Task 2.3 positive population"),
        ("Train/dev: (1592, 973) (398, 973)", "Task 2.3 split"),
        ("0.67711", "Task 2.3 facet macro-F1"),
        ("0.700056", "Task 2.3 micro-F1"),
        ("0.693858", "Task 2.3 samples-F1"),
        ("0.71      0.89      0.79       168", "Task 2.3 ideological-inequality metrics"),
        ("0.69      0.89      0.78       187", "Task 2.3 objectification metrics"),
        ("0.58      0.90      0.70       200", "Task 2.3 stereotyping-dominance metrics"),
        ("0.60      0.75      0.67        96", "Task 2.3 sexual-violence metrics"),
        ("0.33      0.71      0.45        78", "Task 2.3 rare-facet metrics"),
    ]:
        require(task23_output, token, claim, passes)
    require(task23_source, "expected_task23", "Task 2.3 report-distribution guard", passes)
    require(task23_source, "EXIST2026_EXPORT_FINAL_SUBMISSIONS", "Task 2.3 opt-in final export", passes)

    print(f"Notebook/report alignment audit passed: {len(passes)} checks.")
    for claim in passes:
        print(f"  [PASS] {claim}")


if __name__ == "__main__":
    main()
