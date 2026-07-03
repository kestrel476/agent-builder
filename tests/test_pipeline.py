"""End-to-end and unit tests — all run offline (deterministic stub brain)."""

from __future__ import annotations

import json

import pytest

from forge.archetypes import all_archetypes, get
from forge.clarify import ClarifyEngine
from forge.config import ForgeConfig
from forge.integrations import build_llm
from forge.integrations.stub_brain import generic_fill
from forge.model import AgentBlueprint, FieldSpec, IOContract, Stage, TestCase
from forge.pipeline import Forge
from forge.runtime import AgentInput, AgentSpec, run_agent
from forge.runtime.validation import validate
from forge.store import WorkspaceManager

CONTRACT = (
    "This Agreement is made between Acme Corp and Beta LLC, effective January 5, 2024, "
    "total USD 50,000, governed by the laws of Delaware."
)


@pytest.fixture
def forge():
    return Forge(offline=True)


@pytest.fixture
def wm(tmp_path):
    return WorkspaceManager(ForgeConfig.load(home=tmp_path / ".forge", offline=True))


# --------------------------------------------------------------------------- #
def test_archetypes_registered():
    keys = {a.key for a in all_archetypes()}
    assert {"json_extraction", "classification", "risk_check", "clause_comparison",
            "doc_generation", "chat_assistant", "workflow"} <= keys


def test_stub_extraction_offline():
    schema = {"type": "object", "properties": {
        "parties": {"type": "array"}, "effective_date": {"type": "string"},
        "governing_law": {"type": "string"}, "total_amount": {"type": "string"}}}
    out = generic_fill(schema, CONTRACT, {})
    assert out["parties"] == ["Acme Corp", "Beta LLC"]
    assert out["governing_law"] == "Delaware"
    assert "50,000" in out["total_amount"]


def test_validation_catches_type_and_required():
    schema = {"type": "object", "properties": {"a": {"type": "string"}},
              "required": ["a"], "additionalProperties": False}
    assert validate(schema, {"a": "x"}) == []
    assert any("обязательное поле отсутствует" in e for e in validate(schema, {}))
    assert any("ожидался тип string" in e for e in validate(schema, {"a": 5}))
    assert any("непредусмотренное поле" in e for e in validate(schema, {"a": "x", "b": 1}))


def test_intake_flags_ambiguity_not_guess(forge):
    bp = AgentBlueprint(name="x", slug="x", archetype="json_extraction",
                        instructions_raw="Extract relevant fields where applicable. Output JSON.")
    forge.intake(bp)
    assert bp.stage == Stage.INTAKE
    assert bp.instructions.ambiguities  # vague wording captured, not resolved
    assert bp.instructions.missing_data


def test_critics_and_blocking_issues(forge):
    bp = AgentBlueprint(name="x", slug="x", archetype="json_extraction", instructions_raw="do stuff")
    forge.intake(bp)
    forge.analyze(bp, use_llm=False)
    # missing inputs + success criteria are blockers
    blockers = [i for i in bp.blocking_issues]
    assert any(i.where == "io.inputs" for i in blockers)
    assert any(i.where == "io.success_criteria" for i in blockers)


def test_assume_open_records_explicit_assumptions(forge):
    bp = AgentBlueprint(name="x", slug="x", archetype="json_extraction", instructions_raw="do stuff")
    forge.intake(bp)
    forge.analyze(bp, use_llm=False)
    n_blockers = len(bp.blocking_issues)
    made = forge.assume_open(bp, only_blocking=True)
    assert len(made) >= n_blockers
    assert not bp.blocking_issues  # all resolved via assumptions
    assert all(a.confirmed is None for a in bp.assumptions)  # unconfirmed by default


def test_example_conflict_detected(forge):
    bp = AgentBlueprint(name="x", slug="x", archetype="json_extraction", instructions_raw="extract parties")
    forge.intake(bp)
    forge.build_contract(bp)
    bp.test_cases.append(TestCase(id="T-1", name="bad", input_text="x",
                                  expected={"nonexistent_field": 1}))
    eng = ClarifyEngine(build_llm(offline=True))
    eng.analyze(bp, use_llm=False)
    assert any(i.kind.value == "example_conflict" for i in bp.issues)


def test_full_pipeline_offline(forge, wm):
    ws = wm.create("Contract Extractor")
    bp = AgentBlueprint(name="Contract Extractor", slug=ws.slug, archetype="json_extraction",
                        instructions_raw="Extract parties, effective date, governing law from contracts. Output JSON.")
    forge.intake(bp)
    forge.analyze(bp, use_llm=False)
    forge.assume_open(bp)
    forge.build_contract(bp)
    from forge.model import FieldSpec
    bp.io.fields.append(FieldSpec(name="governing_law", type="string"))
    forge.synthesize(bp, ws)
    assert ws.agent_dir.joinpath("agent.yaml").is_file()
    assert ws.agent_dir.joinpath("contract.schema.json").is_file()

    bp.test_cases.append(TestCase(
        id="T-1", name="acme", input_text=CONTRACT,
        expected={"parties": ["Acme Corp", "Beta LLC"], "governing_law": "Delaware"}))
    report = forge.test(bp, ws)
    assert report.green, report.rows()

    ws.save(bp)
    forge.package(bp, ws, report)
    assert bp.stage == Stage.PACKAGED
    assert ws.spec_path.is_file()
    assert (ws.root / "HANDOFF.md").is_file()
    # blueprint reloads cleanly
    assert wm.open(ws.slug).load().name == "Contract Extractor"


def test_generated_bundle_runs(forge, wm):
    ws = wm.create("runner")
    bp = AgentBlueprint(name="runner", slug=ws.slug, archetype="json_extraction",
                        instructions_raw="extract parties and effective date")
    forge.intake(bp)
    forge.build_contract(bp)
    forge.synthesize(bp, ws)
    spec = AgentSpec.load(ws.agent_dir)
    res = run_agent(spec, AgentInput(text=CONTRACT), llm=forge.llm)
    assert res.status in ("ok", "low_confidence", "partial")
    assert res.output["parties"][0] == "Acme Corp"


def test_test_and_refine_loop_converges(forge, wm):
    ws = wm.create("refiner")
    bp = AgentBlueprint(name="refiner", slug=ws.slug, archetype="json_extraction",
                        instructions_raw="extract parties")
    forge.intake(bp)
    forge.build_contract(bp)
    forge.synthesize(bp, ws)
    bp.test_cases.append(TestCase(id="T-1", name="ok", input_text=CONTRACT,
                                  expected={"parties": ["Acme Corp", "Beta LLC"]}))
    history = forge.test_and_refine(bp, ws, max_iters=3)
    assert history[-1][1].green


def test_dry_run_skips_llm(forge, wm):
    ws = wm.create("dry")
    bp = AgentBlueprint(name="dry", slug=ws.slug, archetype="json_extraction",
                        instructions_raw="extract parties")
    forge.intake(bp)
    forge.build_contract(bp)
    forge.synthesize(bp, ws)
    res = forge.run_agent_bundle(ws, "some text", dry_run=True)
    assert res.status == "ok"
    assert res.output.get("_dry_run") is True


def test_classification_archetype_contract(forge, wm):
    ws = wm.create("classifier")
    bp = AgentBlueprint(name="classifier", slug=ws.slug, archetype="classification",
                        instructions_raw="Classify documents as NDA, MSA or Other.")
    forge.intake(bp)
    forge.build_contract(bp)
    names = {f.name for f in bp.io.fields}
    assert "label" in names
    assert bp.io.output_kind.value == "classification"


# --------------------------------------------------------------------------- #
# Regression tests for bugs found during live GigaChat debugging
# --------------------------------------------------------------------------- #
def test_json_extraction_has_no_baked_in_fields():
    """A generic extraction archetype must not ship domain-specific default fields."""
    assert get("json_extraction").default_fields == ()


def test_confidence_is_required_in_schema():
    io = IOContract(fields=[FieldSpec(name="x", type="string", required=True)], confidence_required=True)
    schema = io.output_schema()
    assert "confidence" in schema["required"]


def test_array_of_objects_schema_and_validation():
    f = FieldSpec(name="findings", type="array", required=True, item_fields=[
        FieldSpec(name="condition", type="string", required=True),
        FieldSpec(name="present", type="boolean", required=True),
    ])
    items = f.to_schema()["items"]
    assert items["type"] == "object" and "condition" in items["properties"]
    schema = {"type": "object", "properties": {"findings": f.to_schema()}, "required": ["findings"]}
    assert validate(schema, {"findings": [{"condition": "x", "present": True}]}) == []
    assert validate(schema, {"findings": [{"present": True}]})  # missing required 'condition'


def test_enum_validation_is_case_insensitive():
    schema = {"type": "object", "properties": {"label": {"type": "string", "enum": ["доверенность", "иное"]}}}
    assert validate(schema, {"label": "Доверенность"}) == []   # casing differs — still valid
    assert validate(schema, {"label": "договор"})              # not in enum — error


def test_agent_input_handles_long_text_not_a_path():
    long_text = "СЧЁТ " * 100  # far longer than any real filename
    inp = AgentInput.from_value(long_text)
    assert inp.text == long_text and not inp.files


def test_parse_loose_handles_python_call_style():
    from forge.integrations.llm import _parse_loose
    sch = {"type": "object", "properties": {"situations": {"type": "array"}}}
    # GigaChat иногда пишет вызов как python-код вместо function call:
    assert _parse_loose("emit_result(situations=['Корпоративный контроль', 'Залог'])", sch) == \
        {"situations": ["Корпоративный контроль", "Залог"]}
    assert _parse_loose('{"situations": ["a", "b"]}', sch) == {"situations": ["a", "b"]}
    sch2 = {"type": "object", "properties": {"detected_risks": {"type": "array"}}}
    out = _parse_loose("emit_result(detected_risks=[{'risk_code': 'X', 'title': 'Y'}])", sch2)
    assert out["detected_risks"] == [{"risk_code": "X", "title": "Y"}]


def test_offline_embeddings_are_deterministic_and_lexical():
    a, b, c = build_llm(offline=True).embed(["залог судна", "корпоративный контроль", "залог морского судна"])

    def cos(x, y):
        return sum(i * j for i, j in zip(x, y))

    assert len(a) == 256
    assert cos(a, c) > cos(a, b)  # связанные ближе, чем несвязанные
    assert build_llm(offline=True).embed(["x"])[0][:4] == build_llm(offline=True).embed(["x"])[0][:4]


def test_knowledge_base_loads_and_filters(tmp_path):
    from forge.runtime.knowledge import KnowledgeBase, KnowledgeConfig
    cat = {"R1": {"title": "Залог судна", "fact": "залог судна", "risk_level": "Очевидный риск"},
           "R2": {"title": "Прочее", "fact": "x", "risk_level": "Высокий"}}
    (tmp_path / "cat.json").write_text(json.dumps(cat, ensure_ascii=False), encoding="utf-8")
    cfg = KnowledgeConfig(file="cat.json", filter={"risk_level": "Очевидный риск"})
    kb = KnowledgeBase.load(tmp_path, cfg)
    assert [e.code for e in kb.entries] == ["R1"]  # отфильтровано по категории
    assert kb.get("R1").title == "Залог судна"


def test_catalog_rag_archetype_offline_detects(forge, wm, tmp_path):
    cat = {"R1": {"title": "Залог судна", "fact": "оформление залога морского судна",
                  "risk_level": "Очевидный риск"},
           "R2": {"title": "Прочее", "fact": "нерелевант", "risk_level": "Высокий"}}
    ws = wm.create("ragdet")
    (ws.root / "cat.json").write_text(json.dumps(cat, ensure_ascii=False), encoding="utf-8")
    bp = AgentBlueprint(name="ragdet", slug=ws.slug, archetype="catalog_risk_detection",
                        instructions_raw="детект рисков по каталогу")
    bp.knowledge = {"source": str(ws.root / "cat.json"), "id_field": "__key__",
                    "title_field": "title", "fact_field": "fact",
                    "filter": {"risk_level": "Очевидный риск"}, "top_k": 10}
    forge.intake(bp)
    forge.build_contract(bp)
    forge.synthesize(bp, ws)
    res = forge.run_agent_bundle(ws, "Договор предусматривает оформление в залог морского судна сроком на год.")
    codes = {d["risk_code"] for d in res.output["detected_risks"]}
    assert "R1" in codes  # релевантный риск из категории найден
    assert "R2" not in codes  # вне категории — отфильтрован


def test_rule_engine_matchers():
    from forge.runtime.rules import RuleEngine
    text = "Договор автоматически продлевается на год. Иванов лично гарантирует. Неустойка 5% в день."
    eng = RuleEngine.from_list([
        {"id": "A", "match": "near", "terms": ["автоматически", "продлевается"], "window": 60},
        {"id": "B", "match": "all", "terms": ["лично", "гарантирует"]},
        {"id": "C", "match": "regex", "pattern": r"неустойк\w+\s+\d+%"},
        {"id": "D", "match": "any", "terms": ["блокчейн", "крипта"]},
        {"id": "E", "match": "any", "terms": ["блокчейн"], "negate": True},
    ])
    hit = {f.rule_id for f in eng.evaluate(text)}
    assert hit == {"A", "B", "C", "E"}  # D отсутствует; E сработало через negate


def test_parse_loose_strips_think_blocks():
    from forge.integrations.llm import _parse_loose
    sch = {"type": "object", "properties": {"situations": {"type": "array"}}}
    raw = "<think>надо подумать...</think>\nemit_result(situations=['Залог', 'Поручительство'])"
    assert _parse_loose(raw, sch) == {"situations": ["Залог", "Поручительство"]}


def test_rule_check_archetype_offline(forge, wm):
    ws = wm.create("rc")
    bp = AgentBlueprint(name="rc", slug=ws.slug, archetype="rule_check", instructions_raw="проверка")
    bp.rule_catalog = [
        {"id": "AUTO", "title": "Автопролонгация", "severity": "high", "match": "near",
         "terms": ["автоматически", "продлевается"], "window": 80},
        {"id": "ABSENT", "match": "any", "terms": ["блокчейн"]},
    ]
    forge.intake(bp)
    forge.build_contract(bp)
    forge.synthesize(bp, ws)
    res = forge.run_agent_bundle(ws, "Договор автоматически продлевается на год.")
    ids = {m["rule_id"] for m in res.output["matched_rules"]}
    assert ids == {"AUTO"}  # детерминированно, без LLM


def test_document_package_offline(forge, wm, tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "ustav.txt").write_text("УСТАВ общества РОМАШКА. Уставный капитал общество.", encoding="utf-8")
    (pkg / "prko.txt").write_text("ПРОЕКТ РЕШЕНИЯ. Заёмщик обязан предоставить залог банку.", encoding="utf-8")
    ws = wm.create("pkg")
    bp = AgentBlueprint(name="pkg", slug=ws.slug, archetype="document_package", instructions_raw="пакет")
    bp.package = {"taxonomy": [
        {"code": "ustav", "title": "Устав", "description": "устав общество уставный капитал"},
        {"code": "prko", "title": "ПРКО", "description": "проект решение залог заёмщик банк"}],
        "required": ["ustav", "prko", "doverennost"], "per_type": {}}
    forge.intake(bp)
    forge.build_contract(bp)
    forge.synthesize(bp, ws)
    res = forge.run_agent_bundle(ws, str(pkg))
    types = {d["doc_type"] for d in res.output["documents"]}
    assert {"ustav", "prko"} <= types
    assert res.output["verdict"] == "rejected"           # отсутствует обязательный тип
    assert "doverennost" in res.output["missing_required"]


def test_checklist_archetype_offline(forge, wm):
    ws = wm.create("chk")
    bp = AgentBlueprint(name="chk", slug=ws.slug, archetype="checklist", instructions_raw="экспертиза по чек-листу")
    bp.checklist = [
        {"id": "C1", "question": "Есть ли залог судна?", "criteria": "оформление залога морского судна",
         "severity": "high", "on_yes_position": "Выявлены правовые риски",
         "on_yes_finding": "Риск залога судна.", "on_yes_recommendation": "Проверить регистрацию."},
        {"id": "C2", "question": "Есть ли криптовалюта?", "criteria": "блокчейн криптовалюта токены",
         "severity": "low"},
    ]
    forge.intake(bp)
    forge.build_contract(bp)
    forge.synthesize(bp, ws)
    res = forge.run_agent_bundle(ws, "Договор предусматривает оформление залога морского судна.")
    assert len(res.output["results"]) == 2  # полный аудит по всем проверкам
    assert "C1" in res.output["flagged"] and "C2" not in res.output["flagged"]
    c1 = next(r for r in res.output["results"] if r["check_id"] == "C1")
    assert c1["verdict"] == "yes" and c1["position"] == "Выявлены правовые риски"  # вывод из шаблона
    assert res.output["overall_position"] == "Выявлены правовые риски"


def test_map_concurrent_preserves_order_and_isolates_failures():
    from forge.runtime.executor import _map_concurrent
    out = _map_concurrent([1, 2, 3, 4], lambda x: x * 10)
    assert out == [10, 20, 30, 40]  # порядок сохранён

    def fn(x):
        if x == 2:
            raise ValueError("boom")
        return x

    res = _map_concurrent([1, 2, 3], fn)
    assert res == [1, None, 3]  # сбой одного элемента → None, остальные целы


def test_run_records_llm_stats(forge, wm):
    ws = wm.create("stats")
    bp = AgentBlueprint(name="stats", slug=ws.slug, archetype="json_extraction",
                        instructions_raw="извлечь стороны")
    forge.intake(bp)
    forge.build_contract(bp)
    forge.synthesize(bp, ws)
    res = forge.run_agent_bundle(ws, CONTRACT)
    assert "llm_calls" in res.stats and res.stats["backend"] == "stub"
    assert res.to_dict()["stats"]["llm_calls"] >= 1


def test_compare_runs_with_two_files(forge, wm, tmp_path):
    from forge.runtime import AgentInput, AgentSpec, run_agent
    ws = wm.create("cmp")
    bp = AgentBlueprint(name="cmp", slug=ws.slug, archetype="clause_comparison",
                        instructions_raw="сравнить две редакции")
    forge.intake(bp)
    forge.build_contract(bp)
    forge.synthesize(bp, ws)
    a = tmp_path / "a.txt"
    a.write_text("Редакция А: срок 30 дней.", encoding="utf-8")
    b = tmp_path / "b.txt"
    b.write_text("Редакция Б: срок 10 дней.", encoding="utf-8")
    spec = AgentSpec.load(ws.agent_dir)
    res = run_agent(spec, AgentInput(files=[str(a), str(b)]), llm=forge.llm)
    assert res.status != "error"               # compare получил оба документа
    assert "differences" in res.output


def test_workflow_emits_declared_checks(forge, wm):
    ws = wm.create("wf")
    bp = AgentBlueprint(name="wf", slug=ws.slug, archetype="workflow",
                        instructions_raw="кейс")
    forge.intake(bp)
    bp.instructions.business_rules = ["Правило 1", "Правило 2"]  # после intake (он перезаписывает)
    forge.build_contract(bp)
    forge.synthesize(bp, ws)
    res = forge.run_agent_bundle(ws, "Материалы дела: договор и решение.")
    assert isinstance(res.output, dict)
    assert [c["rule"] for c in res.output["checks"]] == ["Правило 1", "Правило 2"]


def test_knowledge_index_cache_keyed_by_backend(wm, tmp_path):
    from forge.integrations import build_llm
    from forge.runtime.knowledge import KnowledgeBase, KnowledgeConfig
    cat = {"R1": {"title": "Залог", "fact": "залог судна"}}
    (tmp_path / "cat.json").write_text(json.dumps(cat, ensure_ascii=False), encoding="utf-8")
    kb = KnowledgeBase.load(tmp_path, KnowledgeConfig(file="cat.json"))
    kb.ensure_index(build_llm(offline=True))
    assert (tmp_path / "knowledge" / "index-stub.npz").is_file()  # имя кэша по бэкенду
    assert kb.get("R1").title == "Залог"  # O(1) доступ


def test_metric_precision_recall_eval():
    from forge.pipeline.testing import _eval_metric
    metric = {"field": "detected_risks", "key": "risk_code", "expected": ["A", "B"], "min_recall": 1.0}
    out = {"detected_risks": [{"risk_code": "A"}, {"risk_code": "B"}, {"risk_code": "C"}]}
    ok, summary, _ = _eval_metric(metric, out)
    assert ok and "R=1.00" in summary  # оба ожидаемых найдены (precision ниже не важна)
    bad = {"detected_risks": [{"risk_code": "A"}]}
    ok2, _, _ = _eval_metric(metric, bad)
    assert not ok2  # recall 0.5 < порога 1.0


def test_contract_reconciles_blocking_issues(forge, wm):
    ws = wm.create("recon")
    bp = AgentBlueprint(name="recon", slug=ws.slug, archetype="json_extraction",
                        instructions_raw="Извлечь номер и дату из счёта.")
    forge.intake(bp)
    forge.analyze(bp, use_llm=False)
    assert bp.blocking_issues  # inputs/success/error are blocking after intake
    forge.build_contract(bp)
    # the contract auto-filled inputs/success/error → those blockers are cleared
    assert not bp.blocking_issues
    assert bp.stage == Stage.CONTRACTED
