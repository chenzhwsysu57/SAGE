import json
from pathlib import Path
import sys

from base.memory import BaseMemory
from modeling.leduc.archetypes import ARCHETYPE_NAMES
from modeling.leduc.llm_belief import OpponentBelief
from modeling.leduc.statistical_profile import StatisticalProfileMethod
from methods.cfr import CFRPlusTrainer
import runs.leduc.leduc_tom_ablation as tom_ablation
from runs.leduc.leduc_tom_ablation import run_condition


class LegalAwareModel:
    def generate(self, prompt: str) -> str:
        if "\n  call" in prompt:
            return (
                '{"reasoning": "call is legal", "action": {"type": "call"}}'
            )
        return (
            '{"reasoning": "check is legal", "action": {"type": "check"}}'
        )


def test_cfr_condition_saves_both_players_prompts(tmp_path):
    trainer = CFRPlusTrainer(iterations=10)
    trainer.train()
    model_path = tmp_path / "cfr_plus.json"
    model_path.write_text(json.dumps({
        "iterations": 10,
        "num_infosets": len(trainer.strategy),
        "strategy": {key: dict(value) for key, value in trainer.strategy.items()},
    }))
    tom_ablation.CFRGTOAgent._DEFAULT_PATH = str(model_path)

    stats = run_condition(
        LegalAwareModel(),
        condition_key="cfr",
        archetype_name="CallingStation",
        games=1,
        start_seed=42,
        save_prompts=True,
        run_dir=tmp_path,
    )

    assert stats["condition_key"] == "cfr"
    assert stats["games"] == 2

    prompt_files = sorted((tmp_path / "prompts").rglob("*_prompt.txt"))
    response_files = sorted((tmp_path / "prompts").rglob("*_response.txt"))
    assert prompt_files
    assert len(prompt_files) == len(response_files)

    names = [path.name for path in prompt_files]
    assert any("_CFR-P0_prompt.txt" in name for name in names)
    assert any("_CallingStation-P1_prompt.txt" in name for name in names)

    cfr_prompts = [
        path.read_text()
        for path in prompt_files
        if "_CFR-" in path.name
    ]
    llm_prompts = [
        path.read_text()
        for path in prompt_files
        if "_CallingStation-" in path.name
    ]
    assert any("CFR-GTO:" in text for text in cfr_prompts)
    assert any("--- Your Playing Style ---" in text for text in llm_prompts)


def test_tom_condition_does_not_write_keynode_prompts(tmp_path):
    stats = run_condition(
        LegalAwareModel(),
        condition_key="tom_nocfr",
        archetype_name="CallingStation",
        games=1,
        start_seed=42,
        save_prompts=True,
        run_dir=tmp_path,
    )

    assert stats["condition_key"] == "tom_nocfr"
    prompt_files = sorted((tmp_path / "prompts").rglob("*_prompt.txt"))
    response_files = sorted((tmp_path / "prompts").rglob("*_response.txt"))
    assert prompt_files
    assert not list((tmp_path / "prompts").rglob("*_keynode_prompt.txt"))
    assert not list((tmp_path / "prompts").rglob("*_keynode_response.txt"))
    assert all("predicted_opponent_action" not in path.read_text() for path in prompt_files)


def test_llm_belief_prompt_shows_full_distribution_and_smoothing_note():
    belief = OpponentBelief(decay=0.5)
    belief.update({
        "CallingStation": 0.30,
        "TAG": 0.22,
        "Nit": 0.18,
        "GTO": 0.12,
        "LAG": 0.10,
        "Maniac": 0.08,
    }, "test")

    text = belief.to_prompt()

    for arch in ARCHETYPE_NAMES:
        assert f"{arch}:" in text
    assert "Top-ranked:" not in text
    assert "Current beliefs:" in text
    assert "Top two are close" not in text


def test_statistical_profile_uses_supplied_signatures_in_statement():
    signatures = {
        "CallingStation": {
            "no_bet_check": 0.0,
            "no_bet_bet": 1.0,
            "fb_call": 0.0,
            "fb_raise": 1.0,
            "fb_fold": 0.0,
            "selectivity": 0.0,
            "entropy": 0.0,
        },
        "Maniac": {
            "no_bet_check": 1.0,
            "no_bet_bet": 0.0,
            "fb_call": 1.0,
            "fb_raise": 0.0,
            "fb_fold": 0.0,
            "selectivity": 0.0,
            "entropy": 0.0,
        },
    }
    metadata = {
        "mode": "data",
        "source": "historical_manifest",
        "archetypes": {
            "CallingStation": {
                "games": 3,
                "actions": 12,
                "p0_games": 1,
                "p1_games": 2,
            },
            "Maniac": {
                "games": 2,
                "actions": 9,
                "p0_games": 1,
                "p1_games": 1,
            },
        },
    }
    method = StatisticalProfileMethod(
        memory=BaseMemory(),
        signatures=signatures,
        signature_source="historical_manifest",
        signature_metadata=metadata,
    )
    method.update(1, {
        "steps": [{
            "player_id": 1,
            "action": "P1 bets 2",
            "private_cards": {"1": "JH"},
        }]
    })

    text = method.statement(1, n_actions=2)

    assert "Statistical Profile" in text
    assert "observed action frequencies vs archetype signatures" not in text
    assert "Classification mode: q_window_ema" in text
    assert "Q-window: last 10 games" in text
    assert "EMA alpha: 0.85" in text
    assert "Recent style probabilities:" in text
    assert "game 1:" in text
    assert "CallingStation " in text
    assert "Current top match" not in text
    assert "runner-up" not in text
    assert "Most possible" not in text
    assert "Signature training samples" not in text
    assert "Interpretation:" not in text


def test_statistical_profile_q_window_ema_uses_recent_games():
    signatures = {
        "BetStyle": {
            "no_bet_check": 0.0,
            "no_bet_bet": 1.0,
            "fb_call": 0.0,
            "fb_raise": 0.0,
            "fb_fold": 0.0,
            "selectivity": 0.0,
            "entropy": 0.0,
        },
        "CheckStyle": {
            "no_bet_check": 1.0,
            "no_bet_bet": 0.0,
            "fb_call": 0.0,
            "fb_raise": 0.0,
            "fb_fold": 0.0,
            "selectivity": 0.0,
            "entropy": 0.0,
        },
    }
    method = StatisticalProfileMethod(
        memory=BaseMemory(),
        signatures=signatures,
        classification_mode="q_window_ema",
        q_window=1,
        q_alpha=0.0,
    )

    method.update(1, {
        "steps": [{
            "player_id": 1,
            "action": "P1 bets 2",
            "private_cards": {"1": "KH"},
        }],
    })
    first = method._posterior_from_current_profile(1)
    method.update(1, {
        "steps": [{
            "player_id": 1,
            "action": "P1 checks",
            "private_cards": {"1": "JH"},
        }],
    })
    second = method._posterior_from_current_profile(1)
    text = method.statement(1)

    assert first["BetStyle"] > first["CheckStyle"]
    assert second["CheckStyle"] > second["BetStyle"]
    assert "Q-window: last 1 games" in text
    assert "2 total actions, 1 evidence actions" in text
    assert text.index("game 2:") < text.index("game 1:")


def _write_signature_manifest(tmp_path):
    entries = {}
    for i, arch in enumerate(ARCHETYPE_NAMES):
        path = tmp_path / f"{arch}.json"
        p1_action = "P1 calls 2" if i % 2 == 0 else "P1 bets 2"
        p0_action = "P0 calls 2" if i % 2 == 0 else "P0 bets 2"
        path.write_text(json.dumps({
            "batches": [
                {
                    "label": f"Simple_vs_{arch}_P0",
                    "sample_trajectories": [{
                        "steps": [{
                            "player_id": 1,
                            "action": p1_action,
                            "private_cards": {"1": "JH"},
                        }]
                    }],
                },
                {
                    "label": f"{arch}_vs_Simple_P1",
                    "sample_trajectories": [{
                        "steps": [{
                            "player_id": 0,
                            "action": p0_action,
                            "private_cards": {"0": "QH"},
                        }]
                    }],
                },
                {
                    "label": "Unrelated_vs_Other_P0",
                    "sample_trajectories": [{
                        "steps": [{
                            "player_id": 1,
                            "action": "P1 raises 4",
                            "private_cards": {"1": "KH"},
                        }]
                    }],
                },
            ]
        }))
        entries[arch] = {
            "train_json": str(path),
            "train_batch_idx": 0,
            "test_json": str(path),
            "test_batch_idx": 0,
            "opp_id": 1,
        }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"entries": entries}))
    return manifest


def test_main_writes_signature_metadata_to_config_and_summary(tmp_path, monkeypatch):
    manifest = _write_signature_manifest(tmp_path)
    output_dir = tmp_path / "results"

    monkeypatch.setattr(tom_ablation, "OpenAIModel", lambda **kw: LegalAwareModel())
    monkeypatch.setattr(sys, "argv", [
        "leduc_tom_ablation.py",
        "--condition", "simple",
        "--games", "1",
        "--method", "statistical",
        "--signature-manifest", str(manifest),
        "--output-dir", str(output_dir),
    ])

    tom_ablation.main()

    run_dirs = sorted(output_dir.iterdir())
    assert len(run_dirs) == 1
    config = json.loads((run_dirs[0] / "config.json").read_text())
    summary = json.loads((run_dirs[0] / "summary.json").read_text())

    for doc in (config, summary):
        meta = doc["signature_metadata"]
        assert meta["mode"] == "data"
        assert meta["manifest_path"] == str(manifest)
        assert meta["position_aggregation"] == "p0_p1_merged"
        assert meta["fallback_to_persona"] is False
        assert set(meta["archetypes"]) == set(ARCHETYPE_NAMES)
        assert all(item["games"] == 2 for item in meta["archetypes"].values())
        assert all(item["total_games"] == 2 for item in meta["archetypes"].values())
        assert all(item["p0_games"] == 1 for item in meta["archetypes"].values())
        assert all(item["p1_games"] == 1 for item in meta["archetypes"].values())
        assert all(item["actions"] == 2 for item in meta["archetypes"].values())
        assert all(item["total_actions"] == 2 for item in meta["archetypes"].values())
        assert all(len(item["batch_labels"]) == 2 for item in meta["archetypes"].values())


def test_main_defaults_to_statistical_modeling(tmp_path, monkeypatch):
    manifest = _write_signature_manifest(tmp_path)
    output_dir = tmp_path / "results"

    monkeypatch.setattr(tom_ablation, "OpenAIModel", lambda **kw: LegalAwareModel())
    monkeypatch.setattr(sys, "argv", [
        "leduc_tom_ablation.py",
        "--condition", "simple",
        "--games", "1",
        "--signature-manifest", str(manifest),
        "--output-dir", str(output_dir),
    ])

    tom_ablation.main()

    run_dirs = sorted(output_dir.iterdir())
    assert len(run_dirs) == 1
    config = json.loads((run_dirs[0] / "config.json").read_text())
    summary = json.loads((run_dirs[0] / "summary.json").read_text())

    assert config["args"]["method"] == "statistical"
    assert config["args"]["stat_mode"] == "q_window_log"
    assert config["args"]["q_window"] == 10
    assert config["args"]["q_tau"] == 1.5
    assert summary["modeling_method"] == "statistical"
    assert summary["stat_mode"] == "q_window_log"
    assert summary["q_window"] == 10
    assert summary["q_tau"] == 1.5
