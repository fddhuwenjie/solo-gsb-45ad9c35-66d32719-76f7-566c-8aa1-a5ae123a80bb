"""照度异常处置单：创建固化快照（诊断事件 + calendar=display 剂量 + 收据链末端），
结案核对摘要与重算证明；request_key 幂等，快照不可变。"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

BASE = 1_700_000_000  # 2023-11-14T22:13:20Z


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    db_file = tmp_path / "test.db"
    monkeypatch.setenv("DB_PATH", str(db_file))
    from app.main import app

    with TestClient(app) as c:
        yield c, db_file


def make_exhibit(client: TestClient, eid: str | None = None,
                 limit: float = 1000.0, interval: float = 3600.0) -> str:
    body = {"name": "唐代绢画", "dose_limit_luxh": limit, "max_interval_seconds": interval}
    if eid is not None:
        body["id"] = eid
    resp = client.post("/exhibits", json=body)
    assert resp.status_code == 201, resp.text
    return resp.json()["canonical_id"]


def set_calendar_24h(client: TestClient, exhibit_id: str) -> None:
    """每周七天全天开放（00:00–00:00 跨午夜 = 24 小时）。"""
    resp = client.put(f"/exhibits/{exhibit_id}/calendar", json={
        "timezone": "UTC",
        "weekly": [{"weekday": d, "start": "00:00", "end": "00:00"} for d in range(1, 8)],
    })
    assert resp.status_code == 200, resp.text


def send(client: TestClient, exhibit_id: str, readings: list[dict]) -> dict:
    resp = client.post(f"/exhibits/{exhibit_id}/readings", json={"readings": readings})
    assert resp.status_code == 200, resp.text
    return resp.json()


def create_ticket(client: TestClient, exhibit_id: str, request_key: str = "key-1",
                  start: float = BASE, end: float = BASE + 7200,
                  max_rate: float = 1.0, stuck_tol: float = 2.0,
                  min_stuck: float = 1800.0):
    return client.post(f"/exhibits/{exhibit_id}/anomaly-tickets", json={
        "start": iso(start), "end": iso(end),
        "max_rate_lux_per_second": max_rate,
        "stuck_tolerance_lux": stuck_tol,
        "min_stuck_seconds": min_stuck,
        "request_key": request_key,
    })


def close_ticket(client: TestClient, exhibit_id: str, ticket_id: str,
                 expected_digest: str, decision: str = "accepted",
                 note: str = "现场确认", evidence: dict | None = None):
    body = {"expected_digest": expected_digest, "decision": decision, "note": note}
    if evidence is not None:
        body["evidence"] = evidence
    return client.post(
        f"/exhibits/{exhibit_id}/anomaly-tickets/{ticket_id}/close", json=body
    )


def make_jump_exhibit(client: TestClient, eid: str | None = None,
                      limit: float = 100000.0) -> tuple[str, list[dict]]:
    """三条读数分三批上报：r1(100) → r2(500) → r3(100)，产生两个 jump 事件。"""
    eid = make_exhibit(client, eid=eid, limit=limit)
    set_calendar_24h(client, eid)
    receipts = []
    for rid, offset, lux in [("r1", 0, 100), ("r2", 60, 500), ("r3", 120, 100)]:
        out = send(client, eid, [{"id": rid, "ts": iso(BASE + offset), "lux": lux}])
        receipts.append(out["receipt"])
    return eid, receipts


# ---------- 创建：快照与 digest ----------

def test_create_ticket_freezes_snapshot_and_digest(env):
    client, _ = env
    eid, receipts = make_jump_exhibit(client)
    resp = create_ticket(client, eid)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["already_committed"] is False
    assert data["status"] == "open"
    assert data["canonical_id"] == eid == data["requested_id"]
    assert data["closed_at"] is None and data["closure"] is None
    assert len(data["digest"]) == 64

    # 快照：两个 jump 事件（r1→r2、r2→r3），边界基于完整时间线
    events = data["snapshot"]["events"]
    assert [e["type"] for e in events] == ["jump", "jump"]
    assert events[0]["reading_ids"] == ["r1", "r2"]
    assert events[1]["reading_ids"] == ["r2", "r3"]
    assert events[0]["rate_lux_per_second"] == pytest.approx(400 / 60)

    # 快照：calendar=display 剂量（全天开放，closed_seconds 为 0）
    dose = data["snapshot"]["dose"]
    assert dose["closed_seconds"] == 0.0
    assert dose["exceeded"] is False
    assert dose["reading_count"] == 3
    assert dose["known_dose_luxh"] == pytest.approx(
        (100 + 500) / 2 * 60 / 3600 + (500 + 100) / 2 * 60 / 3600
    )

    # 快照：收据链末端（第 3 张收据的链哈希）
    tail = data["snapshot"]["receipt_chain_tail"]
    assert tail["chain_id"] == eid
    assert tail["seq"] == 3
    assert tail["chain_hash"] == receipts[-1]["chain_hash"]


def test_create_ticket_without_calendar_is_409(env):
    client, _ = env
    eid = make_exhibit(client)
    send(client, eid, [{"id": "r1", "ts": iso(BASE), "lux": 100}])
    resp = create_ticket(client, eid)
    assert resp.status_code == 409
    assert "日历" in resp.json()["detail"]
    # 未留处置单
    assert client.get(f"/exhibits/{eid}/anomaly-tickets").json()["tickets"] == []


def test_create_ticket_validates_window_and_exhibit(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    assert create_ticket(client, eid, start=BASE + 100, end=BASE + 100).status_code == 422
    assert create_ticket(client, "ghost").status_code == 404


def test_create_ticket_empty_chain_tail_uses_genesis(env):
    client, _ = env
    eid = make_exhibit(client)
    set_calendar_24h(client, eid)
    resp = create_ticket(client, eid)
    assert resp.status_code == 200
    tail = resp.json()["snapshot"]["receipt_chain_tail"]
    assert tail["seq"] == 0
    assert tail["chain_hash"] == "0" * 64
    assert resp.json()["snapshot"]["events"] == []


# ---------- 创建幂等：重放 / 换目标 409 / 并发 ----------

def test_request_key_replay_and_conflict(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    first = create_ticket(client, eid, request_key="k-same")
    assert first.status_code == 200 and first.json()["already_committed"] is False

    again = create_ticket(client, eid, request_key="k-same")
    assert again.status_code == 200
    body = again.json()
    assert body["already_committed"] is True
    assert body["ticket_id"] == first.json()["ticket_id"]
    assert body["digest"] == first.json()["digest"]
    assert body["created_at"] == first.json()["created_at"]

    # 同 request_key 换时间窗 / 阈值 / 档案 → 409
    assert create_ticket(client, eid, request_key="k-same",
                         end=BASE + 9999).status_code == 409
    assert create_ticket(client, eid, request_key="k-same",
                         max_rate=2.0).status_code == 409
    eid2, _ = make_jump_exhibit(client)
    resp = create_ticket(client, eid2, request_key="k-same")
    assert resp.status_code == 409
    assert "已绑定另一份处置单请求" in resp.json()["detail"]
    # 只有一张处置单
    assert len(client.get(f"/exhibits/{eid}/anomaly-tickets").json()["tickets"]) == 1


def test_concurrent_create_same_request_key_single_ticket(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    statuses: list[int] = []
    tickets: list[str] = []
    replays: list[bool] = []

    def post(_: int) -> None:
        r = create_ticket(client, eid, request_key="conc-key")
        statuses.append(r.status_code)
        tickets.append(r.json()["ticket_id"])
        replays.append(r.json()["already_committed"])

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(post, range(8)))
    assert statuses == [200] * 8
    assert len(set(tickets)) == 1
    assert replays.count(False) == 1 and replays.count(True) == 7


# ---------- 结案：accepted ----------

def test_close_accepted_keeps_events_and_records_reason(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()

    resp = close_ticket(client, eid, ticket["ticket_id"], ticket["digest"],
                        decision="accepted", note="照度尖峰为临时布展灯光，风险已评估")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "accepted"
    assert data["closed_at"] is not None
    closure = data["closure"]
    assert closure["decision"] == "accepted"
    assert closure["note"] == "照度尖峰为临时布展灯光，风险已评估"
    assert closure["expected_digest"] == ticket["digest"]
    assert closure["events_kept"] == 2
    assert closure["proof"] is None and closure["evidence"] is None

    # 快照不可变：事件与 digest 与创建时一致
    assert data["digest"] == ticket["digest"]
    assert data["snapshot"] == ticket["snapshot"]


def test_close_requires_open_status(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()
    assert close_ticket(client, eid, ticket["ticket_id"], ticket["digest"],
                        decision="accepted", note="ok").status_code == 200
    again = close_ticket(client, eid, ticket["ticket_id"], ticket["digest"],
                         decision="accepted", note="重复结案")
    assert again.status_code == 409
    assert "已结案" in again.json()["detail"]


def test_close_digest_mismatch_is_409_and_writes_nothing(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()

    # 创建后新上报读数：收据链末端与诊断事件都变化 → 摘要不同
    send(client, eid, [{"id": "r4", "ts": iso(BASE + 180), "lux": 100}])
    resp = close_ticket(client, eid, ticket["ticket_id"], ticket["digest"],
                        decision="accepted", note="过期摘要")
    assert resp.status_code == 409
    assert "摘要已变化" in resp.json()["detail"]

    # 不写入：处置单仍为 open，无结案记录
    detail = client.get(f"/exhibits/{eid}/anomaly-tickets/{ticket['ticket_id']}").json()
    assert detail["status"] == "open"
    assert detail["closure"] is None
    assert detail["digest_stale"] is True
    assert detail["current_digest"] != ticket["digest"]
    # 用当前摘要结案成功
    ok = close_ticket(client, eid, ticket["ticket_id"], detail["current_digest"],
                      decision="accepted", note="按当前状态接受")
    assert ok.status_code == 200, ok.text


def test_close_digest_changes_with_void_source_priority_and_calendar(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()
    tid = ticket["ticket_id"]

    # 作废变化 → 摘要变化
    receipts = client.get(f"/exhibits/{eid}/receipts").json()["receipts"]
    corr = client.post(f"/exhibits/{eid}/receipts/corrections", json={
        "receipt_id": receipts[1]["receipt_id"], "request_key": "void-r2",
        "reason": "r2 为传感器故障读数",
    })
    assert corr.status_code == 200
    detail = client.get(f"/exhibits/{eid}/anomaly-tickets/{tid}").json()
    assert detail["digest_stale"] is True
    assert close_ticket(client, eid, tid, ticket["digest"],
                        decision="accepted", note="x").status_code == 409

    # 来源优先级变化 → 摘要变化（另建多来源展品）
    eid2 = make_exhibit(client, limit=100000.0)
    set_calendar_24h(client, eid2)
    send(client, eid2, [
        {"id": "a1", "ts": iso(BASE), "lux": 100, "source_id": "sensor-a"},
        {"id": "b1", "ts": iso(BASE), "lux": 500, "source_id": "sensor-b"},
        {"id": "a2", "ts": iso(BASE + 60), "lux": 100, "source_id": "sensor-a"},
    ])
    assert client.put(f"/exhibits/{eid2}/source-priority", json={
        "effective_at": iso(BASE - 10), "priorities": ["sensor-a", "sensor-b"],
        "expected_version": 0,
    }).status_code == 201
    t2 = create_ticket(client, eid2, request_key="k-sp").json()
    assert client.put(f"/exhibits/{eid2}/source-priority", json={
        "effective_at": iso(BASE - 10), "priorities": ["sensor-b", "sensor-a"],
        "expected_version": 1,
    }).status_code == 201
    d2 = client.get(f"/exhibits/{eid2}/anomaly-tickets/{t2['ticket_id']}").json()
    assert d2["digest_stale"] is True
    assert close_ticket(client, eid2, t2["ticket_id"], t2["digest"],
                        decision="accepted", note="x").status_code == 409

    # 日历变化 → 摘要变化（改回 eid：改用 08:00–20:00 的开放时段）
    client.put(f"/exhibits/{eid}/calendar", json={
        "timezone": "UTC",
        "weekly": [{"weekday": d, "start": "08:00", "end": "20:00"} for d in range(1, 8)],
    })
    d3 = client.get(f"/exhibits/{eid}/anomaly-tickets/{tid}").json()
    assert d3["digest_stale"] is True
    assert d3["current_digest"] != ticket["digest"]
    # 旧摘要 409；以当前摘要 accepted 结案成功
    assert close_ticket(client, eid, tid, ticket["digest"],
                        decision="accepted", note="x").status_code == 409
    ok = close_ticket(client, eid, tid, d3["current_digest"],
                      decision="accepted", note="按新日历接受")
    assert ok.status_code == 200, ok.text
    assert ok.json()["status"] == "accepted"


def test_close_deleted_calendar_cannot_recompute(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()
    assert client.delete(f"/exhibits/{eid}/calendar").status_code == 204
    detail = client.get(f"/exhibits/{eid}/anomaly-tickets/{ticket['ticket_id']}").json()
    assert detail["current_digest"] is None
    assert detail["digest_stale"] is True
    resp = close_ticket(client, eid, ticket["ticket_id"], ticket["digest"],
                        decision="accepted", note="日历已删")
    assert resp.status_code == 409


# ---------- 结案：resolved ----------

def test_close_resolved_after_correction(env):
    client, _ = env
    eid, receipts = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()
    assert len(ticket["snapshot"]["events"]) == 2

    # 作废 r2 所在批次：jump 事件消失，剂量未越限
    corr = client.post(f"/exhibits/{eid}/receipts/corrections", json={
        "receipt_id": receipts[1]["receipt_id"], "request_key": "fix-r2",
        "reason": "r2 为接线错误产生的尖峰，批次作废",
    })
    assert corr.status_code == 200

    detail = client.get(f"/exhibits/{eid}/anomaly-tickets/{ticket['ticket_id']}").json()
    assert detail["digest_stale"] is True

    resp = close_ticket(
        client, eid, ticket["ticket_id"], detail["current_digest"],
        decision="resolved", note="故障批次已作废，时间线恢复正常",
        evidence={"receipt_ids": [receipts[1]["receipt_id"]],
                  "correction_request_keys": ["fix-r2"]},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "resolved"
    closure = data["closure"]
    assert closure["decision"] == "resolved"
    assert closure["evidence"]["receipt_ids"] == [receipts[1]["receipt_id"]]
    assert closure["evidence"]["correction_request_keys"] == ["fix-r2"]
    proof = closure["proof"]
    assert proof["original_events_cleared"] is True
    assert proof["dose_exceeded_after"] is False
    assert proof["events_before"] == 2
    assert proof["events_after"] == 0
    # 快照仍是创建时的两个 jump 事件（不可变）
    assert len(data["snapshot"]["events"]) == 2
    assert data["digest"] == ticket["digest"]


def test_close_resolved_requires_evidence(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()
    # 缺 evidence → 422
    assert close_ticket(client, eid, ticket["ticket_id"], ticket["digest"],
                        decision="resolved", note="无佐证").status_code == 422
    # 空 evidence → 422
    assert close_ticket(client, eid, ticket["ticket_id"], ticket["digest"],
                        decision="resolved", note="空佐证",
                        evidence={"receipt_ids": [],
                                  "correction_request_keys": []}).status_code == 422


def test_close_resolved_with_events_remaining_is_409(env):
    client, _ = env
    eid, receipts = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()
    # 什么都没处置，原事件仍在 → 409（摘要未变，能通过摘要核对）
    resp = close_ticket(
        client, eid, ticket["ticket_id"], ticket["digest"],
        decision="resolved", note="未处置就结案",
        evidence={"receipt_ids": [receipts[1]["receipt_id"]],
                  "correction_request_keys": []},
    )
    assert resp.status_code == 409
    assert "原诊断事件仍未全部消失" in resp.json()["detail"]
    assert client.get(
        f"/exhibits/{eid}/anomaly-tickets/{ticket['ticket_id']}"
    ).json()["status"] == "open"


def test_close_resolved_with_dose_still_exceeded_is_409(env):
    client, _ = env
    # 无诊断事件但剂量越限：两条平稳读数，剂量超过上限
    eid = make_exhibit(client, limit=50.0)
    set_calendar_24h(client, eid)
    out1 = send(client, eid, [{"id": "r1", "ts": iso(BASE), "lux": 100}])
    send(client, eid, [{"id": "r2", "ts": iso(BASE + 3600), "lux": 100}])
    # min_stuck 大于 3600：两条平稳读数不会形成 stuck，事件为空但剂量越限
    ticket = create_ticket(client, eid, min_stuck=7200.0).json()
    assert ticket["snapshot"]["events"] == []
    assert ticket["snapshot"]["dose"]["exceeded"] is True

    # 事件（空）已"消失"，但剂量仍越限 → 409
    resp = close_ticket(
        client, eid, ticket["ticket_id"], ticket["digest"],
        decision="resolved", note="剂量仍越限",
        evidence={"receipt_ids": [out1["receipt"]["receipt_id"]],
                  "correction_request_keys": []},
    )
    assert resp.status_code == 409
    assert "剂量仍越限" in resp.json()["detail"]

    # 作废第二批后剂量不再越限 → 可 resolved
    receipts = client.get(f"/exhibits/{eid}/receipts").json()["receipts"]
    client.post(f"/exhibits/{eid}/receipts/corrections", json={
        "receipt_id": receipts[1]["receipt_id"], "request_key": "fix-dose",
        "reason": "超量程批次作废",
    })
    detail = client.get(f"/exhibits/{eid}/anomaly-tickets/{ticket['ticket_id']}").json()
    ok = close_ticket(
        client, eid, ticket["ticket_id"], detail["current_digest"],
        decision="resolved", note="越限批次已作废",
        evidence={"receipt_ids": [], "correction_request_keys": ["fix-dose"]},
    )
    assert ok.status_code == 200, ok.text
    assert ok.json()["closure"]["proof"]["dose_exceeded_after"] is False


def test_close_resolved_evidence_must_exist_and_belong(env):
    client, _ = env
    eid, receipts = make_jump_exhibit(client)
    other, other_receipts = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()
    tid = ticket["ticket_id"]
    # 作废 r2 使事件消失、摘要更新
    client.post(f"/exhibits/{eid}/receipts/corrections", json={
        "receipt_id": receipts[1]["receipt_id"], "request_key": "fix-r2",
        "reason": "故障批次作废",
    })
    digest_now = client.get(f"/exhibits/{eid}/anomaly-tickets/{tid}").json()["current_digest"]

    # 不存在的收据 → 404
    resp = close_ticket(client, eid, tid, digest_now, decision="resolved", note="x",
                        evidence={"receipt_ids": ["deadbeef"],
                                  "correction_request_keys": []})
    assert resp.status_code == 404
    # 其他展品的收据 → 409
    resp = close_ticket(client, eid, tid, digest_now, decision="resolved", note="x",
                        evidence={"receipt_ids": [other_receipts[0]["receipt_id"]],
                                  "correction_request_keys": []})
    assert resp.status_code == 409
    # 不存在的更正 → 404
    resp = close_ticket(client, eid, tid, digest_now, decision="resolved", note="x",
                        evidence={"receipt_ids": [],
                                  "correction_request_keys": ["ghost-key"]})
    assert resp.status_code == 404
    # 仍未结案
    assert client.get(f"/exhibits/{eid}/anomaly-tickets/{tid}").json()["status"] == "open"


def test_close_unknown_ticket_or_exhibit(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    other, _ = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()
    # 别的展品名下 → 404；不存在的处置单 → 404；不存在的展品 → 404
    assert close_ticket(client, other, ticket["ticket_id"], ticket["digest"],
                        decision="accepted", note="x").status_code == 404
    assert close_ticket(client, eid, "no-such-ticket", ticket["digest"],
                        decision="accepted", note="x").status_code == 404
    assert close_ticket(client, "ghost", ticket["ticket_id"], ticket["digest"],
                        decision="accepted", note="x").status_code == 404
    assert client.get(f"/exhibits/{other}/anomaly-tickets/{ticket['ticket_id']}").status_code == 404


def test_concurrent_close_only_one_commits(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()
    tid = ticket["ticket_id"]
    statuses: list[int] = []

    def post(i: int) -> None:
        r = close_ticket(client, eid, tid, ticket["digest"],
                         decision="accepted", note=f"并发结案 {i}")
        statuses.append(r.status_code)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(post, range(6)))
    assert sorted(statuses) == [200] + [409] * 5
    detail = client.get(f"/exhibits/{eid}/anomaly-tickets/{tid}").json()
    assert detail["status"] == "accepted"


# ---------- 别名 / 列表 / 快照不可变 ----------

def test_ticket_via_alias_after_merge(env):
    client, _ = env
    src, _ = make_jump_exhibit(client, eid="src")
    tgt, _ = make_jump_exhibit(client, eid="tgt")
    pv = client.post("/merges/preview", json={"source_id": "src", "target_id": "tgt"})
    assert client.post("/merges/commit",
                       json={"token": pv.json()["token"], "resolutions": []}).status_code == 200

    # 用别名创建：解析到规范档案 tgt
    resp = create_ticket(client, "src", request_key="k-alias")
    assert resp.status_code == 200, resp.text
    ticket = resp.json()
    assert ticket["requested_id"] == "src"
    assert ticket["canonical_id"] == "tgt"

    # 用别名列表 / 查询 / 结案
    lst = client.get("/exhibits/src/anomaly-tickets").json()
    assert lst["requested_id"] == "src" and lst["canonical_id"] == "tgt"
    assert [t["ticket_id"] for t in lst["tickets"]] == [ticket["ticket_id"]]
    detail = client.get(f"/exhibits/src/anomaly-tickets/{ticket['ticket_id']}").json()
    assert detail["canonical_id"] == "tgt"
    closed = close_ticket(client, "src", ticket["ticket_id"], ticket["digest"],
                          decision="accepted", note="别名结案")
    assert closed.status_code == 200
    assert closed.json()["requested_id"] == "src"

    # 别名重放创建：同一规范档案 → 回放首次结果
    replay = create_ticket(client, "src", request_key="k-alias")
    assert replay.status_code == 200
    assert replay.json()["already_committed"] is True
    assert replay.json()["ticket_id"] == ticket["ticket_id"]


def test_snapshot_immutable_after_state_changes(env):
    client, _ = env
    eid, receipts = make_jump_exhibit(client)
    ticket = create_ticket(client, eid).json()
    tid = ticket["ticket_id"]

    # 作废 + 新上报改变读数/作废/收据链
    client.post(f"/exhibits/{eid}/receipts/corrections", json={
        "receipt_id": receipts[1]["receipt_id"], "request_key": "fix-r2",
        "reason": "故障批次作废",
    })
    send(client, eid, [{"id": "r4", "ts": iso(BASE + 180), "lux": 100}])

    detail = client.get(f"/exhibits/{eid}/anomaly-tickets/{tid}").json()
    # 固化快照与 digest 不变；当前摘要已不同
    assert detail["digest"] == ticket["digest"]
    assert detail["snapshot"] == ticket["snapshot"]
    assert detail["snapshot"]["events"][0]["reading_ids"] == ["r1", "r2"]
    assert detail["digest_stale"] is True
    assert detail["current_digest"] != ticket["digest"]

    # 列表同样返回固化快照
    lst = client.get(f"/exhibits/{eid}/anomaly-tickets").json()["tickets"]
    assert lst[0]["digest"] == ticket["digest"]
    assert lst[0]["snapshot"] == ticket["snapshot"]


def test_list_tickets_orders_and_scopes_by_exhibit(env):
    client, _ = env
    eid, _ = make_jump_exhibit(client)
    other, _ = make_jump_exhibit(client)
    t1 = create_ticket(client, eid, request_key="k1").json()
    t2 = create_ticket(client, eid, request_key="k2",
                       start=BASE + 1000, end=BASE + 5000).json()
    # 同 request_key 指向另一档案 → 409
    assert create_ticket(client, other, request_key="k1").status_code == 409
    create_ticket(client, other, request_key="k-other")

    lst = client.get(f"/exhibits/{eid}/anomaly-tickets").json()
    assert [t["ticket_id"] for t in lst["tickets"]] == [t1["ticket_id"], t2["ticket_id"]]
    other_lst = client.get(f"/exhibits/{other}/anomaly-tickets").json()["tickets"]
    assert len(other_lst) == 1
    assert other_lst[0]["canonical_id"] == other
