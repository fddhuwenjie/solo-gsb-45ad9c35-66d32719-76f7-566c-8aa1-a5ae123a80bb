"""照度异常处置单：创建时固化快照（诊断事件 + calendar=display 剂量 + 收据链末端），
结案核对摘要与重算证明。

设计要点：
- **创建**：提交展品 ID（可用别名）、时间窗、诊断阈值与 ``request_key``。
  服务在**一个 SQLite 事务**（``BEGIN IMMEDIATE``）内按**未作废权威时间线**
  重算诊断事件与 ``calendar=display`` 剂量，连同**收据链末端哈希**一起固化为
  快照，并以稳定规范序列化（与存证收据同一套 ``content_hash_of``）计算
  ``digest`` 返回。``request_key`` 为幂等键：相同请求（规范档案 / 时间窗 /
  诊断阈值一致）重放首次结果，换目标 → 409。
- **快照不可变**：``snapshot_json`` 与 ``digest`` 落库后绝不 UPDATE；结案只写
  ``status`` / ``closed_at`` / ``closure_json``。档案归并删除档案后处置单
  连同快照原样保留（本表无外键、不级联），供事后核查。
- **结案**：只能从 ``open`` 结案为 ``resolved`` 或 ``accepted``。提交
  ``expected_digest``、决定与说明；服务在写事务内按固化的时间窗与阈值**重算
  当前摘要**，读数、作废、来源优先级、日历或收据链变化导致摘要不同 →
  **409 且不写入**。``resolved`` 须给出涉及的收据或更正 ``request_key``
  （佐证须存在且属于该规范档案），并证明**重算后原事件全部消失、
  calendar=display 剂量未越限**；``accepted`` 保留事件并记录理由。
  已结案的处置单再次结案 → 409。
- 摘要覆盖：规范档案 ID、时间窗、诊断阈值（含展品上限与最大采样间隔）、
  重算的诊断事件、``calendar=display`` 剂量结果与收据链末端——任一输入
  （读数 / 作废标记 / 来源优先级版本 / 日历 / 收据链）变化都会改变摘要。
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from fastapi import HTTPException

from .arbitration import (
    arbitrate,
    diagnosis_breaks,
    diagnosis_feed,
    load_priority_versions,
)
from .calendar import load_calendar, open_intervals
from .diagnose import DiagnosisEvent, build_events, intersect_window, sort_events
from .dose import compute_dose
from .merge import get_canonical_or_404, resolve_canonical
from .receipts import GENESIS_PREV_HASH, content_hash_of
from .schemas import (
    AnomalyTicketCreateOut,
    AnomalyTicketCreateRequest,
    AnomalyTicketDetailOut,
    AnomalyTicketListOut,
    AnomalyTicketOut,
    AnomalyTicketCloseRequest,
    TicketEvidence,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _to_iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _round6(x: float) -> float:
    return round(x, 6)


# ---------- 快照重算（未作废权威时间线 + calendar=display + 收据链末端） ----------

def _event_dict(e: DiagnosisEvent) -> dict:
    return {
        "type": e.type,
        "start": _to_iso(e.start),
        "end": _to_iso(e.end),
        "reading_ids": list(e.reading_ids),
        "interval_seconds": _round6(e.interval_seconds) if e.interval_seconds is not None else None,
        "rate_lux_per_second": _round6(e.rate_lux_per_second) if e.rate_lux_per_second is not None else None,
        "duration_seconds": _round6(e.duration_seconds) if e.duration_seconds is not None else None,
        "lux_span": _round6(e.lux_span) if e.lux_span is not None else None,
    }


def compute_snapshot(
    conn: sqlite3.Connection,
    exhibit: sqlite3.Row,
    canonical_id: str,
    start_s: float,
    end_s: float,
    *,
    max_rate_lux_per_second: float,
    stuck_tolerance_lux: float,
    min_stuck_seconds: float,
) -> dict:
    """按未作废权威时间线重算诊断事件与 calendar=display 剂量，并取收据链末端。

    与诊断 / 剂量查询接口同口径：配置了来源优先级版本时先逐时刻仲裁，
    保持冲突的时刻不进入诊断输入、在剂量积分中作为断点；诊断事件在完整
    时间线上形成后按窗口相交筛选；剂量按展陈开放日历裁剪。
    未配置展陈开放日历 → 409（与 calendar=display 剂量查询一致）。
    """
    rows = conn.execute(
        "SELECT reading_id, ts, lux, source_id FROM readings "
        "WHERE exhibit_id = ? AND voided_at IS NULL ORDER BY ts, reading_id",
        (canonical_id,),
    ).fetchall()
    versions = load_priority_versions(conn, canonical_id)
    diag_breaks: list[float] = []
    dose_breaks: list[float] = []
    if versions:
        points, arb_conflicts = arbitrate(
            [(r["reading_id"], r["ts"], r["lux"], r["source_id"]) for r in rows],
            versions,
        )
        feed = diagnosis_feed(points)
        diag_breaks = diagnosis_breaks(arb_conflicts)
        dose_readings = [(p.ts, p.lux) for p in points]
        dose_breaks = [c.ts for c in arb_conflicts]
    else:
        feed = [(r["reading_id"], r["ts"], r["lux"]) for r in rows]
        dose_readings = [(r["ts"], r["lux"]) for r in rows]

    events = sort_events(intersect_window(
        build_events(
            feed,
            max_interval_s=exhibit["max_interval_seconds"],
            max_rate_lux_per_s=max_rate_lux_per_second,
            stuck_tol_lux=stuck_tolerance_lux,
            min_stuck_s=min_stuck_seconds,
            breaks=diag_breaks,
        ),
        start_s,
        end_s,
    ))

    cfg = load_calendar(conn, canonical_id)
    if cfg is None:
        raise HTTPException(
            status_code=409,
            detail=f"展品未配置展陈开放日历，无法按 calendar=display 核算: {canonical_id}",
        )
    # 开放区间须覆盖截至窗口末的开放历史（与剂量查询接口同一展开口径）
    first_ts = dose_readings[0][0] if dose_readings else start_s
    intervals = open_intervals(cfg, min(start_s, first_ts), end_s)
    dose = compute_dose(
        readings=dose_readings,
        start_s=start_s,
        end_s=end_s,
        max_interval_s=exhibit["max_interval_seconds"],
        dose_limit_luxh=exhibit["dose_limit_luxh"],
        open_intervals=intervals,
        break_ts=dose_breaks,
    )
    assert dose.closed_seconds is not None  # 传入开放区间后必有闭馆统计

    tail = conn.execute(
        "SELECT seq, chain_hash FROM ingest_receipts WHERE chain_id = ? "
        "ORDER BY seq DESC LIMIT 1",
        (canonical_id,),
    ).fetchone()
    chain_tail = {
        "chain_id": canonical_id,
        "seq": tail["seq"] if tail is not None else 0,
        "chain_hash": tail["chain_hash"] if tail is not None else GENESIS_PREV_HASH,
    }

    return {
        "events": [_event_dict(e) for e in events],
        "dose": {
            "known_dose_luxh": _round6(dose.known_dose_luxh),
            "remaining_luxh": _round6(dose.remaining_luxh),
            "exceeded": dose.exceeded,
            "first_exceeded_at": _to_iso(dose.first_exceeded_at)
            if dose.first_exceeded_at is not None else None,
            "unknown_seconds": _round6(dose.unknown_seconds),
            "covered_seconds": _round6(dose.covered_seconds),
            "closed_seconds": _round6(dose.closed_seconds),
            "reading_count": dose.reading_count,
        },
        "receipt_chain_tail": chain_tail,
    }


def snapshot_digest(
    exhibit: sqlite3.Row,
    canonical_id: str,
    start_s: float,
    end_s: float,
    *,
    max_rate_lux_per_second: float,
    stuck_tolerance_lux: float,
    min_stuck_seconds: float,
    snapshot: dict,
) -> str:
    """快照摘要：SHA-256（规范序列化），覆盖档案、时间窗、阈值与快照内容。

    任何读数 / 作废 / 来源优先级 / 日历 / 收据链变化都会反映到重算的
    事件、剂量或链末端上，从而改变摘要。
    """
    return content_hash_of({
        "canonical_id": canonical_id,
        "window": {"start": _to_iso(start_s), "end": _to_iso(end_s)},
        "params": {
            "max_interval_seconds": exhibit["max_interval_seconds"],
            "dose_limit_luxh": exhibit["dose_limit_luxh"],
            "max_rate_lux_per_second": max_rate_lux_per_second,
            "stuck_tolerance_lux": stuck_tolerance_lux,
            "min_stuck_seconds": min_stuck_seconds,
        },
        "snapshot": snapshot,
    })


def _current_digest(
    conn: sqlite3.Connection, exhibit: sqlite3.Row, row: sqlite3.Row
) -> tuple[str, dict]:
    """按处置单固化的时间窗与阈值，对当前数据重算摘要；返回 (摘要, 快照)。"""
    snapshot = compute_snapshot(
        conn, exhibit, row["canonical_id"], row["start_ts"], row["end_ts"],
        max_rate_lux_per_second=row["max_rate_lux_per_second"],
        stuck_tolerance_lux=row["stuck_tolerance_lux"],
        min_stuck_seconds=row["min_stuck_seconds"],
    )
    return snapshot_digest(
        exhibit, row["canonical_id"], row["start_ts"], row["end_ts"],
        max_rate_lux_per_second=row["max_rate_lux_per_second"],
        stuck_tolerance_lux=row["stuck_tolerance_lux"],
        min_stuck_seconds=row["min_stuck_seconds"],
        snapshot=snapshot,
    ), snapshot


# ---------- 视图 ----------

def _ticket_dict(row: sqlite3.Row, exhibit: sqlite3.Row, requested_id: str) -> dict:
    return {
        "ticket_id": row["ticket_id"],
        "request_key": row["request_key"],
        "requested_id": requested_id,
        "canonical_id": row["canonical_id"],
        "status": row["status"],
        "start": _to_iso(row["start_ts"]),
        "end": _to_iso(row["end_ts"]),
        "max_interval_seconds": exhibit["max_interval_seconds"],
        "dose_limit_luxh": exhibit["dose_limit_luxh"],
        "max_rate_lux_per_second": row["max_rate_lux_per_second"],
        "stuck_tolerance_lux": row["stuck_tolerance_lux"],
        "min_stuck_seconds": row["min_stuck_seconds"],
        "digest": row["digest"],
        "snapshot": json.loads(row["snapshot_json"]),
        "created_at": row["created_at"],
        "closed_at": row["closed_at"],
        "closure": json.loads(row["closure_json"]) if row["closure_json"] else None,
    }


def _load_ticket(conn: sqlite3.Connection, canonical_id: str, ticket_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM anomaly_tickets WHERE ticket_id = ? AND canonical_id = ?",
        (ticket_id, canonical_id),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"处置单不存在: {ticket_id}")
    return row


# ---------- 创建 ----------

def _replay_create(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    requested_id: str,
    payload: AnomalyTicketCreateRequest,
    start_s: float,
    end_s: float,
) -> AnomalyTicketCreateOut:
    """相同 request_key 重放：规范档案 / 时间窗 / 诊断阈值一致则回放，否则 409。"""
    canonical = resolve_canonical(conn, requested_id)
    if canonical is None:
        raise HTTPException(status_code=404, detail=f"展品不存在: {requested_id}")
    same_target = (
        canonical == row["canonical_id"]
        and start_s == row["start_ts"]
        and end_s == row["end_ts"]
        and payload.max_rate_lux_per_second == row["max_rate_lux_per_second"]
        and payload.stuck_tolerance_lux == row["stuck_tolerance_lux"]
        and payload.min_stuck_seconds == row["min_stuck_seconds"]
    )
    if not same_target:
        raise HTTPException(
            status_code=409,
            detail=f"request_key {payload.request_key} 已绑定另一份处置单请求"
                   "（档案/时间窗/诊断阈值不同），不能换目标重复提交",
        )
    exhibit = conn.execute(
        "SELECT * FROM exhibits WHERE id = ?", (row["canonical_id"],)
    ).fetchone()
    return AnomalyTicketCreateOut(
        **_ticket_dict(row, exhibit, requested_id), already_committed=True
    )


def create_ticket(
    conn: sqlite3.Connection,
    requested_id: str,
    payload: AnomalyTicketCreateRequest,
) -> AnomalyTicketCreateOut:
    """在单个写事务内重算快照、固化并返回 digest；request_key 幂等。"""
    start_s = payload.start.timestamp()
    end_s = payload.end.timestamp()
    if end_s <= start_s:
        raise HTTPException(status_code=422, detail="end 必须晚于 start")

    # 幂等回放不依赖写锁之外的状态：先读一次（无行则进入写事务）。
    existing = conn.execute(
        "SELECT * FROM anomaly_tickets WHERE request_key = ?", (payload.request_key,)
    ).fetchone()
    if existing is not None:
        return _replay_create(conn, existing, requested_id, payload, start_s, end_s)

    conn.execute("BEGIN IMMEDIATE")
    try:
        # 锁内重读：并发的相同 request_key 可能已完成落库
        existing = conn.execute(
            "SELECT * FROM anomaly_tickets WHERE request_key = ?", (payload.request_key,)
        ).fetchone()
        if existing is not None:
            conn.commit()  # 尚未写入，COMMIT 仅释放 IMMEDIATE 锁
            return _replay_create(conn, existing, requested_id, payload, start_s, end_s)

        exhibit = get_canonical_or_404(conn, requested_id)
        canonical_id = exhibit["id"]

        # 重算与固化在同一事务内：快照、摘要与处置单行同生共死。
        snapshot = compute_snapshot(
            conn, exhibit, canonical_id, start_s, end_s,
            max_rate_lux_per_second=payload.max_rate_lux_per_second,
            stuck_tolerance_lux=payload.stuck_tolerance_lux,
            min_stuck_seconds=payload.min_stuck_seconds,
        )
        digest = snapshot_digest(
            exhibit, canonical_id, start_s, end_s,
            max_rate_lux_per_second=payload.max_rate_lux_per_second,
            stuck_tolerance_lux=payload.stuck_tolerance_lux,
            min_stuck_seconds=payload.min_stuck_seconds,
            snapshot=snapshot,
        )
        ticket_id = uuid.uuid4().hex
        now_iso = _utc_now_iso()
        conn.execute(
            """INSERT INTO anomaly_tickets
               (ticket_id, request_key, canonical_id, requested_id,
                start_ts, end_ts, max_rate_lux_per_second, stuck_tolerance_lux,
                min_stuck_seconds, digest, snapshot_json, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?)""",
            (
                ticket_id, payload.request_key, canonical_id, requested_id,
                start_s, end_s, payload.max_rate_lux_per_second,
                payload.stuck_tolerance_lux, payload.min_stuck_seconds,
                digest, json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
                now_iso,
            ),
        )
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        raise

    row = conn.execute(
        "SELECT * FROM anomaly_tickets WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    return AnomalyTicketCreateOut(
        **_ticket_dict(row, exhibit, requested_id), already_committed=False
    )


# ---------- 查询 ----------

def list_tickets(conn: sqlite3.Connection, requested_id: str) -> AnomalyTicketListOut:
    """列出规范档案的全部处置单（按创建时间、ticket_id 升序）；别名自动解析。"""
    exhibit = get_canonical_or_404(conn, requested_id)
    canonical_id = exhibit["id"]
    rows = conn.execute(
        "SELECT * FROM anomaly_tickets WHERE canonical_id = ? "
        "ORDER BY created_at, ticket_id",
        (canonical_id,),
    ).fetchall()
    return AnomalyTicketListOut(
        requested_id=requested_id,
        canonical_id=canonical_id,
        tickets=[_ticket_dict(r, exhibit, requested_id) for r in rows],
    )


def get_ticket(
    conn: sqlite3.Connection, requested_id: str, ticket_id: str
) -> AnomalyTicketDetailOut:
    """查询单张处置单：固化快照不变，另附按当前数据实时重算的摘要。

    结案前用 ``current_digest`` 作为 ``expected_digest`` 提交；``digest_stale``
    表示读数 / 作废 / 来源优先级 / 日历 / 收据链相对创建时已变化。
    """
    exhibit = get_canonical_or_404(conn, requested_id)
    row = _load_ticket(conn, exhibit["id"], ticket_id)
    current_digest: str | None
    try:
        current_digest, _ = _current_digest(conn, exhibit, row)
        stale = current_digest != row["digest"]
    except HTTPException:
        # 日历已删除等导致无法重算：视为已变化，但不影响固化快照的读取
        current_digest = None
        stale = True
    return AnomalyTicketDetailOut(
        **_ticket_dict(row, exhibit, requested_id),
        current_digest=current_digest,
        digest_stale=stale,
    )


# ---------- 结案 ----------

def _verify_evidence(
    conn: sqlite3.Connection, canonical_id: str, evidence: TicketEvidence
) -> None:
    """佐证必须存在且属于该规范档案：收据在其链上、更正指向该档案。"""
    for receipt_id in evidence.receipt_ids:
        r = conn.execute(
            "SELECT chain_id FROM ingest_receipts WHERE receipt_id = ?", (receipt_id,)
        ).fetchone()
        if r is None:
            raise HTTPException(
                status_code=404, detail=f"佐证收据不存在: receipt_id={receipt_id}"
            )
        if r["chain_id"] != canonical_id:
            raise HTTPException(
                status_code=409,
                detail=f"佐证收据 {receipt_id} 属于链 {r['chain_id']}，"
                       f"不属于规范档案 {canonical_id} 的收据链",
            )
    for request_key in evidence.correction_request_keys:
        c = conn.execute(
            "SELECT canonical_id FROM reading_corrections WHERE request_key = ?",
            (request_key,),
        ).fetchone()
        if c is None:
            raise HTTPException(
                status_code=404, detail=f"佐证更正不存在: request_key={request_key}"
            )
        if c["canonical_id"] != canonical_id:
            raise HTTPException(
                status_code=409,
                detail=f"佐证更正 {request_key} 属于规范档案 {c['canonical_id']}，"
                       f"不属于 {canonical_id}",
            )


def close_ticket(
    conn: sqlite3.Connection,
    requested_id: str,
    ticket_id: str,
    payload: AnomalyTicketCloseRequest,
) -> AnomalyTicketOut:
    """把 open 处置单结案为 resolved 或 accepted；摘要不符或证明不通过 → 409。"""
    conn.execute("BEGIN IMMEDIATE")
    try:
        exhibit = get_canonical_or_404(conn, requested_id)
        canonical_id = exhibit["id"]
        row = _load_ticket(conn, canonical_id, ticket_id)
        if row["status"] != "open":
            raise HTTPException(
                status_code=409,
                detail=f"处置单 {ticket_id} 已结案（{row['status']}），只能从 open 结案",
            )

        # 摘要核对：读数 / 作废 / 来源优先级 / 日历 / 收据链任一变化都会
        # 改变重算摘要；不一致 → 409 且不写入任何结案记录。
        digest_now, snapshot_now = _current_digest(conn, exhibit, row)
        if digest_now != payload.expected_digest:
            raise HTTPException(
                status_code=409,
                detail="处置单摘要已变化：读数/作废/来源优先级/日历/收据链与预期不一致"
                       f"（当前摘要 {digest_now}），未写入结案记录",
            )

        snapshot_before = json.loads(row["snapshot_json"])
        now_iso = _utc_now_iso()
        if payload.decision == "resolved":
            evidence = payload.evidence
            assert evidence is not None  # pydantic 已校验 resolved 必填佐证
            _verify_evidence(conn, canonical_id, evidence)
            # 重算证明：原事件全部消失（以完整事件记录比对，新出现的事件不
            # 影响原事件已消失的判定），且 calendar=display 剂量未越限。
            before_events = snapshot_before["events"]
            after_events = snapshot_now["events"]
            cleared = all(e not in after_events for e in before_events)
            if not cleared:
                raise HTTPException(
                    status_code=409,
                    detail="重算后快照中的原诊断事件仍未全部消失，不能以 resolved 结案",
                )
            if snapshot_now["dose"]["exceeded"]:
                raise HTTPException(
                    status_code=409,
                    detail="重算后 calendar=display 剂量仍越限，不能以 resolved 结案",
                )
            closure = {
                "decision": "resolved",
                "note": payload.note,
                "expected_digest": payload.expected_digest,
                "evidence": {
                    "receipt_ids": list(evidence.receipt_ids),
                    "correction_request_keys": list(evidence.correction_request_keys),
                },
                "events_kept": None,
                "proof": {
                    "original_events_cleared": True,
                    "dose_exceeded_after": False,
                    "events_before": len(before_events),
                    "events_after": len(after_events),
                },
                "closed_at": now_iso,
            }
        else:
            closure = {
                "decision": "accepted",
                "note": payload.note,
                "expected_digest": payload.expected_digest,
                "evidence": None,
                "events_kept": len(snapshot_now["events"]),
                "proof": None,
                "closed_at": now_iso,
            }

        # 结案只写状态与结案记录：快照与 digest 保持不变（快照不可变）。
        conn.execute(
            "UPDATE anomaly_tickets SET status = ?, closed_at = ?, closure_json = ? "
            "WHERE ticket_id = ? AND status = 'open'",
            (payload.decision, now_iso,
             json.dumps(closure, ensure_ascii=False, sort_keys=True), ticket_id),
        )
        conn.commit()
    except HTTPException:
        conn.rollback()
        raise
    except Exception:
        conn.rollback()
        raise

    row = conn.execute(
        "SELECT * FROM anomaly_tickets WHERE ticket_id = ?", (ticket_id,)
    ).fetchone()
    return AnomalyTicketOut(**_ticket_dict(row, exhibit, requested_id))
