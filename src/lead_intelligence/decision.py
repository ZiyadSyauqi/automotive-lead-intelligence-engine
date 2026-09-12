"""Policy demo yang terpisah dari model; tidak menerima target conversion."""

from dataclasses import asdict, dataclass
from typing import Mapping

import numpy as np
import pandas as pd

# Asumsi demo, bukan threshold tervalidasi atau hasil optimasi test outcome.
POLICY = {
    "high_percentile": 90.0,  # Kapasitas fokus maksimal sekitar top 10%.
    "medium_percentile": 60.0,  # Top 40% tetap mendapat follow-up standar.
    "near_purchase_days": 14,  # Horizon pendek untuk eskalasi saat reachable.
    "meaningful_interactions": 3,  # Proxy engagement di window tujuh hari, bukan recency event.
    "followup_limit": 5,  # Cegah pengulangan kontak agresif saat respons lambat.
    "slow_response_hours": 48,  # Dipakai bersama follow-up count, bukan sendiri.
    "urgent_max_fraction": .10,  # Sanity limit batch, bukan target distribusi.
}
PRIORITIES = ("URGENT", "HIGH", "MEDIUM", "LOW")
ACTIONS = {
    "CONTACT_NOW": "Hubungi lead secepatnya setelah cek izin dan jam kontak",
    "CONFIRM_APPOINTMENT": "Konfirmasi appointment yang sudah dijadwalkan",
    "FOLLOW_UP_TEST_DRIVE": "Follow up pengalaman test drive dan kebutuhan berikutnya",
    "RETRY_CONTACT": "Periksa kanal dan waktu yang tepat untuk mencoba kontak kembali",
    "NURTURE": "Masukkan ke nurture; hindari follow-up berulang saat ini",
    "STANDARD_FOLLOW_UP": "Lakukan follow-up standar untuk memahami kebutuhan lead",
    "REVIEW_LEAD_DATA": "Periksa data operasional sebelum menentukan kontak berikutnya",
}
REASONS = {
    "HIGH_MODEL_SCORE": "Lead termasuk kelompok score tertinggi di batch ini",
    "MID_MODEL_SCORE": "Lead berada di kelompok score menengah di batch ini",
    "LOW_MODEL_SCORE": "Lead berada di kelompok score bawah di batch ini",
    "SHORT_PURCHASE_TIMELINE": "Rencana pembelian maksimal 14 hari",
    "CONTACTABLE": "Status kontak tercatat reachable",
    "NOT_CONTACTABLE": "Status kontak tercatat unreachable",
    "INTERMITTENT_CONTACT": "Kontak belum konsisten; prioritaskan pemulihan kontak",
    "APPOINTMENT_SCHEDULED": "Appointment sudah dijadwalkan",
    "APPOINTMENT_ATTENDED": "Lead sudah menghadiri appointment",
    "TEST_DRIVE_COMPLETED": "Test drive sudah selesai",
    "MEANINGFUL_ENGAGEMENT": "Ada minimal tiga interaksi dalam window observasi",
    "LOW_RECENT_ENGAGEMENT": "Tidak ada engagement bermakna menurut proxy policy",
    "HIGH_RESPONSE_LATENCY": "Rata-rata respons minimal 48 jam",
    "FOLLOW_UP_LIMIT_REACHED": "Sudah ada minimal lima upaya follow-up",
    "UNKNOWN_OPERATIONAL_STATUS": "Status operasional hilang atau belum dikenali",
    "OPERATIONAL_DATA_INCOMPLETE": "Ada angka operasional yang hilang atau tidak valid",
}


@dataclass(frozen=True)
class DecisionResult:
    priority: str
    recommended_action: str
    reason_codes: tuple[str, ...]


def number(value, maximum: float, integer: bool = False) -> float | None:
    """Nilai invalid menjadi unknown, bukan otomatis nol."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value) or value < 0 or value > maximum or (integer and not value.is_integer()):
        return None
    return value


def decide(lead: Mapping, model_score: float, score_percentile: float) -> DecisionResult:
    """Score dipakai lewat percentile batch; aturan pertama yang cocok menang."""
    score_percentile = number(score_percentile, 100)
    if number(model_score, 1) is None or score_percentile is None:
        raise ValueError("Score harus finite dalam [0,1]; percentile dalam [0,100]")
    high = score_percentile >= POLICY["high_percentile"]
    middle = score_percentile >= POLICY["medium_percentile"]
    reasons = ["HIGH_MODEL_SCORE" if high else "MID_MODEL_SCORE" if middle else "LOW_MODEL_SCORE"]

    def result(priority, action, *codes):
        return DecisionResult(priority, action, tuple(reasons + list(codes)))

    contact = lead.get("contactability")
    if not isinstance(contact, str) or contact not in {"reachable", "intermittent", "unreachable"}:
        return result("MEDIUM", "REVIEW_LEAD_DATA", "UNKNOWN_OPERATIONAL_STATUS")
    if contact != "reachable":
        return result("MEDIUM" if middle else "LOW", "RETRY_CONTACT",
                      "NOT_CONTACTABLE" if contact == "unreachable" else "INTERMITTENT_CONTACT")
    appointment, drive = lead.get("appointment_activity"), lead.get("test_drive_activity")
    if (not isinstance(appointment, str) or appointment not in {"none", "scheduled", "attended", "missed"}
            or not isinstance(drive, str) or drive not in {"none", "requested", "completed"}):
        return result("MEDIUM", "REVIEW_LEAD_DATA", "UNKNOWN_OPERATIONAL_STATUS")
    timeline = number(lead.get("purchase_timeline_days"), 365)
    latency = number(lead.get("response_latency_hours"), 168)
    followups = number(lead.get("follow_up_count"), 12, integer=True)
    interactions = number(lead.get("previous_interactions"), 20, integer=True)
    incomplete = any(v is None for v in (timeline, latency, followups, interactions))
    if incomplete:
        reasons.append("OPERATIONAL_DATA_INCOMPLETE")
    if (followups is not None and followups >= POLICY["followup_limit"]
            and latency is not None and latency >= POLICY["slow_response_hours"]
            and appointment != "scheduled"):
        return result("MEDIUM" if high else "LOW", "NURTURE",
                      "FOLLOW_UP_LIMIT_REACHED", "HIGH_RESPONSE_LATENCY")
    if high and not incomplete and timeline <= POLICY["near_purchase_days"]:
        return result("URGENT", "CONTACT_NOW", "SHORT_PURCHASE_TIMELINE", "CONTACTABLE")
    if appointment == "scheduled":
        return result("HIGH", "CONFIRM_APPOINTMENT", "APPOINTMENT_SCHEDULED")
    if drive == "completed":
        return result("HIGH" if high else "MEDIUM", "FOLLOW_UP_TEST_DRIVE", "TEST_DRIVE_COMPLETED")
    if high:
        return result("HIGH", "STANDARD_FOLLOW_UP", "CONTACTABLE")
    if interactions is not None and interactions >= POLICY["meaningful_interactions"]:
        return result("MEDIUM", "STANDARD_FOLLOW_UP", "MEANINGFUL_ENGAGEMENT")
    if appointment == "attended":
        return result("MEDIUM", "STANDARD_FOLLOW_UP", "APPOINTMENT_ATTENDED")
    if middle:
        return result("MEDIUM", "STANDARD_FOLLOW_UP")
    if incomplete:
        return result("LOW", "NURTURE")
    return result("LOW", "NURTURE", "LOW_RECENT_ENGAGEMENT")


def decide_batch(leads: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """scores wajib di-index oleh lead_id; input order dipertahankan, ties tidak dipecah via ID."""
    if leads.empty or "lead_id" not in leads or leads.lead_id.isna().any() or not leads.lead_id.is_unique:
        raise ValueError("Batch harus berisi lead_id unik dan tidak kosong")
    if not scores.index.is_unique or set(scores.index) != set(leads.lead_id):
        raise ValueError("Index score harus tepat cocok dengan lead_id")
    aligned = scores.reindex(leads.lead_id).astype(float)
    if not np.isfinite(aligned).all() or not aligned.between(0, 1).all():
        raise ValueError("Score batch harus finite dalam [0,1]")
    # Persentase lead dengan score STRICTLY lebih rendah: tie besar tidak memicu urgent massal.
    percentile = (aligned.rank(method="min") - 1) / len(aligned) * 100
    rows = []
    for lead, score, pct in zip(leads.to_dict("records"), aligned.to_numpy(), percentile.to_numpy()):
        rows.append({"lead_id": lead["lead_id"], "model_score": float(score),
                     "score_percentile": float(pct), **asdict(decide(lead, score, pct))})
    return pd.DataFrame(rows)


def summarize_decisions(decisions: pd.DataFrame) -> dict:
    """Hard checks untuk output kontradiktif; keragaman action menjadi diagnostik, bukan quota."""
    if decisions.empty:
        raise ValueError("Tidak bisa merangkum batch kosong")
    if (not decisions.lead_id.is_unique or not decisions.priority.isin(PRIORITIES).all()
            or not decisions.recommended_action.isin(ACTIONS).all()
            or not decisions.reason_codes.map(lambda codes: bool(codes) and all(c in REASONS for c in codes)).all()):
        raise ValueError("Output decision tidak valid")
    urgent = decisions.priority == "URGENT"
    if urgent.mean() > POLICY["urgent_max_fraction"] + 1e-12:
        raise ValueError("URGENT melebihi batas sanity policy")
    if not (urgent == (decisions.recommended_action == "CONTACT_NOW")).all():
        raise ValueError("CONTACT_NOW hanya boleh untuk URGENT")
    for row in decisions.itertuples():
        codes = set(row.reason_codes)
        if row.priority == "URGENT" and not {"HIGH_MODEL_SCORE", "SHORT_PURCHASE_TIMELINE", "CONTACTABLE"} <= codes:
            raise ValueError("URGENT tidak punya alasan yang diwajibkan")
        if row.priority == "URGENT" and (row.score_percentile < POLICY["high_percentile"] or "OPERATIONAL_DATA_INCOMPLETE" in codes):
            raise ValueError("URGENT bertentangan dengan score/data operasional")
        required = {"CONFIRM_APPOINTMENT": "APPOINTMENT_SCHEDULED",
                    "FOLLOW_UP_TEST_DRIVE": "TEST_DRIVE_COMPLETED",
                    "REVIEW_LEAD_DATA": "UNKNOWN_OPERATIONAL_STATUS"}
        if row.recommended_action in required and required[row.recommended_action] not in codes:
            raise ValueError("Action tidak didukung reason code")
        if codes & {"NOT_CONTACTABLE", "INTERMITTENT_CONTACT"} and row.recommended_action != "RETRY_CONTACT":
            raise ValueError("Lead sulit dihubungi harus diarahkan ke RETRY_CONTACT")
    high_actions = decisions.loc[decisions.score_percentile >= POLICY["high_percentile"], "recommended_action"].nunique()
    def distribution(column, values):
        return {value: {"count": int((decisions[column] == value).sum()),
                        "percent": round(float((decisions[column] == value).mean() * 100), 6)} for value in values}
    return {"rows": len(decisions), "priority_distribution": distribution("priority", PRIORITIES),
            "action_distribution": distribution("recommended_action", ACTIONS),
            "sanity_checks": {"valid_outputs": True, "urgent_is_minority": True,
                              "high_score_action_count": int(high_actions)},
            "warnings": ["Action pada kelompok score tinggi homogen; cek konteks batch, jangan paksa quota action."]
                        if high_actions < 2 else []}


def format_decision(result: DecisionResult) -> str:
    return (f"Priority: {result.priority}\nTindakan: {ACTIONS[result.recommended_action]}\nAlasan:\n"
            + "\n".join(f"- {REASONS[code]}" for code in result.reason_codes))
