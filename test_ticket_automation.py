from uuid import uuid4
import pytest
import json
from main import (
    MockTicketSystem,
    AuditLogger,
    execute_with_retries,
    reconcile_unknown_outcome
)
# ----------------------
# Shared test setup
# -----------------------
@pytest.fixture
def test_environment(tmp_path):

    ticket_system = MockTicketSystem()
    audit_path = tmp_path/"audit_log.jsonl"
    audit_logger = AuditLogger(str(audit_path))

    ticket_id = ticket_system.create_ticket("Checkout is failing")

    actions = [
        {"action": "assign_team", "team": "sre"},
        {"action": "set_priority", "priority": "critical"},
        {"action": "escalate", "value":"sre"},
        {"action": "draft_response", "message":"We are investigating."}]

    return (
        ticket_system,
        audit_logger,
        ticket_id,
        actions,
        audit_path
    )


# ----------------------------
# Test successfull execution
# ----------------------------
def test_successful_execution(test_environment):

    ticket_system, audit_logger, ticket_id, actions, _ = test_environment

    result = execute_with_retries(actions,ticket_id,plan_id=str(uuid4()), ticket_system=ticket_system,
                         audit_logger=audit_logger)

    ticket = ticket_system.get_ticket(ticket_id)

    assert result.status == "completed"
    assert ticket["assigned_team"] == "sre"
    assert ticket["priority"] == "critical"
    assert ticket["escalated"] is True
    assert len(ticket["response_drafts"]) == 1

# --------------------
# Test idempotency
# --------------------
def test_itempotency(test_environment):
    system, logger, ticket_id, actions, _ = test_environment
    plan_id = str(uuid4())

    for _ in range(2):
        result = execute_with_retries(
            actions,
            ticket_id,
            plan_id,
            system,
            logger
        )

        assert result.status == "completed"

    ticket = system.get_ticket(ticket_id)
    assert len(ticket["response_drafts"]) == 1

# -----------------------------------
# Test temporary failure recovery
# -----------------------------------

def test_temp_failure_recovery(test_environment):
    system, logger, ticket_id, actions, _ = test_environment

    system.inject_failure("set_priority", "temporary")
    plan_id = str(uuid4())

    result = execute_with_retries(
        actions,
        ticket_id,
        plan_id,
        system,
        logger
    )

    ticket = system.get_ticket(ticket_id)

    assert result.status == "completed"
    assert ticket["priority"] == "critical"

# --------------------------
# Test permanent failure
# --------------------------

def test_permanent_failure(test_environment):
    system, logger, ticket_id, actions, _ = test_environment

    system.inject_failure("set_priority", "permanent")

    result = execute_with_retries(
        actions,
        ticket_id,
        str(uuid4()),
        system,
        logger
    )

    ticket = system.get_ticket(ticket_id)

    assert result.status == "needs_review"
    assert result.error_type == "PermanentTicketError"

    assert ticket["assigned_team"] == "sre"
    assert ticket["priority"] is None
    assert ticket["escalated"] is False
    assert len(ticket["response_drafts"]) == 0

# --------------------------------------
# Test unknown outcome reconciliation
# --------------------------------------

def test_unknown_outcome_reconciliation(test_environment):
    system, logger, ticket_id, actions, _ = test_environment

    system.inject_failure("set_priority", "unknown")

    plan_id = str(uuid4())

    result = execute_with_retries(
        actions,
        ticket_id,
        plan_id,
        system,
        logger
    )
    ticket = system.get_ticket(ticket_id)

    assert result.status == "needs_review"
    assert result.error_type == "UnknownTicketError"

    assert ticket["priority"] == "critical"

    
    result = reconcile_unknown_outcome(
        result,
        ticket_id,
        actions,
        system,
        logger,
        plan_id
    )
    ticket = system.get_ticket(ticket_id)

    assert result.status == "completed"
    assert ticket["escalated"] is True
    assert len(ticket["response_drafts"]) == 1

# ---------------------------------------
# Test Audit logging
# ---------------------------------------

def test_audit_logging(test_environment):
    system, logger, ticket_id, actions, audit_path = test_environment

    result = execute_with_retries(
        actions,
        ticket_id,
        str(uuid4()),
        system,
        logger
    )

    assert result.status == "completed"

    events = [ json.loads(line) for
        line in audit_path.read_text().splitlines()
    ]

    started = [e for e in events if e["status"] == "started"]
    succeeded = [e for e in events if e["status"] == "succeeded"]

    assert len(started) == 4
    assert len(succeeded) == 4

    started_ids = {e["operation_id"] for e in started}
    succeeded_ids = {e["operation_id"] for e in succeeded}

    assert started_ids == succeeded_ids

    