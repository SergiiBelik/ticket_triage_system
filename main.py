from openai import OpenAI
from pydantic import BaseModel
from typing import Literal, Any, Callable
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from dataclasses import dataclass
import hashlib
import time
import random

client = OpenAI()

instructions = """
You are a technical support ticket triage system.

Your job is to analyze incoming customer support tickets and produce
a structured triage decision.

For each ticket:

1. Determine the initial severity:
   - low: informational requests, minor usability issues, no meaningful impact
   - normal: standard product issue affecting a customer but with a workaround
     or limited impact
   - high: major functionality unavailable, significant customer impact,
     repeated failures, or no reasonable workaround
   - critical: production outage, widespread service failure, severe data
     integrity issue, security issue, or business-critical operation blocked

2. Identify the product area and issue category.

3. Identify a suspected cause only when the ticket contains enough evidence.
   Do not invent a root cause. Use null when there is insufficient evidence.

4. Route the ticket:
   - support: normal troubleshooting, configuration, usage, or known issues
   - sre: infrastructure, availability, latency, widespread service degradation
   - engineering: reproducible product defects or behavior likely requiring
     a code change

5. Determine whether escalation beyond normal Support handling is required.
   If escalation is not required, escalation_team must be null.

6. Identify information required for further diagnosis.
   Return an empty list if no additional information is needed.

7. Draft a concise professional first response to the customer.
   Do not claim a root cause unless it is supported by the ticket.

Be conservative about severity and escalation.
Base decisions only on information provided by the customer.
"""

@dataclass
class ExecutionResult:
    status: Literal[
        "completed", 
        "retryable_failure", 
        "needs_review"]
    next_action_index: int
    error_type: str | None = None

class Triage(BaseModel):
    severity: Literal["low", "normal", "high","critical"]
    product_area: str | None
    category: str | None
    suspected_cause: str | None
    route_to: Literal["support", "sre", "engineering"]
    needs_escalation: bool
    escalation_team: Literal["support", "sre", "engineering"] | None
    missing_information: list[str]
    recommended_first_response: str

class TemporaryTicketError(Exception):
    """Temporary operation failure, retryable."""

class PermanentTicketError(Exception):
    """Permanent operation failure, cannot be retried without fixing an issue."""

class UnknownTicketError(Exception):
    """We don't know if operation succeeded or not despite an error."""

class MockTicketSystem:

    def __init__(self) -> None:
        self.tickets: dict[int, dict[str, Any]] = {}
        self.next_ticket_id = 1001
        self.fail_next: dict[str, str] = {}
        self.applied_actions = {}

    def create_ticket(self, description: str) -> int:
        '''Returns ticket id'''

        ticket_id = self.next_ticket_id
        self.next_ticket_id += 1

        self.tickets[ticket_id] = {
            "ticket_id": ticket_id,
            "description": description,
            "status": "open",
            "priority": None,
            "assigned_team": None,
            "escalated": False,
            "internal_notes": [],
            "response_drafts": []
        }

        return ticket_id

    def assign_team(self, ticket_id: int, team: str):
        self.tickets[ticket_id]["assigned_team"] = team

    def set_priority(self, ticket_id: int, priority: str):
        self.tickets[ticket_id]["priority"] = priority

    def escalate(self, ticket_id: int, team: str):
        self.tickets[ticket_id]["escalated"] = True
        self.tickets[ticket_id]["escalation_team"] = team

    def add_customer_response(self, ticket_id: int, customer_message: str):
        self.tickets[ticket_id]["response_drafts"].append(customer_message)

    def get_ticket(self, ticket_id: int) -> dict[str, Any]:
        return self.tickets[ticket_id]

    def inject_failure(self, action: str, failure_type: Literal["temporary", "permanent", "unknown"]):
        self.fail_next[action] = failure_type

    def _before_action(self, action: str):
        failure = self.fail_next.get(action)

        if failure == "temporary":
            del self.fail_next[action]
            raise TemporaryTicketError(f"Temporary failure executing {action}")

        elif failure == "permanent":
            del self.fail_next[action]
            raise PermanentTicketError(f"Permanent failure executing {action}")

    def _after_action(self, action: str):
        if self.fail_next.get(action) == "unknown":
            del self.fail_next[action]
            raise UnknownTicketError(f"Outcome of {action} is unknown")

    def action_already_applied(
            self,
            ticket_id: int,
            action_id: str,
            action: dict
    ) -> bool:

        key = (ticket_id, action_id)

        fingerprint = hashlib.sha256(json.dumps(action, sort_keys=True).encode("utf-8")).hexdigest()
        existing = self.applied_actions.get(key)

        if existing is None:
            return False
        
        if existing != fingerprint:
            raise PermanentTicketError("Action ID reused with different parameters")
        return True

    def apply_action(
            self,
            ticket_id: int,
            action_id: str,
            action: dict
    ) -> str:
        
        if self.action_already_applied(ticket_id,action_id, action):
            return "already_applied"

        action_name = action["action"]

        if action_name not in ("assign_team", "set_priority", "escalate", "draft_response"):
            raise PermanentTicketError(f"{action_name} is not valid action")

        self._before_action(action_name)

        if action_name == "assign_team":
            self.assign_team(ticket_id, action["team"])

        elif action_name == "set_priority":
            self.set_priority(ticket_id, action["priority"])

        elif action_name == "escalate":
            self.escalate(ticket_id, action["value"])

        elif action_name == "draft_response":
            self.add_customer_response(ticket_id, action["message"])

        fingerprint = hashlib.sha256(json.dumps(action, sort_keys=True).encode("utf-8")).hexdigest()

        self.applied_actions[(ticket_id, action_id)] = fingerprint

        self._after_action(action_name)

        return "applied"

class AuditLogger():

    def __init__(self, file_path: str = "audit_log.jsonl"):
        self.file_path = Path(file_path)

    def record(
            self,
            ticket_id: int,
            operation_id: str,
            action: str,
            status: str,
            parameters: dict,
            error_type: str | None = None,
            action_id: str | None = None
    ) -> None:
        
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ticket_id": ticket_id,
            "operation_id": operation_id,
            "action": action,
            "status": status,
            "parameters": parameters,
            "error_type": error_type,
            "action_id": action_id
        }

        with self.file_path.open("a") as file:
            file.write(json.dumps(event) + "\n")

def create_action_plan(triage: Triage) -> list[dict[str, str]]:

    actions = []

    actions.append({
        "action": "assign_team",
        "team": triage.route_to
    })

     # Set ticket priority
    actions.append({
        "action": "set_priority",
        "priority": triage.severity
    })

    if triage.needs_escalation and triage.escalation_team is None:
        raise ValueError("Escalation team is required when escalation=True")

    if not triage.needs_escalation and triage.escalation_team is not None:
        raise ValueError("Escalation team must be null when escalation=False")

    if triage.needs_escalation:
        actions.append({
            "action": "escalate",
            "value": triage.escalation_team
        })

    actions.append({
        "action": "draft_response",
        "message": triage.recommended_first_response
    })


    return actions

def execute_action_plan(
        actions: list[dict[str, str]], 
        ticket_id: int, 
        ticket_system: MockTicketSystem, 
        audit_logger: AuditLogger,
        plan_id: str,
        start_index: int = 0) -> ExecutionResult:

    if not 0 <= start_index <= len(actions):
        raise ValueError("Invalid start_index")

    allowed_parameters: dict[str, list[str]] = {
        "assign_team": ["team"],
        "set_priority": ["priority"],
        "escalate": ["value"],
        "draft_response": []
    }


    for index in range(start_index, len(actions)):

        action = actions[index]
        action_name = action.get("action", "Unknown")

        action_id = f"{plan_id}:{index}"

        operation_id = str(uuid4())

        parameters = {
            key: action[key]
            for key in allowed_parameters.get(action_name, [])
            if key in action
        }

        audit_logger.record(
            ticket_id = ticket_id, 
            operation_id = operation_id,
            action_id = action_id,
            action = action_name,
            status="started",
            parameters=parameters
        )

        try:
            outcome = ticket_system.apply_action(
                ticket_id,
                action_id,
                action
            )

        except Exception as exc:

            known_failure = isinstance(exc, (TemporaryTicketError, PermanentTicketError))

            audit_logger.record(
                ticket_id = ticket_id,
                operation_id = operation_id,
                action = action_name,
                status=("failed" if known_failure else "outcome_unknown"),
                parameters=parameters,
                error_type = type(exc).__name__
            )

            if isinstance(exc, TemporaryTicketError):
                return ExecutionResult(
                    status="retryable_failure",
                    next_action_index=index,
                    error_type=type(exc).__name__
                )

            return ExecutionResult(
                status = "needs_review",
                next_action_index = index,
                error_type = type(exc).__name__
            )

        else:
            audit_logger.record(
                ticket_id = ticket_id,
                operation_id = operation_id,
                action = action_name,
                action_id = action_id,
                status=(
                    "already_applied"
                    if outcome == "already_applied" 
                    else "succeeded"
                ),
                parameters=parameters
            )

    return ExecutionResult(
        status = "completed", 
        next_action_index=len(actions)
        )

def execute_with_retries(
        actions: list[dict[str, str]],
        ticket_id: int,
        plan_id: str,
        ticket_system: MockTicketSystem,
        audit_logger: AuditLogger,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        sleep_fn: Callable[[float], None] = time.sleep,
        start_index: int = 0
) -> ExecutionResult:

    if max_retries < 0 or max_delay < 0 or base_delay < 0:
        raise ValueError("Retry settings cannot be negative")

    retry_count = 0

    while True:
        result = execute_action_plan(actions=actions,ticket_id=ticket_id, ticket_system=ticket_system,
                            audit_logger=audit_logger, plan_id=plan_id, start_index=start_index)

        if result.status == "needs_review":
            return result

        if result.status == "completed":
            return result

        if result.status != "retryable_failure":
            raise RuntimeError(f"Uknown execution status: {result.status}")

        if start_index != result.next_action_index:
            retry_count = 0

        start_index = result.next_action_index

        if retry_count >= max_retries:
            print("Maximum number of retries reached")
            return result

        delay_cap = min(
            max_delay,
            base_delay * (2 ** retry_count)
        )

        delay = random.uniform(0, delay_cap)

        retry_count += 1

        print(f"Temporary failure at action: {start_index}.")
        print(f"Retry # {retry_count}/{max_retries}")
        print(f"in {delay:.2f} seconds")

        sleep_fn(delay)
    
def run_stream(user_input: str, previous_response_id:str|None)->tuple[str, str, Triage]:

    with client.responses.stream(
        model="gpt-6-luna",
        input = user_input,
        previous_response_id=previous_response_id,
        instructions = instructions,
        text_format=Triage
    ) as stream:

        response_id = None
        response = None

        for event in stream:
            # if event.type == "response.output_text.delta":
            #     print(event.delta, end="", flush=True)

            if event.type == "response.completed":
                triage = event.response.output_parsed
                response = triage.model_dump_json(indent=2)
                response_id = event.response.id

        if response_id is None:
            raise RuntimeError("Stream was interrupted")

        if response is None:
            raise RuntimeError("No response")

    return response_id, response, triage

def test_failure_recovery():
    ticket_system = MockTicketSystem()
    ticket_id = ticket_system.create_ticket("Checkout unavailable")

    audit_logger = AuditLogger("test_audit.jsonl")

    actions:list[dict[str, str]] = [        
        {
            "action": "assign_team",
            "team": "sre"
        },
        {
            "action": "set_priority",
            "priority": "critical"
        },
        {
            "action": "escalate",
            "value": "sre"
        },
        {
            "action": "draft_response",
            "message": "We are invesgitating the issue"
        }
    ]

    ticket_system.inject_failure("set_priority", "temporary")

    result = execute_action_plan(actions,ticket_id,ticket_system,audit_logger)

    print(f"First execution: {result}")
    print(json.dumps(ticket_system.get_ticket(ticket_id),indent=2))

    if result.status == "retryable_failure":
        result = execute_action_plan(actions,ticket_id,ticket_system,audit_logger,start_index=result.next_action_index)

    print(f"Recovery execution: {result}")
    print(json.dumps(ticket_system.get_ticket(ticket_id),indent=2))

def test_idempotency():

    ticket_system = MockTicketSystem()
    audit_logger = AuditLogger("idempotency_test.jsonl")

    ticket_id = ticket_system.create_ticket(
        "Checkout is failing"
    )

    actions = [
        {"action": "assign_team", "team": "support"},
        {"action": "set_priority", "priority": "normal"},
        {
            "action": "draft_response",
            "message": "Please provide the error details."
        }
    ]

    plan_id = str(uuid4())

    # First execution
    result1 = execute_action_plan(
        actions,
        ticket_id,
        ticket_system,
        audit_logger,
        plan_id=plan_id
    )

    # Execute the EXACT same plan again
    result2 = execute_action_plan(
        actions,
        ticket_id,
        ticket_system,
        audit_logger,
        plan_id=plan_id
    )

    ticket = ticket_system.get_ticket(ticket_id)

    print("First execution:", result1.status)
    print("Second execution:", result2.status)
    print("Number of drafts:", len(ticket["response_drafts"]))

    assert result1.status == "completed"
    assert result2.status == "completed"
    assert len(ticket["response_drafts"]) == 1

def test_automatic_retries():
    ticket_system = MockTicketSystem()
    audit_logger = AuditLogger()

    ticket_id = ticket_system.create_ticket("checkout unavailable")
    plan_id = str(uuid4())
    actions = [
        {"action": "assign_team", "team": "sre"},
        {"action": "set_priority", "priority": "critical"},
        {
            "action": "draft_response",
            "message": "We are investigating this issue."
        }
    ]

    ticket_system.inject_failure("set_priority", "temporary")

    result = execute_with_retries(actions,ticket_id,plan_id, ticket_system,audit_logger)

    print(result)

def test_retry_exhaustion():
    from unittest.mock import patch

    ticket_system = MockTicketSystem()
    audit_logger = AuditLogger("exhaustion_test.jsonl")

    actions = [
        {"action": "set_priority",
         "priority": "low"}
    ]

    ticket_id = ticket_system.create_ticket("user cannot login")
    plan_id = str(uuid4())
    with patch.object(
        ticket_system,
        "_before_action",
        side_effect=TemporaryTicketError("Service Unavailable")
    ):
        result = execute_with_retries(
            actions,
            ticket_id,
            plan_id,
            ticket_system,
            audit_logger

        )

    assert result.status == "retryable_failure"
    assert result.next_action_index == 0
    print(ticket_system.get_ticket(ticket_id))

def reconcile_unknown_outcome(
        result: ExecutionResult,
        ticket_id: int,
        actions: list[dict],
        ticket_system: MockTicketSystem,
        audit_logger: AuditLogger,
        plan_id: str
) -> ExecutionResult:

    if result.status != "needs_review" or result.error_type != "UnknownTicketError":
        return result

    index = result.next_action_index

    if not 0 <= index < len(actions):
        raise ValueError("Invalid action index")

    action_id = f"{plan_id}:{index}"
    action = actions[index]

    print(f"\nReconciling index: {action_id}")

    try:
        was_applied = ticket_system.action_already_applied(ticket_id, action_id, action)

    except Exception as exc:
        print(f"Reconciliation failed: {type(exc).__name__}")
        return result

    audit_logger.record(
        ticket_id,
        operation_id = str(uuid4()),
        action=action["action"],
        action_id = action_id,
        status=("Action was already applied" if was_applied else "Unknown"),
        parameters={}
    )

    if not was_applied:
        print("Action receipt is unknown. Manual review is required")
        return result

    print("Action confirmed to be applied. Resuming execution.")

    return execute_with_retries(actions,ticket_id,plan_id,ticket_system,audit_logger, start_index=index)

def test_reconciliation():

    ticket_system = MockTicketSystem()
    audit_logger = AuditLogger("reconciliation_test.jsonl")

    ticket_id = ticket_system.create_ticket("Users cannot checkout")
    plan_id = str(uuid4())

    actions = [
        {
            "action": "assign_team",
            "team": "sre"
        },
        {
            "action": "set_priority",
            "priority": "major"
        },
        {
            "action": "escalate",
            "value": "sre"
        },
        {
            "action": "draft_response",
            "message": "We are invesgitating the issue"
        }
    ]

    ticket_system.inject_failure("set_priority", "unknown")

    result = execute_with_retries(actions,ticket_id,plan_id,ticket_system,audit_logger)

    print(f"Result before reconciliation: {result}")
    assert result.status == "needs_review"
    assert result.next_action_index == 1

    # Priority was actually updated
    ticket = ticket_system.get_ticket(ticket_id)
    assert ticket["priority"] == "major"

    # Remaining actions did not execute
    assert ticket["escalated"] is False
    assert len(ticket["response_drafts"]) == 0

    result = reconcile_unknown_outcome(result,ticket_id,actions,ticket_system,audit_logger, plan_id)

    print(f"Result AFTER reconciliation: {result}")
    ticket = ticket_system.get_ticket(ticket_id)
    assert result.status=="completed"
    assert ticket["escalated"] is True
    assert ticket["priority"] == "major"
    assert len(ticket["response_drafts"]) == 1

    print("Reconciliation test passed!")


def main():

    ticket_system = MockTicketSystem()
    ticket_system.inject_failure("draft_response", "unknown")
    audit_logger = AuditLogger()
    previous_response_id = None
    active_ticket_id = None
    while True:
        if active_ticket_id is None:
            print("\n*** Enter your ticket details ***")
            print('type "exit" to quit, "new" to enter new ticket')
        
        incoming_ticket = input("\nYou: ").strip()
        user_input = incoming_ticket.lower()

        if user_input == "exit":
            break
        if user_input == "new":
            previous_response_id = None
            active_ticket_id = None
            continue

        if not user_input:
            continue

        if active_ticket_id is None:
            active_ticket_id = ticket_system.create_ticket(user_input)
           
        previous_response_id, response, triage = run_stream(user_input, previous_response_id)
        actions = create_action_plan(triage)

        plan_id = str(uuid4())

        result = execute_with_retries(
            actions,
            active_ticket_id,
            plan_id,
            ticket_system,
            audit_logger,
            max_retries = 3,
            base_delay = 0.5,
            max_delay = 8.0
        )

        print(f"\nExecution status: {result.status}")

        if result.status == "retryable_failure":
            print(
                f"Temporary failure at action index {result.next_action_index}."
                "This action is eligible for controlled retry"
            )
        elif result.status == "needs_review":
            print(f"Action requires review: {result.error_type}")
        elif result.status == "completed":
            print("All actions executed successfully")

        print(f"\nTicket {active_ticket_id} updated")
        print(json.dumps(ticket_system.get_ticket(active_ticket_id), indent=2))

if __name__ == "__main__":
    # main()
    # test_automatic_retries()
    # test_retry_exhaustion()
    test_reconciliation()
