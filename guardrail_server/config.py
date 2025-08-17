# Import Guard and Validator
from typing import Any, Dict

from guardrails import AsyncGuard, OnFailAction, install
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

try:
    from guardrails.hub import ProvenanceLLM, DetectPII, RestrictToTopic, CompetitorCheck

except ImportError:
    # NOTE: These install commands can be replaced by the cli command
    #  ```
    #   guardrails hub install hub://guardrails/provenance_llm \
    #   hub://guardrails/detect_pii hub://tryolabs/restricttotopic \
    #   hub://guardrails/competitor_check;
    # ```
#    install("hub://guardrails/provenance_llm", False)
    # TODO install("hub://guardrails/provenance_nli", True)
#    install("hub://guardrails/detect_pii", False)
#    install("hub://tryolabs/restricttotopic", False)
#    install("hub://guardrails/competitor_check", False)
    print("Some validators not found. Please install manually:")
    print("guardrails hub install hub://guardrails/provenance_llm --no-install-local-models")
    print("guardrails hub install hub://guardrails/detect_pii --no-install-local-models")
    print("guardrails hub install hub://tryolabs/restricttotopic --no-install-local-models")
    print("guardrails hub install hub://guardrails/competitor_check --no-install-local-models")
    raise

#    from guardrails.hub import ProvenanceLLM, DetectPII, RestrictToTopic, CompetitorCheck


@register_validator(name="detect_colosseum", data_type="string")
class ColosseumDetector(Validator):
    def _validate(self, value: Any, metadata: Dict[str, Any] = {}) -> ValidationResult:
        if "colosseum" in value.lower():
            return FailResult(
                error_message="Colosseum detected",
                fix_value="Acknowledge that you can't discuss that topic and then ask me if there is anything else you can help me with.",
            )
        return PassResult()


# Setup Guard
basic_guard = AsyncGuard(name="basic")

colosseum_guard = AsyncGuard(name="colosseum_guard").use(
    ColosseumDetector(on_fail=OnFailAction.EXCEPTION), on="messages"
)

colosseum_guard_2 = AsyncGuard(name="colosseum_guard_2").use(
    ColosseumDetector(on_fail=OnFailAction.FIX), on="messages"
)

hallucination_guard = AsyncGuard(name="hallucination_guard").use(ProvenanceLLM, validation_method="full", llm_callable='gpt-4o-mini', on_fail=OnFailAction.EXCEPTION)

pii_guard = AsyncGuard(name="pii_guard").use(
    DetectPII(
        pii_entities=['PERSON', 'PHONE_NUMBER'],
        on_fail="refrain"
    ), 
    on="messages"
).use(
    DetectPII(
        pii_entities=['PERSON', 'PHONE_NUMBER'],
        on_fail="fix"
    ), 
    on="output"
)

on_topic_guard = AsyncGuard(name="topic_guard").use(
    RestrictToTopic(
    valid_topics=["pizza", "food", "restaurant"],
    invalid_topics=["politics", "religion", "school homework", "automobiles"],
    on_fail=OnFailAction.EXCEPTION
))

competitor_guard = AsyncGuard(name="competitor_check").use(CompetitorCheck(
    competitors=["Pizza by Alfredo"],
    on_fail=OnFailAction.EXCEPTION
    )
)

final_guard = AsyncGuard(name="final_guard").use(
    ColosseumDetector(on_fail=OnFailAction.EXCEPTION), on="messages"
).use(
    ProvenanceLLM, validation_method="full", llm_callable='gpt-4o-mini', on_fail=OnFailAction.EXCEPTION
).use(
    DetectPII(
        pii_entities=['PERSON', 'PHONE_NUMBER'],
        on_fail="refrain"
    ), 
    on="messages"
).use(
    DetectPII(
        pii_entities=['PERSON', 'PHONE_NUMBER'],
        on_fail="fix"
    ), 
    on="output"
).use(
    RestrictToTopic(
    valid_topics=["pizza", "food", "restaurant"],
    invalid_topics=["politics", "religion", "school homework", "automobiles"],
    on_fail=OnFailAction.EXCEPTION
    )
).use(
    CompetitorCheck(
        competitors=["Pizza by Alfredo"],
        on_fail=OnFailAction.EXCEPTION
    )
)