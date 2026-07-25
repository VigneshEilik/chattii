from enum import Enum
from typing import Set


class VocabularyLifecycle(Enum):
    """
    State machine lifecycle representing the stage of a Vocabulary artifact.
    Prevents accidental mutations in production environments.
    """

    CREATED = "CREATED"
    TRAINING = "TRAINING"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"

    def can_transition_to(self, target: "VocabularyLifecycle") -> bool:
        """
        Validates transition rules.
        """
        transitions = {
            VocabularyLifecycle.CREATED: {
                VocabularyLifecycle.TRAINING,
                VocabularyLifecycle.ARCHIVED,
            },
            VocabularyLifecycle.TRAINING: {
                VocabularyLifecycle.VALIDATED,
                VocabularyLifecycle.ARCHIVED,
            },
            VocabularyLifecycle.VALIDATED: {
                VocabularyLifecycle.ACTIVE,
                VocabularyLifecycle.ARCHIVED,
            },
            VocabularyLifecycle.ACTIVE: {VocabularyLifecycle.ARCHIVED},
            VocabularyLifecycle.ARCHIVED: set(),
        }
        valid_targets: Set[VocabularyLifecycle] = transitions.get(self, set())
        return target in valid_targets
