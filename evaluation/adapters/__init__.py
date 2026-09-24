"""Dataset adapters shipped with the RepoGraph evaluation harness."""

from evaluation.adapters.local_tasks import LocalTaskDataset
from evaluation.adapters.swebench import SWEbenchDataset

__all__ = ["LocalTaskDataset", "SWEbenchDataset"]
