"""agentproof — evaluate AI agents by what they DO, not only what they say."""

from .core import (AgentFn, Grader, GraderResult, SuiteReport, Task, TaskReport, Tier, ToolCall,
                   Transcript, TrialResult, run_suite, run_trial)

__all__ = ["AgentFn", "Grader", "GraderResult", "SuiteReport", "Task", "TaskReport", "Tier",
           "ToolCall", "Transcript", "TrialResult", "run_suite", "run_trial"]
__version__ = "0.1.0"
