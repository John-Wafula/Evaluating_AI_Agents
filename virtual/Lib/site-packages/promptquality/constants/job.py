from enum import Enum


class JobStatus(str, Enum):
    unstarted = "unstarted"
    in_progress = "in_progress"
    completed = "completed"
    failed = "failed"
    error = "error"

    @staticmethod
    def is_incomplete(status: "JobStatus") -> bool:
        return status in [JobStatus.unstarted, JobStatus.in_progress]

    @staticmethod
    def is_failed(status: "JobStatus") -> bool:
        return status in [JobStatus.failed, JobStatus.error]
