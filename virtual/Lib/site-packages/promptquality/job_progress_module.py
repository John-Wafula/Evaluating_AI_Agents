from random import random
from time import sleep
from typing import Optional

from pydantic import UUID4
from tqdm.auto import tqdm

from promptquality.constants.job import JobStatus
from promptquality.constants.scorers import Scorers
from promptquality.helpers import get_job_status, get_run_scorer_jobs
from promptquality.types.config import Config
from promptquality.utils.logger import logger


def job_progress(job_id: Optional[UUID4] = None, config: Optional[Config] = None) -> UUID4:
    backoff = random()
    job_status = get_job_status(job_id, config)
    if JobStatus.is_incomplete(job_status.status):
        job_progress_bar = tqdm(
            total=job_status.steps_total,
            position=0,
            leave=True,
            desc=job_status.progress_message,
        )
        while JobStatus.is_incomplete(job_status.status):
            sleep(backoff)
            job_status = get_job_status(job_id, config)
            job_progress_bar.set_description(job_status.progress_message)
            job_progress_bar.update(job_status.steps_completed - job_progress_bar.n)
            backoff = random()
        job_progress_bar.close()
    logger.debug(f"Job {job_id} status: {job_status.status}.")
    if JobStatus.is_failed(job_status.status):
        raise ValueError(f"Job failed with error message {job_status.error_message}.") from None
    print("Initial job complete, executing scorers asynchronously. Current status:")
    scorer_jobs_status(config=config)
    return job_status.id


def scorer_jobs_status(
    project_id: Optional[UUID4] = None, run_id: Optional[UUID4] = None, config: Optional[Config] = None
) -> None:
    scorer_jobs = get_run_scorer_jobs(project_id, run_id)
    for job in scorer_jobs:
        if job.request_data is None or job.request_data.prompt_scorer_settings is None:
            logger.debug(f"Scorer job {job.id} has no scorer settings.")
            continue
        scorer_name = job.request_data.prompt_scorer_settings.scorer_name
        try:
            scorer_name = Scorers(scorer_name).name
        except ValueError:
            pass
        logger.debug(f"Scorer job {job.id} has scorer {scorer_name}.")
        if JobStatus.is_incomplete(job.status):
            print(f"{scorer_name.lstrip('_')}: Computing 🚧")
        elif JobStatus.is_failed(job.status):
            print(f"{scorer_name.lstrip('_')}: Failed ❌, error was: {job.error_message}")
        else:
            print(f"{scorer_name.lstrip('_')}: Done ✅")
