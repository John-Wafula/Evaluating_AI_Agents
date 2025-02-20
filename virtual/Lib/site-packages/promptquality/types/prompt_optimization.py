from pydantic import BaseModel, Field

from promptquality.constants.models import Models


class PromptOptimizationEpochResult(BaseModel):
    key: str
    epoch: int
    prompt: str
    rating: float
    new_prompt: str


class PromptOptimizationResults(BaseModel):
    best_prompt: str
    finished_computing: bool
    epoch: int = 0


class PromptOptimizationConfiguration(BaseModel):
    prompt: str
    evaluation_criteria: str
    task_description: str
    includes_target: bool
    num_rows: int = Field(
        default=30,
        gt=0,
        description=(
            "Number of rows randomly sampled from the dataset to use for training "
            "the optimized prompt. A larger value will produce longer runtime and "
            "incur a higher cost. Galileo research supports that values lower than "
            "30 will not provide an improved prompt."
        ),
    )
    iterations: int = Field(
        default=10,
        gt=0,
        description=(
            "Number of iterations to run the optimization. More iterations will "
            "result in a better prompt, but will take longer to run and cost more. "
            "Galileo research shows that 10 iterations balances quality and cost "
            "to produce optimal results."
        ),
    )
    max_tokens: int = 4096
    temperature: float = 1.0
    generation_model_alias: str = Field(
        default=Models.gpt_35_turbo_16k_0125,
        description=(
            "The model alias to use for generating prompts. The model alias must be "
            "a valid alias within Galileo, see `promptquality.constants.models.Models` "
            "for available models. The default value is the GPT-3.5-turbo "
            "model with a max tokens of 16k and a temperature of 0.125."
        ),
    )
    evaluation_model_alias: str = Field(
        default=Models.gpt_35_turbo_16k_0125,
        description=(
            "The model alias to use for evaluating prompts. The model alias must be "
            "a valid alias within Galileo, see `promptquality.constants.models.Models` "
            "for available models. The default value is the GPT-3.5-turbo "
            "model with a max tokens of 16k and a temperature of 0.125."
        ),
    )
