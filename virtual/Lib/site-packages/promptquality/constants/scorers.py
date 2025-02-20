from enum import Enum


class Scorers(str, Enum):
    completeness_luna = "completeness_nli"
    completeness_plus = "completeness_gpt"
    context_adherence_luna = "adherence_nli"
    context_adherence_plus = "groundedness"
    context_relevance = "context_relevance"
    correctness = "factuality"
    chunk_attribution_utilization_luna = "chunk_attribution_utilization_nli"
    chunk_attribution_utilization_plus = "chunk_attribution_utilization_gpt"
    pii = "pii"
    prompt_injection = "prompt_injection"
    prompt_perplexity = "prompt_perplexity"
    sexist = "sexist"
    tone = "tone"
    toxicity = "toxicity"
    instruction_adherence_plus = "instruction_adherence"
    ground_truth_adherence_plus = "ground_truth_adherence"
