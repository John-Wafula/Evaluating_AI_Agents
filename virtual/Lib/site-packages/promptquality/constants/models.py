from enum import Enum
from typing import List


class Models(str, Enum):
    # OpenAI models.
    chat_gpt = "ChatGPT (4K context)"
    chat_gpt_16k = "ChatGPT (16K context)"
    gpt_35_turbo = "ChatGPT (4K context)"
    gpt_35_turbo_16k = "ChatGPT (16K context)"
    gpt_35_turbo_16k_0125 = "ChatGPT (16K context, 0125)"
    gpt_35_turbo_instruct = "gpt-3.5-turbo-instruct"
    gpt_4 = "gpt-4 (8K context)"
    gpt_4_turbo = "GPT-4 Turbo"
    gpt_4o = "GPT-4o"
    gpt_4o_mini = "GPT-4o mini"
    gpt_4_turbo_0125 = "GPT-4 Turbo (0125)"
    gpt_4_128k = "gpt-4 (128K context)"
    babbage_2 = "babbage-002"
    davinci_2 = "davinci-002"
    o1_preview = "o1-preview"
    o1_mini = "o1-mini"

    # Azure models.
    azure_chat_gpt = "ChatGPT (4K context) (Azure)"
    azure_chat_gpt_16k = "ChatGPT (16K context) (Azure)"
    azure_gpt_35_turbo = "ChatGPT (4K context) (Azure)"
    azure_gpt_35_turbo_16k = "ChatGPT (16K context) (Azure)"
    azure_gpt_35_turbo_instruct = "gpt-3.5-turbo-instruct (Azure)"
    azure_gpt_4 = "gpt-4 (Azure)"
    azure_gpt_4o = "GPT-4o (Azure)"
    azure_gpt_4o_mini = "GPT-4o mini (Azure)"

    # Vertex AI models.
    text_bison = "text-bison"
    text_bison_001 = "text-bison@001"
    gemini_pro = "gemini-1.0-pro"
    gemini_1_pro = "gemini-1.0-pro"
    gemini_15_flash = "gemini-1.5-flash"
    gemini_15_pro = "gemini-1.5-pro"

    # AWS Bedrock models.
    aws_titan_tg1_large = "AWS - Titan TG1 Large (Bedrock)"
    aws_titan_text_lite_v1 = "AWS - Titan Lite v1 (Bedrock)"
    aws_titan_text_express_v1 = "AWS - Titan Express v1 (Bedrock)"
    cohere_command_r_v1 = "Cohere - Command R v1 (Bedrock)"
    cohere_command_r_plus_v1 = "Cohere - Command R+ v1 (Bedrock)"
    cohere_command_text_v14 = "Cohere - Command v14 (Bedrock)"
    cohere_command_light_text_v14 = "Cohere - Command Light v14 (Bedrock)"
    ai21_j2_mid_v1 = "AI21 - Jurassic-2 Mid v1 (Bedrock)"
    ai21_j2_ultra_v1 = "AI21 - Jurassic-2 Ultra v1 (Bedrock)"
    anthropic_claude_instant_v1 = "Anthropic - Claude Instant v1 (Bedrock)"
    anthropic_claude_v1 = "Anthropic - Claude v1 (Bedrock)"
    anthropic_claude_v2 = "Anthropic - Claude v2 (Bedrock)"
    anthropic_claude_v21 = "Anthropic - Claude v2.1 (Bedrock)"
    anthropic_claude_3_sonnet = "Anthropic - Claude 3 Sonnet (Bedrock)"
    anthropic_claude_3_haiku = "Anthropic - Claude 3 Haiku (Bedrock)"
    anthropic_claude_3_opus = "Anthropic - Claude 3 Opus (Bedrock)"
    anthropic_claude_35_sonnet = "Anthropic - Claude 3.5 Sonnet (Bedrock)"
    anthropic_claude_35_sonnet_v2 = "Anthropic - Claude 3.5 Sonnet v2 (Bedrock)"
    meta_llama2_13b_chat_v1 = "Meta - Llama 2 Chat 13B v1 (Bedrock)"
    meta_llama3_8b_instruct_v1 = "Meta - Llama 3 8B Instruct v1 (Bedrock)"
    meta_llama3_70b_instruct_v1 = "Meta - Llama 3 70B Instruct v1 (Bedrock)"
    meta_llama3_1_8b_instruct_v1 = "Meta - Llama 3.1 8B Instruct v1 (Bedrock)"
    meta_llama3_1_70b_instruct_v1 = "Meta - Llama 3.1 70B Instruct v1 (Bedrock)"
    meta_llama3_1_405b_instruct_v1 = "Meta - Llama 3.1 405B Instruct v1 (Bedrock)"
    meta_llama3_2_1b_instruct = "Meta - Llama 3.2 1B Instruct (Bedrock)"
    meta_llama3_2_3b_instruct = "Meta - Llama 3.2 3B Instruct (Bedrock)"
    meta_llama3_2_11b_instruct = "Meta - Llama 3.2 11B Instruct (Bedrock)"
    meta_llama3_2_90b_instruct = "Meta - Llama 3.2 90B Instruct (Bedrock)"
    mistral_7b_instruct = "Mistral - 7B Instruct (Bedrock)"
    mistral_8x7b_instruct = "Mixtral - 8x7B Instruct (Bedrock)"
    mistral_large = "Mistral - Large (Bedrock)"

    # Writer models.
    palmyra_base = "Palmyra Base"
    palmyra_large = "Palmyra Large"
    palmyra_instruct = "Palmyra Instruct"
    palmyra_instruct_30 = "Palmyra Instruct 30"
    palmyra_beta = "Palmyra Beta"
    silk_road = "Silk Road"
    palmyra_e = "Palmyra E"
    palmyra_x = "Palmyra X"
    palmyra_x_32k = "Palmyra X 32K"
    palmyra_med = "Palmyra Med"
    examworks_v1 = "Exam Works"

    @staticmethod
    def for_customized_scorers() -> List["Models"]:
        return [
            Models.chat_gpt,
            Models.chat_gpt_16k,
            Models.gpt_35_turbo,
            Models.gpt_35_turbo_16k,
            Models.gpt_35_turbo_16k_0125,
            Models.gpt_35_turbo_instruct,
            Models.gpt_4,
            Models.gpt_4_turbo,
            Models.gpt_4o,
            Models.gpt_4o_mini,
            Models.gpt_4_turbo_0125,
            Models.gpt_4_128k,
            Models.azure_chat_gpt,
            Models.azure_chat_gpt_16k,
            Models.azure_gpt_35_turbo,
            Models.azure_gpt_35_turbo_16k,
            Models.azure_gpt_35_turbo_instruct,
            Models.azure_gpt_4,
            Models.azure_gpt_4o,
            Models.azure_gpt_4o_mini,
            Models.gemini_15_flash,
            Models.gemini_15_pro,
        ]
