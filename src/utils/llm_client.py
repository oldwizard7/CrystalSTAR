"""LLM client for structure generation (OpenAI v1.0+ compatible)"""

from typing import List, Dict
import logging
import os
from openai import OpenAI

from src.core.structure import CrystalStructure

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Unified LLM client supporting multiple providers (OpenAI/Claude)
    Updated for openai>=1.0 API.
    """

    def __init__(self, config: Dict):
        self.config = config
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.95)
        self.max_tokens = config.get("max_tokens", 2048)
        # Whether to log full prompts for debugging/traceability
        self.print_prompts = config.get("print_prompts", True)
        self.prompt_log_path = config.get("prompt_log_path") or os.getenv("PROMPT_LOG_PATH")
        self.prompt_logger = None

        # Load API key
        api_key_env = config.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.getenv(api_key_env)

        if not self.api_key:
            logger.warning(f"API key not found in environment: {api_key_env}")

        # Init OpenAI client
        if self.provider == "openai":
            self.client = OpenAI(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

        # Setup prompt logger if requested
        if self.prompt_log_path:
            from pathlib import Path

            path = Path(self.prompt_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)

            self.prompt_logger = logging.getLogger(f"prompt_logger_{id(self)}")
            self.prompt_logger.setLevel(logging.INFO)
            self.prompt_logger.propagate = False

            handler = logging.FileHandler(path, mode="a", encoding="utf-8")
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            # Avoid duplicate handlers if multiple LLMClient instances share the same file
            if not self.prompt_logger.handlers:
                self.prompt_logger.addHandler(handler)

            logger.info(f"LLM prompts will be saved to: {path}")

    # ===============================================================
    # HIGH-LEVEL PUBLIC METHODS
    # ===============================================================

    def generate_structures(
        self,
        parent_structures: List[CrystalStructure],
        prompt_template: str,
        num_children: int = 5,
        **kwargs
    ) -> List[str]:
        """
        Generate new crystal structures using the prompt template.
        """
        prompt = self._build_prompt(
            parent_structures, prompt_template, num_children, **kwargs
        )
        logger.info(f"Generating {num_children} structures using {self.model}")

        return self._generate_openai(prompt, num_children)

    def generate(self, prompt: str, n: int = 1) -> List[str]:
        """
        Simple generate interface (no template)
        """
        logger.info(f"Generating {n} response(s) using {self.model}")
        return self._generate_openai(prompt, n)
    
    #LLM prompt for reflection
    def generate_reflection(self, reflection_data: Dict) -> str:
        """
        Generate reflection text about the search process.
        """
        prompt = f"""
Analyze the following crystal structure search results:

Iteration: {reflection_data['iteration']}
Valid structures: {reflection_data['valid_rate']:.1f}%
Metastable structures: {reflection_data['metastable_rate']:.1f}%
Best decomposition energy: {reflection_data['best_ed']:.3f} eV/atom
Average decomposition energy: {reflection_data['avg_ed']:.3f} eV/atom
Structural diversity: {reflection_data['diversity']:.2f}

Recent history:
{reflection_data.get('history', 'None')}

Based on these results, suggest 2–3 specific strategies to improve the next iteration.
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content


    # ===============================================================
    # INTERNAL HELPERS
    # ===============================================================

    def _build_prompt(
        self,
        parents: List[CrystalStructure],
        template: str,
        num_children: int,
        **kwargs
    ) -> str:
        """Build the formatted prompt with reference structures."""
        parent_structures_str = "\n\n".join(
            [f"Parent {i+1}:\n{p.to_poscar()}" for i, p in enumerate(parents)]
        )

        prompt = template.format(
            reference_structures=parent_structures_str,
            reproduction_size=num_children,
            **kwargs
        )
        return prompt

    def _generate_openai(self, prompt: str, n: int) -> List[str]:
        """
        New OpenAI v1.0+ chat completion API.
        """
        try:
            if self.print_prompts and self.prompt_logger:
                self.prompt_logger.info(
                    "MODEL: %s\n%s\n%s",
                    self.model,
                    "=" * 40,
                    prompt,
                )
                logger.info(f"Prompt saved to {self.prompt_log_path} (len={len(prompt)} chars)")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You are an expert materials scientist specializing in crystal structure design."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                n=n,
            )

            return [choice.message.content for choice in response.choices]


        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return []
