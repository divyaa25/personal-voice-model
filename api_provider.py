"""
Abstraction layer for different AI API providers.
Supports: Anthropic Claude, Google Gemini
"""

import os
from abc import ABC, abstractmethod
from typing import Optional


class APIProvider(ABC):
    """Base class for AI API providers."""

    @abstractmethod
    def generate_profile(self, system_prompt: str, user_message: str) -> str:
        """Generate initial style profile."""
        pass

    @abstractmethod
    def refine_profile(self, system_prompt: str, messages: list) -> str:
        """Continue conversation with follow-up questions."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if API credentials are configured."""
        pass


class AnthropicProvider(APIProvider):
    """Anthropic Claude API provider."""

    def __init__(self):
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if self.api_key:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.api_key)
        else:
            self.client = None

    def is_configured(self) -> bool:
        return self.api_key is not None

    def generate_profile(self, system_prompt: str, user_message: str) -> str:
        """Generate initial style profile using Claude."""
        response = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )
        return response.content[0].text

    def refine_profile(self, system_prompt: str, messages: list) -> str:
        """Continue conversation with Claude."""
        response = self.client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1500,
            system=system_prompt,
            messages=messages
        )
        return response.content[0].text


class GeminiProvider(APIProvider):
    """Google Gemini API provider."""

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self.genai = genai
        else:
            self.genai = None

    def is_configured(self) -> bool:
        return self.api_key is not None

    def generate_profile(self, system_prompt: str, user_message: str) -> str:
        """Generate initial style profile using Gemini."""
        model = self.genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt
        )
        response = model.generate_content(user_message)
        return response.text

    def refine_profile(self, system_prompt: str, messages: list) -> str:
        """Continue conversation with Gemini."""
        # Convert Anthropic format to Gemini format if needed
        gemini_messages = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            gemini_messages.append({"role": role, "parts": [msg["content"]]})

        model = self.genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            system_instruction=system_prompt
        )
        chat = model.start_chat(history=gemini_messages)
        # Get the last user message
        last_user_msg = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            None
        )
        if last_user_msg:
            response = chat.send_message(last_user_msg)
            return response.text
        return ""


class GroqProvider(APIProvider):
    """Groq API provider (fast inference)."""

    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if self.api_key:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
        else:
            self.client = None

    def is_configured(self) -> bool:
        return self.api_key is not None

    def generate_profile(self, system_prompt: str, user_message: str) -> str:
        """Generate initial style profile using Groq."""
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_tokens=2000,
            temperature=0.7
        )
        return response.choices[0].message.content

    def refine_profile(self, system_prompt: str, messages: list) -> str:
        """Continue conversation with Groq."""
        # Convert to Groq format (same as OpenAI)
        groq_messages = [{"role": "system", "content": system_prompt}]

        for msg in messages:
            groq_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_messages,
            max_tokens=1500,
            temperature=0.7
        )
        return response.choices[0].message.content


def get_provider() -> APIProvider:
    """
    Get configured API provider based on environment variable.

    Set API_PROVIDER env variable to choose:
    - 'anthropic' (requires ANTHROPIC_API_KEY)
    - 'gemini' (requires GOOGLE_API_KEY, default)
    - 'groq' (requires GROQ_API_KEY)
    """
    provider_name = os.getenv("API_PROVIDER", "gemini").lower()

    if provider_name == "anthropic":
        provider = AnthropicProvider()
        if not provider.is_configured():
            raise EnvironmentError(
                "ANTHROPIC_API_KEY environment variable not set.\n"
                "Set your key: export ANTHROPIC_API_KEY='sk-...'"
            )
        return provider

    elif provider_name == "gemini":
        provider = GeminiProvider()
        if not provider.is_configured():
            raise EnvironmentError(
                "GOOGLE_API_KEY environment variable not set.\n"
                "Set your key: export GOOGLE_API_KEY='your-key-here'"
            )
        return provider

    elif provider_name == "groq":
        provider = GroqProvider()
        if not provider.is_configured():
            raise EnvironmentError(
                "GROQ_API_KEY environment variable not set.\n"
                "Set your key: export GROQ_API_KEY='your-key-here'"
            )
        return provider

    else:
        raise ValueError(
            f"Unknown API_PROVIDER: {provider_name}\n"
            "Supported: 'anthropic', 'gemini', 'groq'"
        )


def list_providers() -> dict:
    """List available providers and their configuration status."""
    providers = {
        "anthropic": AnthropicProvider(),
        "gemini": GeminiProvider(),
        "groq": GroqProvider(),
    }
    env_vars = {
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY"
    }
    return {
        name: {
            "configured": provider.is_configured(),
            "env_var": env_vars[name]
        }
        for name, provider in providers.items()
    }
