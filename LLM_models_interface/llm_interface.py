"""
LLM interface for the MAST failure-mode judge.

Wraps Anthropic (via Vertex AI), Gemini (via Vertex AI), and Ollama behind a single call: 
llm_interface.judge(trace, config) -> JudgeResponse.
"""

import time
import datetime
from pathlib import Path
from collections import defaultdict
import re
import yaml
import json
import random
from dataclasses import dataclass, field
from google import genai
from google.genai import types as genai_types
import google.auth.impersonated_credentials
import google.auth.transport.requests
import google.oauth2.service_account
from google.cloud import secretmanager
from anthropic import AnthropicVertex
import ollama

# Model prices per 1M tokens: (price_in, price_out)
PRICES: dict[str, tuple[float, float]] = {
    "o1":                        (15.00, 60.00),
    "claude-sonnet-4-6":         ( 3.00, 15.00),
    "claude-haiku-4-5":          ( 1.00,  1.25),
    "gemini-2.5-pro":            ( 2.50, 10.00),
    "gpt-4.1":                   ( 2.00,  8.00),
    "o3-mini":                   ( 1.10,  4.40),
    "gpt-5":                     ( 1.25, 10.00),
    "grok-4.3":                  ( 1.25,  2.50),
    "gemini-2.5-flash":          ( 0.30, 2.50),
    "llama3.1:8b":               ( 0.0, 0.0),  
    "qwen2.5:7b":                ( 0.0, 0.0),     
    "llama3.2:3b":               ( 0.0, 0.0),
    "llama3.1:8b":               ( 0.0, 0.0),
    "llama3.1:70b":              ( 0.0, 0.0),
    "llama3.2:3b":               ( 0.0, 0.0),
    "qwen2.5:7b":                ( 0.0, 0.0),
}

FAILURE_MODES = ["1.1","1.2","1.3","1.4","1.5","2.1","2.2","2.3","2.4","2.5","2.6","3.1","3.2","3.3"]

@dataclass
class JudgeConfig:
    name: str = ""
    model: str = ""
    backend: str = "genai"   # "genai" | "anthropic" | "ollama"
    temperature: float = 0.0
    reasoning: bool = False
    shots: int = 0
    slice_n: int | None = None
    system_prompt: str = ""
    definitions_path: str = "../../data/prompts/definitions.txt"
    examples_path: str  = "../../data/prompts/examples.txt"
    dataset_path: str   = ""
    genai_project: str  = "ingka-map-services-dev"
    genai_location: str = "europe-west1"
    ollama_host: str    = "http://localhost:11434"

def load_configs(path: str) -> list[JudgeConfig]:                                                                                                        
      config_path = Path(path).resolve()                                                                                                                   
      config_dir = config_path.parent                                                                                                                      
      with open(config_path) as f:                                                                                                                         
          data = yaml.safe_load(f)                                                                 
      configs = []
      for item in data["experiments"]:
          cfg = JudgeConfig(**item)
          for field in ("definitions_path", "examples_path", "dataset_path"):
              val = getattr(cfg, field)
              if val and not Path(val).is_absolute():
                  setattr(cfg, field, str(config_dir / val))
          configs.append(cfg)
      return configs

@dataclass
class JudgeResponse:
    """Holds the raw and parsed output of a single LLM call, plus usage stats."""

    trace_id: str
    raw_text: str
    model_id: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    annotations: dict[str, int] = field(default_factory=dict)
    cost_usd: float = field(init=False)

    def __post_init__(self):
        price_in, price_out = PRICES.get(self.model_id, (0.0, 0.0))
        self.cost_usd = (
            self.tokens_in * price_in
            + self.tokens_out * price_out
        ) / 1_000_000


def build_judge_prompt(trace: str, definitions: str, examples: str=''):
    prompt = (
    "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
    "In the traces, analyze the system behaviour."
    "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
    "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
    "Also tell me whether the task is successfully completed or not, as a binary yes or no."
    "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
    "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
    "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
    "*** begin of things you should answer *** @@"
    "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
    "B. Whether the task is successfully completed or not: <yes or no>"
    "C. Whether you encounter any of the failure modes or inefficiencies:"
    "1.1 Disobey Task Specification: <yes or no>"
    "1.2 Disobey Role Specification: <yes or no>"
    "1.3 Step Repetition: <yes or no>"
    "1.4 Loss of Conversation History: <yes or no>"
    "1.5 Unaware of Termination Conditions: <yes or no>"
    "2.1 Conversation Reset: <yes or no>"
    "2.2 Fail to Ask for Clarification: <yes or no>"
    "2.3 Task Derailment: <yes or no>"
    "2.4 Information Withholding: <yes or no>"
    "2.5 Ignored Other Agent's Input: <yes or no>"
    "2.6 Action-Reasoning Mismatch: <yes or no>"
    "3.1 Premature Termination: <yes or no>"
    "3.2 No or Incomplete Verification: <yes or no>"
    "3.3 Incorrect Verification: <yes or no>"
    "@@*** end of your answer ***"
    "An example answer is: \n"
    "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
    "B. no \n"
    "C. \n"
    "1.1 no \n"
    "1.2 no \n"
    "1.3 no \n"
    "1.4 no \n"
    "1.5 no \n"
    "1.6 yes \n"
    "2.1 no \n"
    "2.2 no \n"
    "2.3 yes \n"
    "2.4 no \n"
    "2.5 no \n"
    "2.6 yes \n"
    "2.7 no \n"
    "3.1 no \n"
    "3.2 yes \n"
    "3.3 no \n"   
    "Here is the trace: \n"
    f"{trace}"
    "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
    f"{definitions} \n"
    "Here are some examples of the failure modes and inefficiencies: \n"
    f"{examples}"
)
    return prompt


class GCPAuth:
    def __init__(self, project_id: str, impersonate_service_account: str = None, lifetime: int = 3600):
        """
        Initializes the GCPAuth class.
        """
        self.project_id = project_id
        self.impersonate_service_account = impersonate_service_account
        self.lifetime = lifetime
        self.creds = None

    def get_remaining_lifetime(self):
        """        
        Returns the remaining lifetime of the credentials in seconds.
        If the credentials are None, returns 0.
        If the credentials are expired, returns 0.
        If the credentials are valid, returns the remaining lifetime in seconds.
        """
        if self.creds is not None and self.creds.expiry is not None:
            now = datetime.datetime.now(datetime.timezone.utc).timestamp()
            expiration = self.creds.expiry.replace(
                tzinfo=datetime.timezone.utc).timestamp()
            return max(int(expiration) - int(now), 0)
        return 0

    def is_token_expired(self):
        """       
        Checks if the token is expired.
        Returns True if the token is expired, False otherwise.
        """
        remaining_lifetime = self.get_remaining_lifetime()
        return remaining_lifetime == 0

    @staticmethod
    def access_secret_version(creds, project_id, secret_id, version_id="latest"):
        client = secretmanager.SecretManagerServiceClient(
            credentials=creds)
        name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode('UTF-8')

    def get_credentials(self):
        """
        Retrieves the GCP credentials.
        If the credentials are already cached and not expired, returns them.
        If the credentials are not cached or expired, authenticates and returns new credentials.
        """
        if self.creds is not None and not self.is_token_expired():
            # logger.info("Using cached credentials")
            print("Using cached credentials")
            return self.creds

        if self.creds is None:
            # logger.info("No credentials found, authenticating...")
            print("No credentials found, authenticating...")
            target_scopes = [
                "https://www.googleapis.com/auth/cloud-platform"
            ]
            # On Streamlit Cloud, use the service account key stored in secrets.
            _streamlit_creds_loaded = False
            try:
                import streamlit as st
                if "gcp_service_account" in st.secrets:
                    sa_info = dict(st.secrets["gcp_service_account"])
                    creds = google.oauth2.service_account.Credentials.from_service_account_info(
                        sa_info, scopes=target_scopes
                    )
                    print("Authenticated via Streamlit secrets (service account)")
                    _streamlit_creds_loaded = True
            except Exception as e:
                print(f"Streamlit secrets auth failed: {e}")
            if not _streamlit_creds_loaded:
                creds, _ = google.auth.default(scopes=target_scopes)
            if self.impersonate_service_account is not None:
                self.creds = google.auth.impersonated_credentials.Credentials(
                    source_credentials=creds,
                    target_principal=self.impersonate_service_account,
                    target_scopes=target_scopes,
                    lifetime=self.lifetime,  # seconds up to 3600 (1h)
                )
                # logger.info(f"Authenticated with SA {self.impersonate_service_account}")
                print(
                    f"Authenticated with SA {self.impersonate_service_account}")
            else:
                self.creds = creds
                if hasattr(self.creds, "service_account_email"):
                    print(self.creds.service_account_email)
                else:
                    print("Authenticated with default credentials ")
        # logger.info("Refreshing credentials...")
        print("Refreshing credentials...")
        self.creds.refresh(google.auth.transport.requests.Request())
        return self.creds
    
_gcp_auth: GCPAuth | None = None

def _get_gcp_credentials(project: str) -> object:
    global _gcp_auth
    if _gcp_auth is None:
        _gcp_auth = GCPAuth(project_id=project)
    return _gcp_auth.get_credentials()

def _call_genai(model: str, prompt: str, temperature: float, trace_id: str, project: str, location: str, system_prompt="", reasoning: bool = False) -> JudgeResponse:
    """Call the Google GenAI API."""

    client = genai.Client(
    vertexai=True,
    project=project,
    location=location,
    credentials=_get_gcp_credentials(project),
    )

    config = genai_types.GenerateContentConfig(
        system_instruction=system_prompt or None,
        temperature=temperature,
        thinking_config=genai_types.ThinkingConfig(thinking_budget=-1) if reasoning else None,
    )

    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config
    )

    latency = time.perf_counter() - t0
    raw = response.text
    usage = response.usage_metadata
    return JudgeResponse(
        trace_id=trace_id,
        raw_text=raw,
        model_id=model,
        tokens_in=usage.prompt_token_count or 0,
        tokens_out=usage.candidates_token_count or 0,
        latency_s=latency
    )

def _call_anthropic_vertex(model: str, prompt: str, temperature: float, trace_id: str, project: str, location: str, system_prompt: str = "", reasoning: bool = False, max_tokens: int = 16000) -> JudgeResponse:
    """Call a Claude model hosted on Vertex AI Model Garden."""

    client = AnthropicVertex(
        region=location,
        project_id=project,
        credentials=_get_gcp_credentials(project),
    )

    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning:
        # Extended thinking requires budget_tokens and disables temperature
        kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}
    else:
        kwargs["temperature"] = temperature
    if system_prompt:
        kwargs["system"] = system_prompt

    t0 = time.perf_counter()
    response = client.messages.create(**kwargs)
    latency = time.perf_counter() - t0

    raw = next((b.text for b in response.content if b.type == "text"), "")
    return JudgeResponse(
        trace_id=trace_id,
        raw_text=raw,
        model_id=model,
        tokens_in=response.usage.input_tokens or 0,
        tokens_out=response.usage.output_tokens or 0,
        latency_s=latency,
    )


def _call_ollama(model: str, prompt: str, temperature: float, trace_id: str, host: str, system_prompt="") -> JudgeResponse:
    """Call a local Ollama model."""

    client = ollama.Client(host=host)
    t0 = time.perf_counter()
    messages=[]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat(
        model=model,
        messages=messages,
        options={"temperature": temperature}
    )
    latency = time.perf_counter() - t0
    raw = response.message.content
    return JudgeResponse(
        trace_id=trace_id,
        raw_text=raw,
        model_id=model,
        tokens_in=response.prompt_eval_count or 0,
        tokens_out=response.eval_count or 0,
        latency_s=latency
    )

def parse_14_modes(response: str):
    """
    Parse the LLM responses to extract yes/no answers for each failure mode.
    
    Args:
        responses: List of LLM responses evaluating traces
        
    Returns:
        Dictionary mapping failure mode codes to lists of binary values (0 for no, 1 for yes)
    """
    
    # Initialize dictionary with empty lists for each failure mode
    result = {}
    
    try:
            # Clean up the response - remove @@ markers if present
            cleaned_response = response.strip()
            if cleaned_response.startswith('@@'):
                cleaned_response = cleaned_response[2:]
            if cleaned_response.endswith('@@'):
                cleaned_response = cleaned_response[:-2]
            
            # Process each failure mode
            for mode in FAILURE_MODES:
                # Various patterns to match different response formats
                patterns = [
                    # Format with C. prefix and colon
                    rf"C\..*?{mode}.*?(yes|no)",
                    # Format with just C prefix without dot
                    rf"C{mode}\s+(yes|no)",
                    # Format with mode directly (with or without spaces)
                    rf"{mode}\s*[:]\s*(yes|no)",
                    rf"{mode}\s+(yes|no)",
                    # Format with newlines
                    rf"{mode}\s*\n\s*(yes|no)",
                    # Format with C prefix and newlines
                    rf"C\.{mode}\s*\n\s*(yes|no)"
                ]
                
                found = False
                for pattern in patterns:
                    matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                    if matches:
                        # Use the first match
                        value = 1 if matches[0].lower() == 'yes' else 0
                        result[mode] = value
                        found = True
                        break
                
                if not found:
                    # If we still can't find a match, try a more general approach
                    # Look for the mode number followed by any text and then yes/no
                    general_pattern = rf"(?:C\.)?{mode}.*?(yes|no)"
                    match = re.search(general_pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                    
                    if match:
                        value = 1 if match.group(1).lower() == 'yes' else 0
                        result[mode] = value
                    else:
                        # If all attempts fail, default to 'no'
                        print(f"Warning: Could not find mode {mode} in response")
                        result[mode] = 0
                    
    except Exception as e:
        print(f"Error processing response {e}")
        # If there's an error, default to 'no' for all modes for this response
        for mode in FAILURE_MODES:
            if mode not in result:  # Only append if we haven't already
                result[mode] = 0
    
    return result


def build_localise_prompt(steps: list[dict], definitions: str, examples: str = '') -> str:
    """
    Build a judge prompt with explicitly numbered steps for localization.
    
    Args:
        steps: List of step dicts with 'agent', 'content', 'kind', 'metadata'
        definitions: Failure mode definitions text
        examples: Few-shot examples
        
    Returns:
        Prompt asking the model to identify step indices where each mode occurs
    """
    # Format steps with explicit numbering
    steps_text = ""
    for step in steps:
        idx = step.get('metadata', {}).get('step_index', 0)
        agent = step.get('agent', 'Unknown')
        content = step.get('content', '')
        steps_text += f"[Step {idx}] {agent}:\n{content}\n\n"
    
    prompt = (
        "Below I will provide a multiagent system trace with explicitly numbered steps. "
        "Analyze the trace for failure modes and inefficiencies.\n\n"
        "For EACH failure mode, tell me:\n"
        "1. Is it present in the trace? (yes/no)\n"
        "2. If yes, which STEP INDEX(es) does it occur in? (e.g., 'steps 3, 5' or 'step 7' or 'global')\n\n"
        "Some modes are GLOBAL properties (occur throughout or cannot be localized to a single step):\n"
        "- 1.5 Unaware of Termination Conditions: global property\n"
        "- 3.1 Premature Termination: global property\n"
        "For global modes, answer 'global' instead of a step index.\n\n"
        "For modes that occur in multiple steps, list all relevant steps separated by commas.\n\n"
        "Here is the trace with numbered steps:\n"
        f"{steps_text}\n"
        "Now analyze and provide your answers in this format:\n"
        "@@\n"
        "1.1 Disobey Task Specification: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "1.2 Disobey Role Specification: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "1.3 Step Repetition: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "1.4 Loss of Conversation History: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "1.5 Unaware of Termination Conditions: <yes or no>; steps: <'global' or 'n/a'>\n"
        "2.1 Conversation Reset: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "2.2 Fail to Ask for Clarification: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "2.3 Task Derailment: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "2.4 Information Withholding: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "2.5 Ignored Other Agent's Input: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "2.6 Action-Reasoning Mismatch: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "3.1 Premature Termination: <yes or no>; steps: <'global' or 'n/a'>\n"
        "3.2 No or Incomplete Verification: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "3.3 Incorrect Verification: <yes or no>; steps: <step_index(s) or 'global' or 'n/a'>\n"
        "@@\n\n"
        "Here are the failure mode definitions:\n"
        f"{definitions}\n\n"
        "Here are some examples:\n"
        f"{examples}"
    )
    return prompt


def parse_14_modes_with_steps(response: str) -> dict[str, dict]:
    """
    Parse the LLM response to extract yes/no and step indices for each mode.
    
    Returns:
        Dict mapping mode code to {'present': 0/1, 'steps': [list of step indices or 'global']}
    """
    result = {}
    
    try:
        cleaned = response.strip()
        if cleaned.startswith('@@'):
            cleaned = cleaned[2:]
        if cleaned.endswith('@@'):
            cleaned = cleaned[:-2]
        
        for mode in FAILURE_MODES:
            # Look for pattern: "1.1 <name>: yes; steps: 3, 5"
            pattern = rf"{re.escape(mode)}\s+[^:]*:\s*(yes|no)\s*;?\s*steps?:\s*([^;\n]*)"
            match = re.search(pattern, cleaned, re.IGNORECASE)
            
            if match:
                present = 1 if match.group(1).lower() == 'yes' else 0
                steps_str = match.group(2).strip().lower()
                
                # Parse step indices
                if 'global' in steps_str:
                    steps = ['global']
                elif 'n/a' in steps_str or steps_str == 'no' or not steps_str:
                    steps = []
                else:
                    # Parse comma-separated step indices like "3, 5" or "step 3, 5"
                    steps = []
                    for part in steps_str.split(','):
                        part = part.strip()
                        # Extract just the number
                        nums = re.findall(r'\d+', part)
                        if nums:
                            steps.append(int(nums[0]))
                
                result[mode] = {'present': present, 'steps': steps}
            else:
                # Fallback: just look for yes/no
                simple_pattern = rf"{re.escape(mode)}\s+[^:]*:\s*(yes|no)"
                match = re.search(simple_pattern, cleaned, re.IGNORECASE)
                present = 1 if (match and match.group(1).lower() == 'yes') else 0
                result[mode] = {'present': present, 'steps': []}
    
    except Exception as e:
        print(f"Error parsing localization response: {e}")
        for mode in FAILURE_MODES:
            result[mode] = {'present': 0, 'steps': []}
    
    return result


def build_subordinate_localise_prompt(mode_code: str, mode_name: str, steps: list[dict], definitions: str, examples: str = '') -> str:
    """
    Build a localization prompt that assumes the mode IS PRESENT (subordinate to baseline).
    
    Only asks WHERE the mode occurs (step indices), not WHETHER it's present.
    Used as Stage 2 of two-stage pipeline: baseline detects, subordinate localizes.
    
    Args:
        mode_code: e.g., "1.1"
        mode_name: e.g., "Disobey Task Specification"
        steps: List of step dicts with 'agent', 'content', 'kind', 'metadata'
        definitions: Full definitions text
        examples: Few-shot examples
        
    Returns:
        Prompt asking only for step indices where the known-present mode occurs
    """
    # Format steps with explicit numbering
    steps_text = ""
    for step in steps:
        idx = step.get('metadata', {}).get('step_index', 0)
        agent = step.get('agent', 'Unknown')
        content = step.get('content', '')
        steps_text += f"[Step {idx}] {agent}:\n{content}\n\n"
    
    # Global modes: 1.1, 1.5, 3.1 (should not be localized to steps)
    global_modes = ['1.1', '1.5', '3.1']
    
    if mode_code in global_modes:
        prompt = (
            f"Below is a trace where failure mode **{mode_code} {mode_name}** is CONFIRMED PRESENT.\n\n"
            f"This mode is a GLOBAL PROPERTY of the entire trace, not localized to individual steps. "
            f"Answer: GLOBAL\n\n"
            f"Trace:\n{steps_text}"
        )
    else:
        prompt = (
            f"Below is a trace where failure mode **{mode_code} {mode_name}** is CONFIRMED PRESENT.\n\n"
            f"Your task: Identify the SPECIFIC STEP(S) where this mode occurs.\n\n"
            f"Instructions:\n"
            f"1. Do NOT decide if the mode is present (it is). Only locate where.\n"
            f"2. Answer with step index(es): e.g., 'steps 3, 5' or 'step 7' or 'all steps' if throughout.\n"
            f"3. If it is a property of the entire trace rather than specific steps, answer: GLOBAL\n\n"
            f"Trace:\n{steps_text}\n\n"
            f"Answer (step indices or GLOBAL):"
        )
    return prompt


def parse_localized_steps(response: str) -> list | str:
    """
    Parse response from subordinate localizer that ONLY extracts step indices.
    
    Assumes the mode is present (no presence decision needed).
    Returns:
        Either a list of step integers, or the string 'global'
    """
    response_lower = response.lower().strip()
    
    # Check for global
    if 'global' in response_lower:
        return 'global'
    
    # Try to extract step numbers
    steps = []
    numbers = re.findall(r'\d+', response_lower)
    if numbers:
        steps = [int(n) for n in numbers]
        return sorted(list(set(steps)))  # unique and sorted
    
    # If no steps found, return empty list
    return []


# ── Subordinate localizer frameworks ─────────────────────────────────────────
#
# Three prompt configurations for the Subordinated Step Localisation experiment,
# testing different levels of constraint on the model.
#
#   FORCED      — model must output a step coordinate; no escape route.
#   SEMI-FORCED — model can retract the baseline flag via NO_STEP_FOUND.
#   RELAXED     — model sees no baseline signal; evaluates all steps blindly.
#
# All three share _format_steps() and _extract_mode_definition().
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_MODES = {"1.1", "1.5", "3.1"}

_FAILURE_MODES_NAMED = [
    "1.1 Disobey Task Specification",
    "1.2 Disobey Role Specification",
    "1.3 Step Repetition",
    "1.4 Loss of Conversation History",
    "1.5 Unaware of Termination Conditions",
    "2.1 Conversation Reset",
    "2.2 Fail to Ask for Clarification",
    "2.3 Task Derailment",
    "2.4 Information Withholding",
    "2.5 Ignored Other Agent's Input",
    "2.6 Action-Reasoning Mismatch",
    "3.1 Premature Termination",
    "3.2 No or Incomplete Verification",
    "3.3 Incorrect Verification",
]


def _format_steps(steps: list[dict]) -> str:
    parts = []
    for step in steps:
        idx = step.get("metadata", {}).get("step_index", 0)
        agent = step.get("agent", "Unknown")
        content = step.get("content", "")
        parts.append(f"[Step {idx}] {agent}:\n{content}")
    return "\n\n".join(parts) + "\n"


def _extract_mode_definition(mode_code: str, definitions: str) -> str:
    """Return the first paragraph of a mode's definition from definitions.txt."""
    pattern = rf"{re.escape(mode_code)}\s+[^\n]+:\n(.+?)(?=\n\d+\.\d+|\Z)"
    match = re.search(pattern, definitions, re.DOTALL)
    if match:
        return match.group(1).strip()[:400]
    return ""


# ── 1. FORCED framework ───────────────────────────────────────────────────────

def build_forced_localise_prompt(
    mode_code: str,
    mode_name: str,
    steps: list[dict],
    definitions: str,
) -> str:
    """
    FORCED subordinate localizer.

    The baseline verdict is presented as definitive. The model has no escape
    hatch — it must output a step coordinate or GLOBAL. No conversational output
    is permitted by the output rules.

    Returns a single user-turn prompt string (no system/user split needed;
    caller wraps in whatever message schema the judge uses).
    """
    if mode_code in GLOBAL_MODES:
        return (
            f"Failure mode {mode_code} {mode_name} is a global trace property "
            f"and cannot be localised to individual steps.\n\n"
            f"Output exactly one token: GLOBAL"
        )

    defn = _extract_mode_definition(mode_code, definitions)
    steps_text = _format_steps(steps)

    return (
        f"FORENSIC TRACE ANALYSIS — STEP LOCALISATION\n\n"
        f"A full-trace evaluation has definitively confirmed that failure mode "
        f"**{mode_code} {mode_name}** is present in the trace below.\n\n"
        f"Definition:\n{defn}\n\n"
        f"Your only task is to identify the exact step(s) where this failure "
        f"mode manifests. Do not question whether the mode is present. It is.\n\n"
        f"TRACE:\n{steps_text}\n"
        f"OUTPUT RULES — output nothing except one of these exact forms:\n"
        f"  Step X          — single step\n"
        f"  Steps X, Y, Z   — multiple non-contiguous steps\n"
        f"  Steps X-Y       — contiguous range\n"
        f"  GLOBAL          — mode is a property of the whole trace\n"
        f"No explanation. No preamble. No trailing text.\n\n"
        f"Step coordinate:"
    )


def parse_forced_steps(response: str, n_steps: int) -> list[int] | str:
    """
    Parse FORCED localizer output.

    Returns:
        'global'   — mode is trace-level
        list[int]  — validated step indices (empty list if nothing parseable)
    """
    text = response.strip()

    if re.search(r"\bGLOBAL\b", text, re.IGNORECASE):
        return "global"

    # "Steps X-Y" range form
    range_match = re.search(r"[Ss]teps?\s+(\d+)\s*[-–]\s*(\d+)", text)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        return sorted(s for s in range(lo, hi + 1) if 0 <= s < n_steps)

    # "Step X" / "Steps X, Y, Z" — collect all digits
    nums = [int(n) for n in re.findall(r"\d+", text)]
    valid = sorted({n for n in nums if 0 <= n < n_steps})
    return valid  # empty list signals unparseable


# ── 2. SEMI-FORCED framework ──────────────────────────────────────────────────

def build_semi_forced_localise_prompt(
    mode_code: str,
    mode_name: str,
    steps: list[dict],
    definitions: str,
) -> str:
    """
    SEMI-FORCED subordinate localizer.

    The baseline flag is presented as an initial hypothesis, not a verdict.
    The model may confirm + locate, or retract via the sentinel NO_STEP_FOUND
    if step-level inspection does not support the flag.

    Returns a single user-turn prompt string.
    """
    if mode_code in GLOBAL_MODES:
        return (
            f"The baseline analysis flagged failure mode {mode_code} {mode_name} "
            f"as likely present. This mode is a global trace property.\n\n"
            f"Inspect the trace and output one of:\n"
            f"  GLOBAL          — confirmed present across the trace\n"
            f"  NO_STEP_FOUND   — baseline flag was incorrect; mode not present\n\n"
            f"No other text."
        )

    defn = _extract_mode_definition(mode_code, definitions)
    steps_text = _format_steps(steps)

    return (
        f"STEP-LEVEL VERIFICATION — FAILURE MODE LOCALISATION\n\n"
        f"A full-trace baseline analysis flagged failure mode "
        f"**{mode_code} {mode_name}** as likely present.\n\n"
        f"Definition:\n{defn}\n\n"
        f"Perform a careful step-by-step inspection. You have two options:\n"
        f"  A) If the mode IS present: output the exact step(s) where it occurs.\n"
        f"  B) If step-level evidence does NOT support the flag: output NO_STEP_FOUND.\n\n"
        f"TRACE:\n{steps_text}\n"
        f"OUTPUT RULES — output nothing except one of these exact forms:\n"
        f"  Step X          — confirmed at a single step\n"
        f"  Steps X, Y, Z   — confirmed at multiple steps\n"
        f"  Steps X-Y       — confirmed across a contiguous range\n"
        f"  GLOBAL          — confirmed as a whole-trace property\n"
        f"  NO_STEP_FOUND   — baseline flag retracted after step inspection\n"
        f"No explanation. No preamble. No trailing text.\n\n"
        f"Verdict:"
    )


def parse_semi_forced_steps(
    response: str, n_steps: int
) -> list[int] | str | None:
    """
    Parse SEMI-FORCED localizer output.

    Returns:
        None       — model retracted the baseline (NO_STEP_FOUND)
        'global'   — confirmed, trace-level
        list[int]  — confirmed, specific step indices (empty = unparseable)
    """
    text = response.strip()

    if re.search(r"NO_STEP_FOUND", text, re.IGNORECASE):
        return None  # explicit retraction

    if re.search(r"\bGLOBAL\b", text, re.IGNORECASE):
        return "global"

    range_match = re.search(r"[Ss]teps?\s+(\d+)\s*[-–]\s*(\d+)", text)
    if range_match:
        lo, hi = int(range_match.group(1)), int(range_match.group(2))
        return sorted(s for s in range(lo, hi + 1) if 0 <= s < n_steps)

    nums = [int(n) for n in re.findall(r"\d+", text)]
    return sorted({n for n in nums if 0 <= n < n_steps})


# ── 3. RELAXED framework ──────────────────────────────────────────────────────

def build_relaxed_localise_prompt(
    steps: list[dict],
    definitions: str,
) -> str:
    """
    RELAXED (blind) subordinate localizer.

    No baseline signal. The model evaluates every step independently against
    all 14 MAST failure modes and flags any it detects. This is a bottom-up
    scan with no prior hypothesis.

    Returns a single user-turn prompt string.
    """
    steps_text = _format_steps(steps)
    modes_list = "\n".join(f"  {m}" for m in _FAILURE_MODES_NAMED)

    return (
        f"STEP-LEVEL FAILURE MODE ANALYSIS\n\n"
        f"Below is a multi-agent system trace with numbered steps. "
        f"Evaluate each step independently against all 14 MAST failure modes.\n\n"
        f"FAILURE MODES:\n{modes_list}\n\n"
        f"DEFINITIONS:\n{definitions}\n\n"
        f"TRACE:\n{steps_text}\n"
        f"OUTPUT RULES:\n"
        f"  • For each step where you detect one or more failure modes, output:\n"
        f"      Step X: 1.1, 2.3\n"
        f"  • For global-property modes (1.1, 1.5, 3.1) that apply to the whole\n"
        f"    trace rather than a single step, output:\n"
        f"      GLOBAL: 1.1\n"
        f"  • Omit clean steps entirely.\n"
        f"  • No explanations. No preamble. Only the flagged-step lines.\n\n"
        f"Flagged steps:"
    )


def parse_relaxed_steps(
    response: str, n_steps: int
) -> dict[str, list[str]]:
    """
    Parse RELAXED localizer output.

    Returns a dict mapping step key → list of mode codes detected there.
      Key is a stringified step index ('0', '3', …) or 'global'.

    Example: {'3': ['1.3', '2.6'], '7': ['3.2'], 'global': ['1.1']}
    Empty dict if the model found no failure modes.
    """
    result: dict[str, list[str]] = {}

    for line in response.splitlines():
        line = line.strip()
        if not line:
            continue

        # "GLOBAL: 1.1, 3.1"
        global_match = re.match(r"GLOBAL\s*[:\-]\s*(.*)", line, re.IGNORECASE)
        if global_match:
            codes = re.findall(r"\b\d+\.\d+\b", global_match.group(1))
            valid = [c for c in codes if c in FAILURE_MODES]
            if valid:
                result.setdefault("global", []).extend(valid)
            continue

        # "Step X: 1.1, 2.3"
        step_match = re.match(r"[Ss]tep\s+(\d+)\s*[:\-]\s*(.*)", line)
        if step_match:
            idx = int(step_match.group(1))
            if 0 <= idx < n_steps:
                codes = re.findall(r"\b\d+\.\d+\b", step_match.group(2))
                valid = [c for c in codes if c in FAILURE_MODES]
                if valid:
                    result.setdefault(str(idx), []).extend(valid)

    return {k: sorted(set(v)) for k, v in result.items()}


# ── Combined judge+locate prompt (single-call design) ────────────────────────

def _format_steps_with_pct(steps: list[dict]) -> str:
    """Format steps annotated with their position as % through the trace."""
    n = len(steps)
    parts = []
    for i, step in enumerate(steps):
        idx   = step.get("metadata", {}).get("step_index", i)
        pct   = round(100 * i / max(n - 1, 1))
        agent = step.get("agent", "Unknown")
        content = step.get("content", "")
        parts.append(f"[Step {idx} | {pct}%] {agent}:\n{content}")
    return "\n\n".join(parts) + "\n"


def build_judge_locate_prompt(steps: list[dict], definitions: str) -> str:
    """
    Single-call combined judge + forced localizer.

    The model detects which failure modes are present AND locates the step
    where each occurs, reported as a percentage through the trace.
    Global modes (1.1, 1.5, 3.1) are trace-level and output 'global'.
    All other present modes must commit to a specific step — no hedging.
    """
    steps_text = _format_steps_with_pct(steps)
    n_steps = len(steps)

    modes_block = "\n".join(f"  {m}" for m in _FAILURE_MODES_NAMED)

    return f"""You are evaluating a multi-agent system (MAS) trace for failure modes.

Each step is annotated with its position as a percentage through the trace (e.g. [Step 14 | 23%]).

=== TRACE ({n_steps} steps) ===
{steps_text}

=== FAILURE MODE DEFINITIONS ===
{definitions}

=== TASK ===
For each of the 14 failure modes below, decide whether it is present in this trace.
- If absent → output:  <code>: absent
- If present at a specific step → output:  <code>: step <index> (<pct>%)
- If present as a global trace property (only 1.1, 1.5, 3.1 may use this) → output:  <code>: global

Rules:
1. Output exactly 14 lines, one per mode, in order.
2. If a mode is present, you MUST commit to the single most representative step. No hedging.
3. Modes 1.1, 1.5, and 3.1 are trace-level properties — if present, output "global" (never a step index).
4. All other modes require a specific step index if present.
5. Use the step index and percentage exactly as shown in the trace header, e.g. "step 14 (23%)".

Failure modes:
{modes_block}

Output:"""


def parse_judge_locate_response(
    response: str, n_steps: int
) -> dict[str, dict]:
    """
    Parse combined judge+locate response.

    Returns:
        {mode: {'present': bool, 'step_idx': int|None, 'pct': float|None, 'is_global': bool}}
    Modes missing from the response are recorded as present=False.
    """
    result: dict[str, dict] = {}

    for line in response.splitlines():
        line = line.strip().strip("*").strip()
        m = re.search(r"(\d+\.\d+)[^:\n]*:\s*(.*)", line)
        if not m:
            continue
        code = m.group(1).strip()
        if code not in FAILURE_MODES:
            continue
        val = m.group(2).strip().strip("*").strip()

        if re.search(r"\babsent\b", val, re.IGNORECASE):
            result[code] = {"present": False, "step_idx": None, "pct": None, "is_global": False}
        elif re.search(r"\bglobal\b", val, re.IGNORECASE):
            result[code] = {"present": True, "step_idx": None, "pct": None, "is_global": True}
        else:
            step_m = re.search(r"step\s+(\d+)", val, re.IGNORECASE)
            pct_m  = re.search(r"(\d+(?:\.\d+)?)\s*%", val)
            if step_m:
                idx = int(step_m.group(1))
                pct = float(pct_m.group(1)) if pct_m else None
                valid = 0 <= idx < n_steps
                result[code] = {
                    "present":   True,
                    "step_idx":  idx if valid else None,
                    "pct":       pct,
                    "is_global": False,
                }

    # Fill any modes the model omitted
    for code in FAILURE_MODES:
        if code not in result:
            result[code] = {"present": False, "step_idx": None, "pct": None, "is_global": False}

    return result


# ── Batch localizer prompts (judge-then-batch design) ────────────────────────
# After a shared judge call identifies which modes are present, these prompts
# localize ALL detected modes in a single call — one for Forced, one for Semi-Forced.


def build_forced_batch_prompt(
    present_modes: list[str],
    steps: list[dict],
    definitions: str,
) -> str:
    """
    FORCED batch localizer (1 call for all judge-confirmed modes).

    The judge already confirmed these modes are present. The model must output
    step coordinate(s) for each — no escape, no hedging.
    """
    steps_text = _format_steps(steps)
    name_map = {n.split()[0]: n for n in _FAILURE_MODES_NAMED}

    mode_list = "\n".join(f"  • {name_map.get(m, m)}" for m in present_modes)
    output_template = "\n".join(
        f"{name_map.get(m, m)}: global"
        if m in GLOBAL_MODES else
        f"{name_map.get(m, m)}: <step index(es) or range X-Y>"
        for m in present_modes
    )

    return (
        "STEP LOCALISATION — CONFIRMED FAILURE MODES\n\n"
        "A full-trace evaluation confirmed that the following failure modes are present. "
        "Identify the exact step(s) where each one occurs in the trace below.\n\n"
        "Do not question whether a mode is present — it has been confirmed. "
        "Commit to step coordinate(s) for every mode listed. "
        "For trace-level modes (1.1, 1.5, 3.1) output 'global' instead of a step number.\n\n"
        f"Confirmed modes:\n{mode_list}\n\n"
        f"TRACE:\n{steps_text}\n"
        f"DEFINITIONS:\n{definitions}\n\n"
        "Output between @@ markers, one line per mode (replace placeholders with actual values):\n"
        "@@\n"
        f"{output_template}\n"
        "@@"
    )


def build_semi_forced_batch_prompt(
    present_modes: list[str],
    steps: list[dict],
    definitions: str,
) -> str:
    """
    SEMI-FORCED batch localizer (1 call for all judge-flagged modes).

    The judge flagged these modes as likely present. The model inspects step by step
    and may confirm + locate, or retract via NO_STEP_FOUND if evidence is absent.
    """
    steps_text = _format_steps(steps)
    name_map = {n.split()[0]: n for n in _FAILURE_MODES_NAMED}

    mode_list = "\n".join(f"  • {name_map.get(m, m)}" for m in present_modes)
    output_template = "\n".join(
        f"{name_map.get(m, m)}: <global or NO_STEP_FOUND>"
        if m in GLOBAL_MODES else
        f"{name_map.get(m, m)}: <step index(es), range X-Y, or NO_STEP_FOUND>"
        for m in present_modes
    )

    return (
        "STEP LOCALISATION — FLAGGED FAILURE MODES\n\n"
        "A full-trace evaluation flagged the following failure modes as likely present. "
        "Inspect the trace step by step for each:\n"
        "  • If confirmed present: output the step index(es), or 'global' for trace-level modes.\n"
        "  • If step-level evidence does NOT support the flag: output NO_STEP_FOUND.\n\n"
        f"Flagged modes:\n{mode_list}\n\n"
        f"TRACE:\n{steps_text}\n"
        f"DEFINITIONS:\n{definitions}\n\n"
        "Output between @@ markers, one line per mode (replace placeholders with actual values):\n"
        "@@\n"
        f"{output_template}\n"
        "@@"
    )


def parse_batch_localise_response(response: str, n_steps: int) -> dict[str, dict]:
    """
    Parse a FORCED or SEMI-FORCED batch localization response.

    Returns:
        dict[mode_code → {'steps': list[int]|'global'|None, 'retracted': bool}]

    'steps' is None when the model output NO_STEP_FOUND (retracted=True).
    Only modes present in the response are included.
    """
    result: dict[str, dict] = {}

    cleaned = response.strip()
    inner = re.search(r"@@\s*(.*?)\s*@@", cleaned, re.DOTALL)
    if inner:
        cleaned = inner.group(1)

    for line in cleaned.splitlines():
        line = line.strip()
        m = re.search(r"(\d+\.\d+)\s+[^:]*:\s*(.*)", line)  # re.search handles markdown prefixes
        if not m:
            continue
        mode = m.group(1)
        if mode not in FAILURE_MODES:
            continue
        val = m.group(2).strip().strip("*").strip()  # strip trailing ** from bold markdown

        if re.search(r"NO_STEP_FOUND", val, re.IGNORECASE):
            result[mode] = {"steps": None, "retracted": True}
        elif re.search(r"\bglobal\b", val, re.IGNORECASE):
            result[mode] = {"steps": "global", "retracted": False}
        else:
            range_m = re.search(r"(\d+)\s*[-–]\s*(\d+)", val)
            if range_m:
                lo, hi = int(range_m.group(1)), int(range_m.group(2))
                steps_: list | str = sorted(s for s in range(lo, hi + 1) if 0 <= s < n_steps)
            else:
                nums = [int(n) for n in re.findall(r"\d+", val)]
                steps_ = sorted({n for n in nums if 0 <= n < n_steps})
            result[mode] = {"steps": steps_, "retracted": False}

    return result


# ── Full-trace framework prompts (3-call design) ──────────────────────────────
# One LLM call per framework; each call covers all 14 failure modes at once.
# The three frameworks differ in how much guidance/constraint the model receives.


def build_forced_full_prompt(steps: list[dict], definitions: str) -> str:
    """
    FORCED full-trace localizer (1 call, all 14 modes).

    Model receives the trace + definitions + mode checklist and must commit
    to a verdict + step(s) for every mode. No escape hatch.
    """
    steps_text = _format_steps(steps)
    return (
        "FORENSIC TRACE ANALYSIS — FULL-TRACE FAILURE MODE LOCALISATION\n\n"
        "Analyze the multi-agent trace below for all 14 MAST failure modes.\n\n"
        "For EVERY mode you must:\n"
        "  1. Decide if it is present (yes or no).\n"
        "  2. If yes: output the exact step index(es) where it occurs, or 'global' "
        "for trace-level modes (1.1, 1.5, 3.1).\n"
        "  3. If no: output 'n/a' for steps.\n\n"
        "Do not hedge. Commit to a verdict for every single mode.\n\n"
        f"TRACE:\n{steps_text}\n"
        f"FAILURE MODE DEFINITIONS:\n{definitions}\n\n"
        "Output your verdicts between @@ markers, one line per mode:\n\n"
        "@@\n"
        "1.1 Disobey Task Specification: <yes or no>; steps: <step index(es), 'global', or 'n/a'>\n"
        "1.2 Disobey Role Specification: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "1.3 Step Repetition: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "1.4 Loss of Conversation History: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "1.5 Unaware of Termination Conditions: <yes or no>; steps: <'global' or 'n/a'>\n"
        "2.1 Conversation Reset: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "2.2 Fail to Ask for Clarification: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "2.3 Task Derailment: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "2.4 Information Withholding: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "2.5 Ignored Other Agent's Input: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "2.6 Action-Reasoning Mismatch: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "3.1 Premature Termination: <yes or no>; steps: <'global' or 'n/a'>\n"
        "3.2 No or Incomplete Verification: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "3.3 Incorrect Verification: <yes or no>; steps: <step index(es) or 'n/a'>\n"
        "@@"
    )


def build_semi_forced_full_prompt(steps: list[dict], definitions: str) -> str:
    """
    SEMI-FORCED full-trace localizer (1 call, all 14 modes).

    Same checklist as FORCED but the model may output NO_STEP_FOUND per mode
    if it genuinely does not observe evidence for that mode.
    """
    steps_text = _format_steps(steps)
    return (
        "STEP-LEVEL FAILURE MODE ANALYSIS\n\n"
        "Analyze the multi-agent trace below for all 14 MAST failure modes.\n\n"
        "For each mode:\n"
        "  • If it IS present: output 'yes' and the exact step index(es), or 'global' "
        "for trace-level modes (1.1, 1.5, 3.1).\n"
        "  • If it is NOT present after careful inspection: output NO_STEP_FOUND.\n\n"
        "You are not required to find every mode. Only flag what you genuinely observe.\n\n"
        f"TRACE:\n{steps_text}\n"
        f"FAILURE MODE DEFINITIONS:\n{definitions}\n\n"
        "Output your verdicts between @@ markers, one line per mode:\n\n"
        "@@\n"
        "1.1 Disobey Task Specification: <yes or NO_STEP_FOUND>; steps: <step index(es), 'global', or 'n/a'>\n"
        "1.2 Disobey Role Specification: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "1.3 Step Repetition: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "1.4 Loss of Conversation History: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "1.5 Unaware of Termination Conditions: <yes or NO_STEP_FOUND>; steps: <'global' or 'n/a'>\n"
        "2.1 Conversation Reset: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "2.2 Fail to Ask for Clarification: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "2.3 Task Derailment: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "2.4 Information Withholding: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "2.5 Ignored Other Agent's Input: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "2.6 Action-Reasoning Mismatch: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "3.1 Premature Termination: <yes or NO_STEP_FOUND>; steps: <'global' or 'n/a'>\n"
        "3.2 No or Incomplete Verification: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "3.3 Incorrect Verification: <yes or NO_STEP_FOUND>; steps: <step index(es) or 'n/a'>\n"
        "@@"
    )


def build_relaxed_full_prompt(steps: list[dict]) -> str:
    """
    RELAXED full-trace localizer (1 call, no FM list or definitions).

    Model sees only the numbered trace and is asked to identify MAST failure modes
    using its own knowledge. No checklist, no definitions, no nudging.
    """
    steps_text = _format_steps(steps)
    return (
        "STEP-LEVEL FAILURE MODE ANALYSIS\n\n"
        "Below is a multi-agent system trace with numbered steps. "
        "Using your knowledge of the MAST (Multi-Agent System Taxonomy) failure mode "
        "framework, identify any failure modes you observe (codes 1.1–3.3).\n\n"
        "For each step where you detect a failure mode, output:\n"
        "  Step X: <mode code> <mode name>\n"
        "For failure modes that apply to the whole trace (global properties), output:\n"
        "  GLOBAL: <mode code> <mode name>\n"
        "Omit steps where no failure mode is detected.\n\n"
        f"TRACE:\n{steps_text}\n"
        "Flagged steps:"
    )


def parse_full_framework_response(response: str, n_steps: int) -> dict[str, dict]:
    """
    Parse a FORCED or SEMI-FORCED full-trace response.

    Returns:
        dict[mode_code → {'present': 0|1, 'steps': list[int]|'global', 'retracted': bool}]

    'retracted' is True when the model output NO_STEP_FOUND (semi-forced only).
    """
    result: dict[str, dict] = {}

    cleaned = response.strip()
    # unwrap @@ block if present
    inner = re.search(r"@@\s*(.*?)\s*@@", cleaned, re.DOTALL)
    if inner:
        cleaned = inner.group(1)

    for mode in FAILURE_MODES:
        # NO_STEP_FOUND check (semi-forced retraction)
        if re.search(
            rf"{re.escape(mode)}\b[^@\n]*?NO_STEP_FOUND",
            cleaned,
            re.IGNORECASE,
        ):
            result[mode] = {"present": 0, "steps": [], "retracted": True}
            continue

        # "1.1 Name: yes; steps: 3, 5"
        match = re.search(
            rf"{re.escape(mode)}\b[^:]*:\s*(yes|no)\s*[;,]?\s*steps?\s*[:\-]?\s*([^\n@@]*)",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            present = 1 if match.group(1).strip().lower() == "yes" else 0
            steps_str = match.group(2).strip().lower()

            if "global" in steps_str:
                steps: list | str = "global"
            elif not present or "n/a" in steps_str or not steps_str:
                steps = []
            else:
                nums = [int(n) for n in re.findall(r"\d+", steps_str)]
                steps = sorted({n for n in nums if 0 <= n < n_steps})

            result[mode] = {"present": present, "steps": steps, "retracted": False}
        else:
            result[mode] = {"present": 0, "steps": [], "retracted": False}

    return result


def _stratified_sample(data: list[dict], n: int, key: str, seed: int = 42) -> list[dict]:
    """Sample n items proportionally per mas_name, with a fixed seed."""

    rng = random.Random(seed)
    by_mas_name: dict[str, list] = defaultdict(list)
    for item in data:
        by_mas_name[item[key]].append(item)

    total = len(data)
    result: list[dict] = []
    remainder: list[tuple[float, str]] = []

    for mas_name, items in by_mas_name.items():
        exact = n * len(items) / total
        quota = int(exact)
        result.extend(rng.sample(items, min(quota, len(items))))
        remainder.append((exact - quota, mas_name))

    leftover = n - len(result)
    sampled_ids = {id(x) for x in result}
    for _, mas_name in sorted(remainder, reverse=True)[:leftover]:
        pool = [x for x in by_mas_name[mas_name] if id(x) not in sampled_ids]
        if pool:
            chosen = rng.choice(pool)
            result.append(chosen)
            sampled_ids.add(id(chosen))

    rng.shuffle(result)
    return result


def load_dataset(config: JudgeConfig) -> list[dict]:
    with open(config.dataset_path) as f:
        data = json.load(f)
    if config.slice_n is not None and config.slice_n < len(data):
        data = _stratified_sample(data, config.slice_n, key="mas_name", seed=42)
    return data

_MODULE_DIR = Path(__file__).parent

class LLMJudge:
    def __init__(self, config: JudgeConfig):
        self.config = config
        self.definitions = Path(config.definitions_path).read_text()
        self.examples = Path(config.examples_path).read_text() if config.examples_path else ""

    def judge_trace(self, trace_id: str, trace_text: str) -> JudgeResponse:
        examples = self.examples if self.config.shots > 0 else ""
        prompt = build_judge_prompt(trace_text, self.definitions, examples)
        response = self._dispatch(prompt, trace_id)
        response.annotations = parse_14_modes(response.raw_text)
        return response

    def _dispatch(self, prompt, trace_id) -> JudgeResponse:
        if self.config.backend == "ollama":
            return _call_ollama(self.config.model, prompt, self.config.temperature, trace_id, self.config.ollama_host, self.config.system_prompt)
        if self.config.backend == "anthropic":
            return _call_anthropic_vertex(self.config.model, prompt, self.config.temperature, trace_id,
                    self.config.genai_project, self.config.genai_location, self.config.system_prompt, self.config.reasoning)
        if self.config.backend == "genai":
            return _call_genai(self.config.model, prompt, self.config.temperature, trace_id,
                    self.config.genai_project, self.config.genai_location, self.config.system_prompt, self.config.reasoning)
        raise ValueError(f"Unknown backend: {self.config.backend!r}. Use 'genai', 'anthropic', or 'ollama'.")