"""
Uniform LLM interface.

All providers are called through a single `judge()` function that returns a
`JudgeResponse` with token counts, latency, cost, and optionally a parsed
Pydantic object when a schema is supplied.
"""

import time
import datetime
import os
import re
import yaml
from dataclasses import dataclass, field
from google import genai
from google.genai import types as genai_types
import google.auth.impersonated_credentials
import google.auth.transport.requests
from google.cloud import secretmanager
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
}

FAILURE_MODES = ["1.1","1.2","1.3","1.4","1.5","2.1","2.2","2.3","2.4","2.5","2.6","3.1","3.2","3.3"]

@dataclass
class JudgeConfig:
    model: str
    backend: str = "genai"   # "genai" | "ollama"
    temperature: float = 0.0
    reasoning: bool = False
    system_prompt: str = ""
    definitions_path: str = "taxonomy_definitions_examples/definitions.txt"
    examples_path: str  = "taxonomy_definitions_examples/examples.txt"
    dataset_path: str   = ""
    genai_project: str  = "ingka-map-services-dev"
    genai_location: str = "europe-west1"
    ollama_host: str    = "http://localhost:11434"

def load_config(path: str) -> JudgeConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return JudgeConfig(**data)


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
    "3.2 No or Incorrect Verification: <yes or no>"
    "3.3 Weak Verification: <yes or no>"
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
        if self.creds is not None:
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
            creds, pid = google.auth.default(scopes=target_scopes)
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


class LLMJudge:
      def __init__(self, config: JudgeConfig):
          self.config = config
          self.definitions = open(config.definitions_path).read()
          self.examples = open(config.examples_path).read() if config.examples_path else ""

      def judge_trace(self, trace_id: str, trace_text: str) -> JudgeResponse:
          prompt = build_judge_prompt(trace_text, self.definitions, self.examples)
          response = self._dispatch(prompt, trace_id)
          response.annotations = parse_14_modes(response.raw_text)
          return response

      def _dispatch(self, prompt, trace_id) -> JudgeResponse:
          if self.config.backend == "ollama":                                                                                              
            return _call_ollama(self.config.model, prompt, self.config.temperature, trace_id, self.config.ollama_host, self.config.system_prompt)                   
          else:                                                                                                                            
            return _call_genai(self.config.model, prompt, self.config.temperature, trace_id,
                     self.config.genai_project, self.config.genai_location, self.config.system_prompt, self.config.reasoning)