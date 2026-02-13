import hashlib
import logging
import os
import pathlib

from typing import List, Union

from dotenv import load_dotenv
from joblib import Memory # type: ignore
import requests
import concurrent.futures

from ollama import Client # type: ignore
from openai import OpenAI # type: ignore

# Gemini imports with graceful fallback
try:
    from google import genai
    from google.genai import types as genai_types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

from colorama import Fore, Style

from mind.utils.utils import init_logger, load_yaml_config_file, get_optimization_settings

# Determine project root (3 levels up from this file: src/mind/prompter -> src/mind -> src -> root)
PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
memory = Memory(location=CACHE_DIR, verbose=0)

def hash_input(*args):
    return hashlib.md5(str(args).encode()).hexdigest()

class Prompter:
    # Class-level client instances (initialized on first use)
    ollama_client = None
    gemini_client = None
    def __init__(
        self,
        model_type: str,
        llm_server: str = None,
        logger: logging.Logger = None,
        config_path: pathlib.Path = pathlib.Path("config/config.yaml"),
        temperature: float = None,
        seed: int = None,
        max_tokens: int = None,
        openai_key: str = None,
    ):
        self._logger = logger if logger else init_logger(config_path, __name__)
        self.config = load_yaml_config_file(config_path, "llm", logger)
        
        # OPT-010: Load optimization settings for batched LLM calls
        self._opt_settings = get_optimization_settings(str(config_path), self._logger)
        self._batched_llm_calls = self._opt_settings.get("batched_llm_calls", False)
        if self._batched_llm_calls:
            self._logger.info("OPT-010: Batched LLM calls enabled")

        self.GPT_MODELS = self.config.get(
            "gpt", {}).get("available_models", {})
        self.OLLAMA_MODELS = self.config.get(
            "ollama", {}).get("available_models", {})
        self.VLLM_MODELS = self.config.get(
            "vllm", {}).get("available_models", {})
        self.GEMINI_MODELS = self.config.get(
            "gemini", {}).get("available_models", [])
        # Convert to set for consistent lookup
        if isinstance(self.GEMINI_MODELS, list):
            self.GEMINI_MODELS = set(self.GEMINI_MODELS)

        self.model_type = model_type
        self.context = None
        self.params = self.config.get("parameters", {})
        
        # We can override the temperature and seed from the config file if given as arguments
        if temperature is not None:
            self.params["temperature"] = temperature
            self._logger.info(f"Setting temperature to: {temperature}")
        if seed is not None:
            self.params["seed"] = seed
            self._logger.info(f"Setting seed to: {seed}")
        if max_tokens is not None:
            # set max_tokens only if provided by the user; otherwise the default values are used
            
            # for gpt models, the parameter is 'max_completion_tokens'
            if model_type in self.GPT_MODELS:
                self.params["max_completion_tokens"] = max_tokens
                self._logger.info(f"Setting max_completion_tokens to: {max_tokens}")
            # for ollama models, the parameter is 'num_predict'
            # https://github.com/ollama/ollama/blob/main/docs/modelfile.md
            elif model_type in self.OLLAMA_MODELS:
                self.params["num_predict"] = max_tokens
                self._logger.info(f"Setting num_predict to: {max_tokens}")
            elif model_type in self.GEMINI_MODELS:
                self.params["max_output_tokens"] = max_tokens
                self._logger.info(f"Setting max_output_tokens to: {max_tokens}")
            else:
                raise ValueError("Unsupported model_type specified.")

        if model_type in self.GPT_MODELS:
            load_dotenv(self.config.get("gpt", {}).get("path_api_key", ".env"))
            self.backend = "openai"
            self._logger.info(f"Using OpenAI API with model: {model_type}")
            
            if openai_key is not None:
                os.environ["OPENAI_API_KEY"] = openai_key
                self._logger.info(f"Setting OpenAI API key from argument.")
            else:
                openai_key = os.getenv("OPENAI_API_KEY")
                if openai_key is None:
                    raise ValueError("OpenAI API key not found. Please set it in the .env file or pass it as an argument.")
            
        elif model_type in self.OLLAMA_MODELS:
            ollama_host = llm_server or self.config.get("ollama", {}).get(
                "host", "http://kumo01.tsc.uc3m.es:11434"
            )
            self._logger.info(f"Using ollama host: {ollama_host}")
            os.environ['OLLAMA_HOST'] = ollama_host
            self.backend = "ollama"
            # Initialize as class-level variable to be able to use it in the cache function
            Prompter.ollama_client = Client(
                host=ollama_host,
                headers={'x-some-header': 'some-value'}
            )
            self._logger.info(
                f"Using OLLAMA API with host: {ollama_host}"
            )
        elif model_type in self.VLLM_MODELS:
            vllm_host = llm_server or self.config.get("vllm", {}).get(
                "host", "http://localhost:6000/v1"
            )
            os.environ['VLLM_HOST'] = vllm_host
            self.backend = "vllm"
            self._logger.info(
                f"Using VLLM API with host: {vllm_host}"
            )
        elif model_type == "llama_cpp":
            self.llama_cpp_host = llm_server or self.config.get("llama_cpp", {}).get(
                "host", "http://kumo01:11435/v1/chat/completions"
            )
            self.backend = "llama_cpp"
            self._logger.info(
                f"Using llama_cpp API with host: {self.llama_cpp_host}"
            )
        elif model_type in self.GEMINI_MODELS:
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "google-genai package is required for Gemini models. "
                    "Install with: pip install google-genai"
                )
            
            load_dotenv(self.config.get("gemini", {}).get("path_api_key", ".env"))
            gemini_api_key = os.getenv("GOOGLE_API_KEY")
            
            if gemini_api_key is None:
                raise ValueError(
                    "GOOGLE_API_KEY not found. Please set it in the .env file."
                )
            
            # Check for Vertex AI configuration
            vertex_project = self.config.get("gemini", {}).get("vertex_project")
            vertex_location = self.config.get("gemini", {}).get("vertex_location")
            
            if vertex_project and vertex_location:
                # Vertex AI endpoint
                Prompter.gemini_client = genai.Client(
                    vertexai=True,
                    project=vertex_project,
                    location=vertex_location,
                )
                self._logger.info(
                    f"Using Vertex AI with project: {vertex_project}, location: {vertex_location}"
                )
            else:
                # Google AI Studio endpoint (default)
                Prompter.gemini_client = genai.Client(api_key=gemini_api_key)
                self._logger.info(f"Using Google AI Studio with model: {model_type}")
            
            self.backend = "gemini"
        else:
            raise ValueError("Unsupported model_type specified.")

    @staticmethod
    @memory.cache
    def _cached_prompt_impl(
        template: str,
        question: str,
        model_type: str,
        backend: str,
        params: tuple,
        context=None,
        use_context: bool = False,
    ) -> dict:
        """Caching setup."""

        #print("Cache miss: computing results...")
        
        if backend == "openai" or backend == "vllm":
            result, logprobs = Prompter._call_openai_api_vllm(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),
                backend=backend
            )
        elif backend == "ollama":
            result, logprobs, context = Prompter._call_ollama_api(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),
                context=context,
            )
        elif backend == "llama_cpp":
            result, logprobs = Prompter._call_llama_cpp_api(
                template=template,
                question=question,
                params=dict(params),
            )
        elif backend == "gemini":
            result, logprobs = Prompter._call_gemini_api(
                template=template,
                question=question,
                model_type=model_type,
                params=dict(params),
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

        return {
            "inputs": {
                "template": template,
                "question": question,
                "model_type": model_type,
                "backend": backend,
                "params": dict(params),
                "context": context if use_context else None,
                "use_context": use_context,
            },
            "outputs": {
                "result": result,
                "logprobs": logprobs,
            },
        }

    @staticmethod
    def _call_openai_api_vllm(template, question, model_type, params, backend):
        """Handles the OpenAI API call."""

        if template is not None:
            messages = [
                {"role": "system", "content": template},
                {"role": "user", "content": question},
            ]
        else:
            messages = [
                {"role": "user", "content": question},
            ]

        if backend == "openai":
            open_ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif backend == "vllm":
            open_ai_client = OpenAI(
                base_url=os.getenv("VLLM_HOST"),
                api_key="THIS_IS_AN_UNUSED_REQUIRED_PLACEHOLDER",
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        response = open_ai_client.chat.completions.create(
            model=model_type,
            messages=messages,
            stream=False,
            temperature=params["temperature"],
            max_tokens=params.get("max_tokens", 1000),
            seed=params.get("seed", 1234),
            logprobs=True,
            top_logprobs=20, # this is the maximum value
        )
        result = response.choices[0].message.content
        logprobs = response.choices[0].logprobs.content
        return result, logprobs

    @staticmethod
    def _call_ollama_api(template, question, model_type, params, context):
        """Handles the OLLAMA API call."""

        if Prompter.ollama_client is None:
            raise ValueError("OLLAMA client is not initialized. Check the model type configuration.")

        if template is not None:
            response = Prompter.ollama_client.generate(
                system=template,
                prompt=question,
                model=model_type,
                stream=False,
                options=params,
                context=context,
            )
        else:
            response = Prompter.ollama_client.generate(
                prompt=question,
                model=model_type,
                stream=False,
                options=params,
                context=context,
            )
        result = response["response"]
        logprobs = None
        context = response.get("context", None)
        return result, logprobs, context

    @staticmethod
    def _call_llama_cpp_api(template, question, params, llama_cpp_host="http://kumo01:11435/v1/chat/completions"):
        """Handles the llama_cpp API call."""
        payload = {
            "messages": [
                {"role": "system", "content": template},
                {"role": "user", "content": question},
            ],
            "temperature": params.get("temperature", 0),
            "max_tokens": params.get("max_tokens", 100),
            "logprobs": 1,
            "n_probs": 1,
        }
        response = requests.post(llama_cpp_host, json=payload)
        response_data = response.json()

        if response.status_code == 200:
            result = response_data["choices"][0]["message"]["content"]
            logprobs = response_data.get("completion_probabilities", [])
        else:
            raise RuntimeError(f"llama_cpp API error: {response_data.get('error', 'Unknown error')}")

        return result, logprobs

    @staticmethod
    def _call_gemini_api(template, question, model_type, params):
        """Handles the Google Gemini API call.
        
        Parameters
        ----------
        template : str or None
            The system prompt template.
        question : str
            The user question/content.
        model_type : str
            The Gemini model name (e.g., 'gemini-2.0-flash').
        params : dict
            Generation parameters.
            
        Returns
        -------
        tuple
            (result_text, logprobs) where logprobs is always None for Gemini.
        """
        if Prompter.gemini_client is None:
            raise ValueError(
                "Gemini client is not initialized. Check the model type configuration."
            )
        
        try:
            # Build generation config
            gen_config = genai_types.GenerateContentConfig(
                temperature=params.get("temperature", 0),
                top_p=params.get("top_p", 0.1),
                max_output_tokens=params.get("max_output_tokens", 1000),
            )
            
            # Add system instruction if template is provided
            if template is not None:
                gen_config.system_instruction = template
            
            # Make API call
            response = Prompter.gemini_client.models.generate_content(
                model=model_type,
                contents=question,
                config=gen_config,
            )
            
            result = response.text
            logprobs = None  # Gemini does not provide logprobs in the same format
            
            return result, logprobs
            
        except Exception as e:
            # Handle Gemini-specific errors
            error_msg = str(e)
            
            if "RESOURCE_EXHAUSTED" in error_msg or "429" in error_msg:
                raise RuntimeError(
                    f"Gemini rate limit exceeded. Consider reducing batch size. Error: {e}"
                )
            elif "PERMISSION_DENIED" in error_msg or "403" in error_msg:
                raise RuntimeError(
                    f"Gemini API key invalid or lacks permission. Check GOOGLE_API_KEY. Error: {e}"
                )
            elif "INVALID_ARGUMENT" in error_msg or "400" in error_msg:
                raise RuntimeError(
                    f"Invalid request to Gemini API. Check model name and parameters. Error: {e}"
                )
            else:
                raise RuntimeError(f"Gemini API error: {e}")

    def prompt(
        self,
        question: str,
        system_prompt_template_path: str = None,
        use_context: bool = False,
        temperature: float = None,
        dry_run: bool = False,
    ) -> Union[str, List[str]]:
        """Public method to execute a prompt given a system prompt template and a question."""

        if dry_run:
            return "Dry run mode is ON — no LLM calls will be made.", None
        
        # Load the system prompt template
        system_prompt_template = None
        if system_prompt_template_path is not None:
            with open(system_prompt_template_path, "r") as file:
                system_prompt_template = file.read()

        # Ensure hashable params for caching and get cached data / execute prompt
        if temperature is not None:
            self.params["temperature"] = temperature
        params_tuple = tuple(sorted(self.params.items()))
        
        #print("Cache key:", hash_input(system_prompt_template, question, self.model_type, self.backend, params_tuple, self.context, use_context))
        cached_data = self._cached_prompt_impl(
            template=system_prompt_template,
            question=question,
            model_type=self.model_type,
            backend=self.backend,
            params=params_tuple,
            context=self.context if use_context else None,
            use_context=use_context,
        )

        result = cached_data["outputs"]["result"]
        logprobs = cached_data["outputs"]["logprobs"]

        # Update context if necessary
        if use_context:
            self.context = cached_data["inputs"]["context"]
            
        if "<think>" in result:
            # print in green that "thinking" model is used
            print(f"{Fore.GREEN}<think> in reponse:{Style.RESET_ALL} {result}")
            result = result.split("</think>")[-1].strip()
            print(f"{Fore.RED}this is what was kept:{Style.RESET_ALL} {result}")

        return result, logprobs
    
    # =========================================================================
    # OPT-010: Batched LLM Calls
    # =========================================================================
    
    def prompt_batch(
        self,
        questions: List[str],
        system_prompt_template_path: str = None,
        temperature: float = None,
        max_workers: int = 4,
        dry_run: bool = False,
    ) -> List[tuple]:
        """
        OPT-010: Execute multiple prompts concurrently for reduced latency.
        
        Parameters
        ----------
        questions : List[str]
            List of questions to prompt.
        system_prompt_template_path : str, optional
            Path to system prompt template file.
        temperature : float, optional
            Override temperature for all prompts.
        max_workers : int, optional
            Maximum concurrent API calls. Default is 4.
        dry_run : bool, optional
            If True, skip actual API calls.
            
        Returns
        -------
        List[tuple]
            List of (result, logprobs) tuples for each question.
        """
        if dry_run:
            return [("Dry run mode is ON — no LLM calls will be made.", None) for _ in questions]
        
        if not questions:
            return []
        
        self._logger.info(f"OPT-010: Batching {len(questions)} prompts with {max_workers} workers")
        
        def process_single(question: str):
            """Process a single prompt."""
            return self.prompt(
                question=question,
                system_prompt_template_path=system_prompt_template_path,
                temperature=temperature,
                dry_run=False,
            )
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all prompts
            future_to_question = {
                executor.submit(process_single, q): q for q in questions
            }
            
            # Collect results in order
            for future in concurrent.futures.as_completed(future_to_question):
                question = future_to_question[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self._logger.error(f"Prompt failed for question: {question[:50]}... Error: {e}")
                    results.append((None, None))
        
        self._logger.info(f"OPT-010: Completed {len(results)} prompts")
        return results
    
    def prompts_auto(
        self,
        questions: List[str],
        system_prompt_template_path: str = None,
        temperature: float = None,
        max_workers: int = 4,
        dry_run: bool = False,
    ) -> List[tuple]:
        """
        OPT-010: Automatically choose between batched or sequential execution based on config.
        
        This is a convenience wrapper that checks the `batched_llm_calls` config flag
        and routes to either `prompt_batch()` or sequential `prompt()` calls.
        
        Parameters
        ----------
        questions : List[str]
            List of questions to prompt.
        system_prompt_template_path : str, optional
            Path to system prompt template file.
        temperature : float, optional
            Override temperature for all prompts.
        max_workers : int, optional
            Maximum concurrent API calls (only used if batching enabled).
        dry_run : bool, optional
            If True, skip actual API calls.
            
        Returns
        -------
        List[tuple]
            List of (result, logprobs) tuples for each question.
        """
        if self._batched_llm_calls:
            return self.prompt_batch(
                questions=questions,
                system_prompt_template_path=system_prompt_template_path,
                temperature=temperature,
                max_workers=max_workers,
                dry_run=dry_run,
            )
        else:
            # Sequential execution
            results = []
            for question in questions:
                result = self.prompt(
                    question=question,
                    system_prompt_template_path=system_prompt_template_path,
                    temperature=temperature,
                    dry_run=dry_run,
                )
                results.append(result)
            return results