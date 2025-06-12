#!/usr/bin/env python3

import os
import json
import hashlib
import hmac
import logging
import threading
import requests
from datetime import datetime
from typing import Dict, List, Optional
import re

from flask import Flask, request, jsonify

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN')
GITHUB_URL = os.getenv('GITHUB_URL', 'https://api.github.com')
WEBHOOK_SECRET = os.getenv('WEBHOOK_SECRET')
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'

# AI Configuration - Fully Configurable
MODEL_PATH = os.getenv('MODEL_PATH', '../models/llama3/models/solar/solar-10.7b-instruct-v1.0.Q4_K_M.gguf')
MODEL_NAME = os.getenv('MODEL_NAME', 'solar-10.7b-instruct')
MODEL_TYPE = os.getenv('MODEL_TYPE', 'auto')  # auto, llama, codellama, deepseek, starcoder, qwen, solar

# Model-specific parameters
USE_GPU = os.getenv('USE_GPU', 'false').lower() == 'true'
GPU_LAYERS = int(os.getenv('GPU_LAYERS', '0'))
CONTEXT_SIZE = int(os.getenv('CONTEXT_SIZE', '2048'))
CPU_THREADS = int(os.getenv('CPU_THREADS', '4'))
MAX_TOKENS = int(os.getenv('MAX_TOKENS', '1500'))
TEMPERATURE = float(os.getenv('TEMPERATURE', '0.1'))
TOP_P = float(os.getenv('TOP_P', '0.9'))

# Chat template configuration
CHAT_TEMPLATE = os.getenv('CHAT_TEMPLATE', 'auto')  # auto, chatml, llama, alpaca, vicuna, none
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT', 'auto')  # auto or custom system prompt
USER_PROMPT_PREFIX = os.getenv('USER_PROMPT_PREFIX', '')  # Custom prefix for user prompts
STOP_TOKENS = os.getenv('STOP_TOKENS', '</s>,Human:,Assistant:').split(',')

# Analysis configuration
USE_PATTERN_ONLY = os.getenv('USE_PATTERN_ONLY', 'false').lower() == 'true'
ENABLE_REFACTORING = os.getenv('ENABLE_REFACTORING', 'true').lower() == 'true'
FALSE_POSITIVE_THRESHOLD = float(os.getenv('FALSE_POSITIVE_THRESHOLD', '0.3'))  # Max issues per line ratio

# GitHub configuration
SECURITY_FAIL_THRESHOLD = int(os.getenv('SECURITY_FAIL_THRESHOLD', '1'))
QUALITY_SCORE_THRESHOLD = int(os.getenv('QUALITY_SCORE_THRESHOLD', '70'))

class ConfigurableModelAnalyzer:
    def __init__(self):
        self.model_path = MODEL_PATH
        self.model_name = MODEL_NAME
        self.model_type = MODEL_TYPE
        self.llama = None
        self._initialize_model()

    def summarize_code_purpose(self, file_path: str, code_lines: List[str]) -> str:
        """Generate a plain-English summary of what the file does"""
        system_prompt = "You are a senior software engineer. Summarize what this code does in plain language. Focus on explaining the business logic or workflow in 2-4 sentences."

        logger.info(f"🧠 Generating summary for {file_path}")
        code_snippet = '\n'.join(code_lines[:80])
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nSummarize the purpose of this code:\n{file_path}\n\n{code_snippet}<|im_end|>\n<|im_start|>assistant"
        try:
            logger.info(f"🧠 Summarizing code: {file_path}")
            result = self.llama(
                prompt,
                max_tokens=300,
                temperature=0.5,
                top_p=0.9,
                stop=STOP_TOKENS
            )
            result_text = result['choices'][0]['text'].strip()

            # Remove system prompt echo if it's repeated at the start
            if result_text.startswith(system_prompt[:20]):  # match first few words
                result_text = result_text[len(system_prompt):].lstrip()

            return result_text
        except Exception as e:
            logger.error(f"Failed to summarize business logic for {file_path}: {e}")
            return f"Could not summarize {file_path}"
        
    def _detect_model_type(self) -> str:
        """Auto-detect model type from model name/path"""
        model_name_lower = self.model_name.lower()
        model_path_lower = self.model_path.lower()
        
        if any(name in model_name_lower or name in model_path_lower for name in ['codellama', 'code-llama']):
            return 'codellama'
        elif any(name in model_name_lower or name in model_path_lower for name in ['deepseek', 'coder']):
            return 'deepseek'
        elif any(name in model_name_lower or name in model_path_lower for name in ['starcoder', 'star-coder']):
            return 'starcoder'
        elif any(name in model_name_lower or name in model_path_lower for name in ['qwen', 'qw']):
            return 'qwen'
        elif any(name in model_name_lower or name in model_path_lower for name in ['solar']):
            return 'solar'
        elif any(name in model_name_lower or name in model_path_lower for name in ['llama', 'alpaca', 'vicuna']):
            return 'llama'
        else:
            return 'generic'
    
    def _get_chat_template(self) -> str:
        """Get appropriate chat template based on model type"""
        if CHAT_TEMPLATE != 'auto':
            return CHAT_TEMPLATE
            
        detected_type = self._detect_model_type() if self.model_type == 'auto' else self.model_type
        
        template_map = {
            'codellama': 'llama',
            'deepseek': 'deepseek', 
            'starcoder': 'none',
            'qwen': 'chatml',
            'solar': 'chatml',
            'llama': 'chatml',
            'generic': 'none'
        }
        
        return template_map.get(detected_type, 'none')
    
    def _get_system_prompt(self) -> str:
        """Get appropriate system prompt based on model type"""
        if SYSTEM_PROMPT != 'auto':
            return SYSTEM_PROMPT
            
        detected_type = self._detect_model_type() if self.model_type == 'auto' else self.model_type
        
        if detected_type in ['codellama', 'deepseek', 'starcoder']:
            return "You are an expert code security reviewer specializing in vulnerability detection. Analyze code carefully for REAL security issues, bugs, and quality problems. Avoid false positives by considering full context."
        elif detected_type in ['qwen', 'solar']:
            return "You are an expert code security reviewer. Analyze code for security vulnerabilities, bugs, and quality issues. Be precise and avoid false positives. Consider the full context of the code."
        else:
            return "You are a helpful assistant that analyzes code for security issues and bugs."
    
    def _initialize_model(self):
        """Initialize the GGUF model with configurable parameters"""
        try:
            # Try to import llama-cpp-python
            try:
                from llama_cpp import Llama
            except ImportError:
                logger.error("❌ llama-cpp-python not installed. Run: pip install llama-cpp-python")
                return
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                logger.error(f"❌ Model file not found: {self.model_path}")
                return
            
            logger.info(f"🔄 Loading GGUF model: {self.model_name}")
            logger.info(f"📁 Path: {self.model_path}")
            logger.info(f"🤖 Type: {self._detect_model_type() if self.model_type == 'auto' else self.model_type}")
            
            # GPU configuration
            n_gpu_layers = 0
            if USE_GPU:
                try:
                    # Check if CUDA is available
                    import torch
                    if torch.cuda.is_available():
                        n_gpu_layers = GPU_LAYERS
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
                        logger.info(f"🚀 GPU layers: {n_gpu_layers}")
                    else:
                        logger.warning("⚠️ GPU requested but CUDA not available")
                except ImportError:
                    logger.warning("⚠️ PyTorch not found, cannot detect GPU")
                except Exception as e:
                    logger.warning(f"⚠️ GPU detection failed: {e}")
            else:
                logger.info("🖥️ CPU-only mode")
            
            # Initialize the model with configurable parameters
            self.llama = Llama(
                model_path=self.model_path,
                n_ctx=CONTEXT_SIZE,
                n_threads=CPU_THREADS,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                use_mlock=True,
                use_mmap=True,
                low_vram=False,
                f16_kv=True,
                chat_format="chatml"
            )
            
            logger.info(f"✅ Model loaded: {self.model_name}")
            logger.info(f"⚙️ Context: {CONTEXT_SIZE}, Threads: {CPU_THREADS}, GPU Layers: {n_gpu_layers}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.llama = None
    
    def analyze_code_with_llama(self, file_path: str, code_lines: List[str], patch: str = '') -> Dict:
        """Analyze code using direct GGUF model"""
        # Option to skip AI completely and use only pattern-based analysis
        if USE_PATTERN_ONLY:
            logger.info(f"🔄 Using pattern-only analysis for {file_path}")
            return self._fallback_analysis(file_path, code_lines)
            
        if not self.llama:
            logger.warning("⚠️ GGUF model not available, using fallback analysis")
            return self._fallback_analysis(file_path, code_lines)
        
        try:
            code_content = '\n'.join(code_lines)
            logger.info(f"🤖 Analyzing {file_path} with {self.model_name} ({len(code_lines)} lines)")
            
            # Try AI analysis with timeout and fallback
            try:
                prompt = self._create_analysis_prompt(file_path, code_content)
                
                # Generate response with configurable parameters
                response = self.llama(
                    prompt,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    stop=STOP_TOKENS
                )
                
                analysis_text = response['choices'][0]['text']
                
                # Log the raw response for debugging
                logger.info(f"🔍 Raw AI response length: {len(analysis_text)} chars")
                logger.info(f"🔍 Raw AI response preview: {analysis_text[:100]}...")
                
                logger.info("✅ AI model response received")
                
                # Parse and validate the response
                parsed_result = self._parse_llama_response(analysis_text, file_path, code_lines)
                
                # If AI gives too many false positives, fall back to pattern-based
                total_issues = (len(parsed_result.get('security_issues', [])) + 
                              len(parsed_result.get('bugs', [])) + 
                              len(parsed_result.get('code_quality', [])))
                
                # If AI finds too many issues (likely false positives), use pattern-based instead
                # if total_issues > len(code_lines) * FALSE_POSITIVE_THRESHOLD:
                #     logger.warning(f"⚠️ AI found {total_issues} issues in {len(code_lines)} lines, using pattern-based analysis instead")
                #     return self._fallback_analysis(file_path, code_lines)
                logger.info(f"🧪 Keeping AI output even with {total_issues} issues in {len(code_lines)} lines")

                
                return parsed_result
                
            except Exception as ai_error:
                logger.error(f"❌ AI analysis failed: {ai_error}")
                logger.info("🔄 Falling back to pattern-based analysis")
                return self._fallback_analysis(file_path, code_lines)
                
        except Exception as e:
            logger.error(f"❌ Error in analyze_code_with_llama: {e}")
            return self._fallback_analysis(file_path, code_lines)
    
    def _create_analysis_prompt(self, file_path: str, code_content: str, chat_template: str = 'none', 
                               system_prompt: str = None, USER_PROMPT_PREFIX: str = "") -> str:
        file_ext = os.path.splitext(file_path)[-1].lower()

        # Language hints
        language_hint = {
            ".gradle": "Gradle build script (Groovy)",
            ".java": "Java source file",
            ".xml": "XML configuration file",
            ".yml": "YAML configuration file",
            ".yaml": "YAML configuration file",
            ".json": "JSON configuration file",
            ".py": "Python source file",
            ".js": "JavaScript source file",
            ".ts": "TypeScript source file"
        }.get(file_ext, "Source code")

        system_hint = ""
        if file_ext == ".java":
            system_hint = "\nUse Java coding conventions. Follow OWASP, avoid bad exception handling, threading issues, hardcoded values."
        elif file_ext == ".gradle":
            system_hint = "\nCheck for insecure credential usage, deprecated Gradle syntax, secret exposure, bad plugin or dependency declarations."
        elif file_ext == ".py":
            system_hint = "\nUse Python coding conventions. Follow PEP8, OWASP. Avoid unused imports, overly complex functions, hardcoded secrets, or missing exception handling."

        final_system_prompt = (
    "You are a meticulous static code analyzer with expertise in detecting security issues, bugs, code quality problems, and anti-patterns in Java code. "
    "You do NOT avoid reporting multiple issues. Your job is to flag everything that could be improved or corrected, even small things. "
    "Be precise, detailed, and exhaustive."
    + system_hint
)

        lines = code_content.split('\n')
        if len(lines) > 400:
            lines = lines[:400]
        numbered_lines = '\n'.join([f"{i+1:3d}: {line}" for i, line in enumerate(lines)])

        user_prompt = f"""
You are reviewing a {language_hint}. Please analyze the code line-by-line and identify the following:

1. 🔐 **Credential & PII Detection**  
   - Hardcoded credentials: API keys, tokens, passwords, secret strings  
   - Personally Identifiable Information (PII): email addresses, phone numbers, names, addresses, SSNs  
   - Flag use of any PII without masking, encryption, or anonymization

2. 🛡️ **Security Validation**  
   - Is PII being stored or logged without protection?  
   - Are encryption libraries used? Flag unsafe handling of user-sensitive data  
   - SQL injection risks, insecure file access, unsafe eval/exec patterns

3. 🧠 **Bugs & Code Smells**  
   - Logic bugs (e.g., `if (a = b)`), unreachable code  
   - Misused operators, shadowed variables, infinite loops, etc.

4. 🧹 **Code Quality**  
   - Use of debug prints, deeply nested blocks, unhandled exceptions  
   - Violations of standard naming, spacing, or modularity  
   - TODO comments or commented-out legacy code

5. 🚀 **Optimization Suggestions**  
   - Recommend cleaner or faster approaches (e.g., better data structures, short-circuiting)  
   - Suggest splitting large methods or removing redundant code

6. 📄 **(Optional) Refactor Hints**  
   - Point out code that could be extracted into reusable functions  
   - Recommend the use of constants, enums, or config values

{USER_PROMPT_PREFIX}

{numbered_lines}

⚠️ **Instructions for Response Format:**  
- Output **strictly valid JSON** with this format (no extra text, no markdown):

{{
  "security_issues": [],
  "bugs": [],
  "code_quality": [],
  "refactoring_suggestions": [],
  "optimizations": [],
  "overall_score": 0,
  "confidence": 0.0
}}

⚠️ Important:
- DO NOT include markdown code fences or explanation paragraphs.
- DO NOT flag `os.getenv`, `os.environ.get`, or secure config lookups.
- Highlight any variable or string that *looks* like PII or a credential.
- Use `"line_number"`, `"line_content"`, and `"suggestion"` in each issue for precision.
- Be exhaustive. Report **all valid issues**, even if many.
"""

        if chat_template == 'chatml':
            return f"<|im_start|>system\n{final_system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant"
        elif chat_template == 'deepseek':
            return f"You are an AI programming assistant.\n{final_system_prompt}\n\n### Instruction:\n{user_prompt}\n\n### Response:"
        elif chat_template == 'llama-3':
            return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n{final_system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        else:
            return f"{final_system_prompt}\n\n{user_prompt}"
    
    def _parse_llama_response(self, response_text: str, file_path: str, code_lines: List[str]) -> Dict:
        """Parse Llama's JSON response with fallback and cleanup"""
        try:
            cleaned = response_text.strip()
            logger.error(f"🔍 Full raw model response:\n{cleaned[:500]}")

            # Remove common markdown wrappers and explanation text
            cleaned = re.sub(r'^```(?:json)?', '', cleaned, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r'```$', '', cleaned).strip()
            cleaned = re.sub(r'(?i)^Here.*?:', '', cleaned).strip()
            cleaned = re.sub(r'(?i)^response\s*:', '', cleaned).strip()

            # Try to extract the first JSON object block
            json_match = re.search(r'\{.*\}', cleaned, flags=re.DOTALL)
            if not json_match:
                raise ValueError("No JSON braces found")

            json_block = json_match.group(0)

            analysis = json.loads(json_block)

            for issue_type in ['security_issues', 'bugs', 'code_quality']:
                for issue in analysis.get(issue_type, []):
                    line_num = issue.get('line_number', 1)
                    if line_num > len(code_lines):
                        issue['line_number'] = len(code_lines)
                    elif line_num < 1:
                        issue['line_number'] = 1

            return {
                'security_issues': analysis.get('security_issues', []),
                'bugs': analysis.get('bugs', []),
                'code_quality': analysis.get('code_quality', []),
                'refactoring_suggestions': analysis.get('refactoring_suggestions', []),
                'optimizations': analysis.get('optimizations', []),
                'overall_score': analysis.get('overall_score', 80),
                'confidence': analysis.get('confidence', 0.8),
                'file_path': file_path,
                'analysis_source': 'direct_gguf'
            }

        except Exception as e:
            logger.error(f"❌ JSON parsing error: {e}")
        return self._fallback_analysis(file_path, code_lines)

    
    def _fallback_analysis(self, file_path: str, code_lines: List[str]) -> Dict:
        """Fallback pattern-based analysis"""
        logger.info("🔄 Using fallback pattern-based analysis")
        
        issues = {
            'security_issues': [],
            'bugs': [],
            'code_quality': []
        }
        
        score = 85
        
        # Analyze each line
        for line_num, line in enumerate(code_lines, 1):
            line_lower = line.lower().strip()
            
            # Security patterns
            if any(pattern in line_lower for pattern in ['api_key', 'password', 'secret', 'token']):
                if '=' in line and not line_lower.startswith('#'):
                    issues['security_issues'].append({
                        'severity': 'high',
                        'type': 'hardcoded_credential',
                        'line_number': line_num,
                        'line_content': line.strip(),
                        'description': f'Hardcoded credential detected on line {line_num}',
                        'suggestion': 'Use environment variables',
                        'confidence': 0.9
                    })
                    score -= 25
            
            # SQL injection
            if 'select' in line_lower and '${' in line:
                issues['security_issues'].append({
                    'severity': 'critical',
                    'type': 'sql_injection',
                    'line_number': line_num,
                    'line_content': line.strip(),
                    'description': f'SQL injection vulnerability on line {line_num}',
                    'suggestion': 'Use parameterized queries',
                    'confidence': 0.8
                })
                score -= 30
            
            # Logic bugs
            if ' = ' in line and ('if' in line_lower or 'while' in line_lower):
                if not ('==' in line or '===' in line):
                    issues['bugs'].append({
                        'severity': 'high',
                        'type': 'logic_error',
                        'line_number': line_num,
                        'line_content': line.strip(),
                        'description': f'Assignment instead of comparison on line {line_num}',
                        'suggestion': 'Use === for comparison',
                        'confidence': 0.9
                    })
                    score -= 15
            
            # Debug statements
            if any(debug in line_lower for debug in ['console.log', 'print(', 'debugger']):
                if not line_lower.strip().startswith('#'):
                    issues['code_quality'].append({
                        'severity': 'medium',
                        'type': 'debug_statement',
                        'line_number': line_num,
                        'line_content': line.strip(),
                        'description': f'Debug statement on line {line_num}',
                        'suggestion': 'Remove before production',
                        'confidence': 0.95
                    })
                    score -= 5
            
            # TODO comments
            if 'todo' in line_lower:
                issues['code_quality'].append({
                    'severity': 'low',
                    'type': 'todo_comment',
                    'line_number': line_num,
                    'line_content': line.strip(),
                    'description': f'TODO comment on line {line_num}',
                    'suggestion': 'Complete or create ticket',
                    'confidence': 0.9
                })
                score -= 2
        
        return {
            'security_issues': issues['security_issues'],
            'bugs': issues['bugs'],
            'code_quality': issues['code_quality'],
            'refactoring_suggestions': [],  # Added back (empty for fallback)
            'optimizations': [],
            'overall_score': max(0, score),
            'confidence': 0.8,
            'file_path': file_path,
            'analysis_source': 'pattern_based'
        }

class GitHubClient:
    def __init__(self, token: str, base_url: str = 'https://api.github.com'):
        self.token = token
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json',
            'Content-Type': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict]:
        """Get pull request file changes"""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/files"
            logger.info(f"🔄 Fetching PR files from: {url}")
            
            response = self.session.get(url, timeout=15)
            
            if response.status_code == 404:
                logger.error(f"❌ PR files not found for {pr_number}")
                return []
            
            response.raise_for_status()
            files = response.json()
            logger.info(f"✅ Fetched {len(files)} changed files")
            return files
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to get PR files: {e}")
            return []
    
    def create_pr_comment(self, owner: str, repo: str, pr_number: int, body: str) -> bool:
        """Create a general comment on pull request"""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/issues/{pr_number}/comments"
            data = {'body': body}
            
            logger.info(f"💬 Posting general comment to PR {pr_number}")
            
            response = self.session.post(url, json=data, timeout=15)
            response.raise_for_status()
            
            logger.info("✅ Successfully posted PR comment")
            return True
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to post PR comment: {e}")
            return False
    
    def create_pr_review_comment(self, owner: str, repo: str, pr_number: int, 
                               commit_sha: str, file_path: str, line: int, body: str) -> bool:
        """Create an inline review comment on a specific line"""
        try:
            url = f"{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/comments"
            data = {
                'body': body,
                'commit_id': commit_sha,
                'path': file_path,
                'line': line,
                'side': 'RIGHT'
            }
            
            logger.info(f"📝 Posting inline comment on {file_path}:{line}")
            
            response = self.session.post(url, json=data, timeout=15)
            response.raise_for_status()
            
            logger.info("✅ Successfully posted inline comment")
            return True
            
        except requests.RequestException as e:
            logger.error(f"❌ Failed to post inline comment: {e}")
            return False

def verify_github_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature"""
    if not WEBHOOK_SECRET:
        logger.warning("No webhook secret configured - skipping verification")
        return True
    
    try:
        expected_signature = hmac.new(
            WEBHOOK_SECRET.encode('utf-8'),
            payload,
            hashlib.sha1
        ).hexdigest()
        
        if signature.startswith('sha1='):
            signature = signature[5:]
        
        return hmac.compare_digest(signature, expected_signature)
        
    except Exception as e:
        logger.error(f"Error verifying signature: {e}")
        return False

def extract_code_changes_from_webhook(payload: Dict, action: str) -> List[Dict]:
    """Extract code changes from GitHub webhook based on action"""
    pr_data = payload.get('pull_request', {})
    repository = payload.get('repository', {})

    if not pr_data or not repository or not github_client:
        logger.warning("Missing PR or repository data")
        return []

    owner = repository.get('owner', {}).get('login')
    repo_name = repository.get('name')
    pr_number = pr_data.get('number')

    if action == 'opened' or action == 'reopened':
        logger.info("📦 PR opened – scanning all commits")
        return extract_all_commits_from_pr(owner, repo_name, pr_number)
    elif action == 'synchronize':
        logger.info("🔁 PR synchronized – scanning only latest commit")
        head_sha = pr_data.get('head', {}).get('sha')
        return extract_single_commit(owner, repo_name, head_sha)
    else:
        logger.info(f"⏭️ Action '{action}' not handled for commit extraction")
        return []

def extract_all_commits_from_pr(owner: str, repo: str, pr_number: int) -> List[Dict]:
    """Get all commits in PR and extract changes"""
    commits_url = f"{github_client.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/commits"
    try:
        resp = github_client.session.get(commits_url, timeout=10)
        resp.raise_for_status()
        commits = resp.json()
    except Exception as e:
        logger.error(f"❌ Failed to fetch commits in PR #{pr_number}: {e}")
        return []

    all_files = {}

    for commit in commits:
        sha = commit['sha']
        single_files = extract_single_commit(owner, repo, sha)
        for f in single_files:
            fp = f['file_path']
            if fp not in all_files:
                all_files[fp] = f
            else:
                all_files[fp]['added_lines'] += f['added_lines']
    
    return list(all_files.values())
def extract_single_commit(owner: str, repo: str, commit_sha: str) -> List[Dict]:
    """Extract file changes from a single commit"""
    url = f"{github_client.base_url}/repos/{owner}/{repo}/commits/{commit_sha}"
    try:
        resp = github_client.session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.error(f"❌ Failed to fetch commit {commit_sha}: {e}")
        return []

    changes = []
    for f in data.get('files', []):
        if f.get('status') == 'removed':
            continue
        filename = f['filename']
        if any(filename.endswith(ext) for ext in {'.md', '.txt', '.json', '.yml', '.yaml', '.xml', '.lock'}):
            continue
        patch = f.get('patch', '')
        added_lines = [line[1:] for line in patch.split('\n') if line.startswith('+') and not line.startswith('+++')]
        if added_lines:
            changes.append({
                'file_path': filename,
                'added_lines': added_lines,
                'additions': f.get('additions', 0),
                'patch': patch,
                'source': 'single_commit',
                'commit_sha': commit_sha
            })
    return changes


def map_to_patch_line(patch: str, line_content: str) -> Optional[int]:
    """Map a line of code to its line number in the GitHub patch"""
    if not patch or not line_content:
        return None
    
    # Clean the line content for better matching
    clean_content = line_content.strip()
    if not clean_content:
        return None
    
    patch_lines = patch.split('\n')
    line_number = 0
    
    for patch_line in patch_lines:
        if patch_line.startswith('@@'):
            try:
                # Parse the line number from @@ -old_start,old_count +new_start,new_count @@
                parts = patch_line.split('+')[1].split(' ')[0]
                line_number = int(parts.split(',')[0]) - 1
            except:
                line_number = 1
        elif patch_line.startswith('+'):
            line_number += 1
            patch_content = patch_line[1:].strip()  # Remove '+' and whitespace
            
            # Try exact match first
            if clean_content == patch_content:
                return line_number
            
            # Try partial matches for longer lines
            if len(clean_content) > 20:
                # Check if the core part of the line matches (first 30 chars)
                if clean_content[:30] == patch_content[:30]:
                    return line_number
                # Check if significant keywords match
                if any(keyword in patch_content for keyword in ['API_KEY', 'password', 'console.log', 'SELECT', 'if']):
                    if clean_content[:15] in patch_content or patch_content[:15] in clean_content:
                        return line_number
            else:
                # For shorter lines, try substring matching
                if clean_content in patch_content or patch_content in clean_content:
                    return line_number
                    
        elif not patch_line.startswith('-'):
            if patch_line.strip() and not patch_line.startswith('@@'):
                line_number += 1
    
    # If no exact match found, try to find the closest match by searching for key identifiers
    if any(identifier in clean_content for identifier in ['API_KEY', 'DB_PASSWORD', 'console.log', 'SELECT', 'user.email']):
        patch_lines = patch.split('\n')
        line_number = 0
        
        for patch_line in patch_lines:
            if patch_line.startswith('@@'):
                try:
                    parts = patch_line.split('+')[1].split(' ')[0]
                    line_number = int(parts.split(',')[0]) - 1
                except:
                    line_number = 1
            elif patch_line.startswith('+'):
                line_number += 1
                # Look for key identifiers
                if 'API_KEY' in clean_content and 'API_KEY' in patch_line:
                    return line_number
                elif 'DB_PASSWORD' in clean_content and 'password' in patch_line.lower():
                    return line_number
                elif 'console.log' in clean_content and 'console.log' in patch_line:
                    return line_number
                elif 'SELECT' in clean_content and 'SELECT' in patch_line:
                    return line_number
                elif 'user.email' in clean_content and 'user.email' in patch_line:
                    return line_number
            elif not patch_line.startswith('-'):
                if patch_line.strip() and not patch_line.startswith('@@'):
                    line_number += 1
    
    return None

def format_inline_comment(issue: Dict) -> str:
    """Format an issue into a simple inline comment"""
    description = issue.get('description', 'Issue detected')
    suggestion = issue.get('suggestion', '')
    
    # Simple format: just problem and fix
    comment = f"**Problem:** {description}\n"
    
    if suggestion:
        comment += f"**Fix:** {suggestion}"
    
    return comment

def get_file_language(file_path: str) -> str:
    """Get the programming language for syntax highlighting based on file extension"""
    ext = file_path.split('.')[-1].lower()
    language_map = {
        'py': 'python',
        'js': 'javascript', 
        'ts': 'typescript',
        'jsx': 'jsx',
        'tsx': 'tsx',
        'java': 'java',
        'cpp': 'cpp',
        'c': 'c',
        'cs': 'csharp',
        'php': 'php',
        'rb': 'ruby',
        'go': 'go',
        'rs': 'rust',
        'kt': 'kotlin',
        'swift': 'swift'
    }
    return language_map.get(ext, 'text')

def generate_refactored_file(file_path: str, code_lines: List[str], analysis: Dict) -> Dict:
    """Generate a completely refactored version of the file with all issues fixed"""
    try:
        # Only refactor if there are significant issues (not just low-severity quality issues)
        significant_issues = []
        significant_issues.extend([i for i in analysis.get('security_issues', []) if i.get('severity') in ['critical', 'high']])
        significant_issues.extend([i for i in analysis.get('bugs', []) if i.get('severity') in ['critical', 'high']])
        
        # Skip refactoring if no significant issues or file is too large
        if not significant_issues or len(code_lines) > 50:
            return None
        
        logger.info(f"🔧 Quick refactoring for {file_path} ({len(significant_issues)} critical issues)")
        
        # Simple rule-based refactoring for common issues
        refactored_lines = []
        for line in code_lines:
            refactored_line = line
            
            # Fix common patterns
            for issue in significant_issues:
                issue_type = issue.get('type', '')
                line_content = issue.get('line_content', '').strip()
                
                if line_content and line_content in line:
                    if issue_type == 'hardcoded_credential':
                        # Replace hardcoded values with env variables
                        if 'API_KEY' in line and '=' in line:
                            refactored_line = line.replace('"sk-', 'process.env.API_KEY || "sk-')
                            refactored_line = refactored_line.replace("'sk-", "process.env.API_KEY || 'sk-")
                        elif 'password' in line.lower() and '=' in line:
                            refactored_line = line.replace('"', 'process.env.PASSWORD || "')
                            
                    elif issue_type == 'logic_error' and ' = ' in line and 'if' in line:
                        # Fix assignment in conditions
                        refactored_line = line.replace(' = ', ' === ')
                        
                    elif issue_type == 'debug_statement':
                        # Comment out debug statements
                        if 'console.log' in line or 'print(' in line:
                            refactored_line = '// ' + line.strip() + ' // Removed for production'
            
            refactored_lines.append(refactored_line)
        
        # Only return if we actually made changes
        refactored_code = '\n'.join(refactored_lines)
        original_code = '\n'.join(code_lines)
        
        if refactored_code != original_code:
            return {
                'file_path': file_path,
                'original_lines': len(code_lines),
                'issues_fixed': len(significant_issues),
                'refactored_code': refactored_code
            }
        else:
            return None
            
    except Exception as e:
        logger.error(f"❌ Error in quick refactoring for {file_path}: {e}")
        return None

def analyze_files_with_ai(code_changes: List[Dict]) -> Dict:
    """Analyze all changed files using AI"""
    analyzer = ConfigurableModelAnalyzer()
    
    all_results = {
        'files_analyzed': [],
        'security_issues': [],
        'bugs': [],
        'code_quality': [],
        'refactoring_suggestions': [],  # Added back
        'optimizations': [],
        'overall_scores': [],
        'files_count': len(code_changes),
        'lines_count': 0,
        'inline_comments': [],
        'refactored_files': [],  # New: store complete refactored files,
        "code_summaries": []
    }
    
    for change in code_changes:
        file_path = change['file_path']
        added_lines = change.get('added_lines', [])
        patch = change.get('patch', '')
        commit_sha = change.get('commit_sha', '')
        
        all_results['lines_count'] += len(added_lines)
        
        if added_lines:
            logger.info(f"🔍 Analyzing {file_path} ({len(added_lines)} lines from commit {commit_sha[:8] if commit_sha else 'unknown'})")
            
            # Analyze with AI
            file_analysis = analyzer.analyze_code_with_llama(file_path, added_lines, patch)
            
            # Aggregate results
            all_results['files_analyzed'].append(file_analysis)
            all_results['security_issues'].extend(file_analysis.get('security_issues', []))
            all_results['bugs'].extend(file_analysis.get('bugs', []))
            all_results['code_quality'].extend(file_analysis.get('code_quality', []))
            all_results['refactoring_suggestions'].extend(file_analysis.get('refactoring_suggestions', []))  # Added back
            all_results['overall_scores'].append(file_analysis.get('overall_score', 80))
            try:
                purpose = analyzer.summarize_code_purpose(file_path, added_lines)
                all_results["code_summaries"].append(f"**{file_path}**: {purpose}")
            except Exception as e:
                logger.error(f"Failed to summarize {file_path}: {e}")
            # Generate refactored version ONLY if enabled and has significant issues
            if ENABLE_REFACTORING:
                refactored_file = generate_refactored_file(file_path, added_lines, file_analysis)
                if refactored_file:
                    all_results['refactored_files'].append(refactored_file)
            
            # Prepare inline comments (simple issues only)
            for issue_type in ['security_issues', 'bugs', 'code_quality']:
                for issue in file_analysis.get(issue_type, []):
                    if issue.get('line_number') and issue.get('line_content'):
                        # Map to actual line in patch
                        actual_line = map_to_patch_line(patch, issue['line_content'])
                        
                        logger.info(f"🔍 DEBUG: Issue on line {issue.get('line_number')}: {issue.get('description')}")
                        logger.info(f"🔍 DEBUG: Looking for content: '{issue.get('line_content')[:50]}...'")
                        logger.info(f"🔍 DEBUG: Mapped to patch line: {actual_line}")
                        
                        if actual_line:
                            inline_comment = {
                                'path': file_path,
                                'line': actual_line,
                                'body': format_inline_comment(issue)
                            }
                            all_results['inline_comments'].append(inline_comment)
                            logger.info(f"✅ DEBUG: Added inline comment for {file_path}:{actual_line}")
                        else:
                            logger.warning(f"⚠️ DEBUG: Could not map line for: {issue.get('line_content')[:50]}...")
                            # Try to add a general comment on the first added line as fallback
                            first_added_line = None
                            patch_lines = patch.split('\n')
                            line_num = 0
                            for patch_line in patch_lines:
                                if patch_line.startswith('@@'):
                                    try:
                                        parts = patch_line.split('+')[1].split(' ')[0]
                                        line_num = int(parts.split(',')[0]) - 1
                                    except:
                                        line_num = 1
                                elif patch_line.startswith('+'):
                                    line_num += 1
                                    first_added_line = line_num
                                    break
                                elif not patch_line.startswith('-'):
                                    if patch_line.strip() and not patch_line.startswith('@@'):
                                        line_num += 1
                            
                            if first_added_line:
                                logger.info(f"🔄 DEBUG: Adding comment to first added line {first_added_line}")
                                inline_comment = {
                                    'path': file_path,
                                    'line': first_added_line,
                                    'body': format_inline_comment(issue)
                                }
                                all_results['inline_comments'].append(inline_comment)
                                logger.info(f"✅ DEBUG: Added fallback inline comment for {file_path}:{first_added_line}")
    
    logger.info(f"📝 DEBUG: Total inline comments prepared: {len(all_results['inline_comments'])}")
    
    # Calculate aggregate scores
    if all_results['overall_scores']:
        all_results['average_score'] = sum(all_results['overall_scores']) / len(all_results['overall_scores'])
    else:
        all_results['average_score'] = 80
    
    # Determine if should block
    critical_issues = [issue for issue in all_results['security_issues'] 
                      if issue.get('severity') in ['critical', 'high']]
    
    all_results['should_block'] = (
        len(critical_issues) >= SECURITY_FAIL_THRESHOLD or 
        all_results['average_score'] < QUALITY_SCORE_THRESHOLD
    )
    
    return all_results

def format_ai_analysis_report(analysis: Dict, pr_info: Dict) -> str:
    """Generate comprehensive AI analysis report"""
    
    status_emoji = "🚨" if analysis['should_block'] else "✅"
    
    report = f"""## {status_emoji} AI Code Review Report (Powered by {MODEL_NAME})

**Pull Request:** {pr_info.get('title', 'Unknown PR')}
**Author:** {pr_info.get('user', {}).get('login', 'Unknown')}
**Files Analyzed:** {analysis['files_count']}
**Lines Analyzed:** {analysis['lines_count']}
**Overall Score:** {analysis['average_score']:.1f}/100
**AI Model:** {MODEL_NAME} (GGUF)

---

"""
    
    # Security Issues
    if analysis['security_issues']:
        report += "### 🔒 Security Issues Found\n"
        for issue in analysis['security_issues']:
            severity_emoji = {
                'critical': '🚨', 'high': '⚠️', 'medium': '🔶', 'low': 'ℹ️'
            }.get(issue.get('severity', 'medium'), '⚠️')
            
            report += f"- {severity_emoji} **{issue.get('severity', 'medium').upper()}**: {issue.get('description', 'Security issue detected')}\n"
            if issue.get('line_content'):
                report += f"  ```\n  {issue['line_content']}\n  ```\n"
            if issue.get('suggestion'):
                report += f"  **💡 Suggestion:** {issue['suggestion']}\n"
            report += "\n"
    
    # Bugs
    if analysis['bugs']:
        report += "### 🐛 Bugs Detected\n"
        for bug in analysis['bugs']:
            severity_emoji = {'high': '🚨', 'medium': '⚠️', 'low': 'ℹ️'}.get(bug.get('severity', 'medium'), '⚠️')
            
            report += f"- {severity_emoji} **{bug.get('severity', 'medium').upper()}**: {bug.get('description', 'Bug detected')}\n"
            if bug.get('line_content'):
                report += f"  ```\n  {bug['line_content']}\n  ```\n"
            if bug.get('suggestion'):
                report += f"  **💡 Fix:** {bug['suggestion']}\n"
            report += "\n"
    
    # Code Quality
    if analysis['code_quality']:
        report += "### 📊 Code Quality Issues\n"
        for issue in analysis['code_quality']:
            severity_emoji = {'medium': '🔶', 'low': 'ℹ️'}.get(issue.get('severity', 'low'), 'ℹ️')
            
            report += f"- {severity_emoji} **{issue.get('severity', 'low').upper()}**: {issue.get('description', 'Quality issue')}\n"
            if issue.get('line_content'):
                report += f"  ```\n  {issue['line_content']}\n  ```\n"
            if issue.get('suggestion'):
                report += f"  **💡 Improvement:** {issue['suggestion']}\n"
            report += "\n"
    
    # Refactoring Suggestions (if any from AI)
    if analysis.get('refactoring_suggestions'):
        report += "### 🔧 AI Refactoring Suggestions\n"
        for suggestion in analysis['refactoring_suggestions']:
            report += f"- **{suggestion.get('type', 'optimization').replace('_', ' ').title()}**: {suggestion.get('description', 'Code improvement suggested')}\n"
            if suggestion.get('line_content'):
                report += f"  **Original:**\n  ```\n  {suggestion['line_content']}\n  ```\n"
            if suggestion.get('suggestion'):
                report += f"  **💡 Benefit:** {suggestion['suggestion']}\n"
            report += "\n"
    
    # Refactored Files Section (complete file refactoring)
    if analysis.get('refactored_files'):
        report += "### 🔧 Complete Refactored Files\n"
        report += "Below are the complete refactored versions of files with all issues fixed:\n\n"
        
        for refactored_file in analysis['refactored_files']:
            file_path = refactored_file['file_path']
            issues_fixed = refactored_file['issues_fixed']
            refactored_code = refactored_file['refactored_code']
            
            report += f"#### 📄 {file_path}\n"
            report += f"**Issues Fixed:** {issues_fixed} | **Original Lines:** {refactored_file['original_lines']}\n\n"
            report += f"**Complete Refactored Code:**\n"
            report += f"```{get_file_language(file_path)}\n{refactored_code}\n```\n\n"
    
    # Summary
    if not any([analysis['security_issues'], analysis['bugs'], analysis['code_quality'], analysis['refactoring_suggestions']]):
        report += "### ✅ No Issues Found\n"
        report += "Excellent work! No major issues detected by AI analysis.\n\n"
    else:
        # Summary stats
        total_issues = len(analysis['security_issues']) + len(analysis['bugs']) + len(analysis['code_quality'])
        total_suggestions = len(analysis['refactoring_suggestions'])
        
        if total_suggestions > 0:
            report += f"### 📈 Summary\n"
            report += f"**Issues Found:** {total_issues} | **Refactoring Suggestions:** {total_suggestions}\n\n"
    
    # Status and Next Steps
    if analysis['should_block']:
        critical_count = len([i for i in analysis['security_issues'] if i.get('severity') in ['critical', 'high']])
        report += "### 🚫 Merge Blocked\n"
        report += f"This PR has {critical_count} critical/high-severity issues that must be resolved.\n\n"
        report += "**Next Steps:**\n"
        report += "1. 🔧 Fix the security and high-priority issues above\n"
        report += "2. 🧪 Test your changes thoroughly\n"
        report += "3. 🔄 Push fixes to trigger new AI analysis\n\n"
    else:
        report += "### ✅ Ready for Review\n"
        report += "AI analysis completed successfully. No blocking issues found.\n\n"
    
    # File Analysis Summary
    report += "### 📁 File Analysis Summary\n"
    for file_analysis in analysis['files_analyzed']:
        file_score = file_analysis.get('overall_score', 80)
        score_emoji = "🟢" if file_score >= 80 else "🟡" if file_score >= 60 else "🔴"
        confidence = file_analysis.get('confidence', 0.8)
        
        report += f"- {score_emoji} **{file_analysis['file_path']}**: Score {file_score}/100 (Confidence: {confidence:.0%})\n"
    
    report += f"\n*AI Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC*\n"
    report += f"*Powered by {MODEL_NAME} (GGUF)*"
    
    return report

def generate_merge_request_description(analysis: Dict, pr_info: Dict) -> str:
    """Generate a high-level summary of the merge request based on code changes and AI review"""
    summary_lines = []

    summary_lines.append(f"### 📜 Merge Request Summary\n")
    summary_lines.append(f"**PR Title:** {pr_info.get('title', 'Unknown PR')}")
    summary_lines.append(f"**Author:** {pr_info.get('user', {}).get('login', 'Unknown')}")
    summary_lines.append(f"**Files Changed:** {analysis.get('files_count', 0)}")
    summary_lines.append(f"**Lines Modified:** {analysis.get('lines_count', 0)}\n")

    # Insert business logic summary if available
    if analysis.get('code_summaries'):
        summary_lines.append("### 🧐 Code Purpose Summary")
        for summary in analysis['code_summaries']:
            summary_lines.append(f"- {summary}")
        summary_lines.append("\n")

    if analysis.get('should_block'):
        summary_lines.append("⚠️ **This MR contains critical or high-severity issues that must be resolved before merging.**\n")
    else:
        summary_lines.append("✅ **No critical issues found. This MR appears to be safe to merge from a static analysis perspective.**\n")

    summary_lines.append("### 🔍 Change Highlights\n")
    if analysis['security_issues']:
        summary_lines.append(f"- {len(analysis['security_issues'])} potential security issues detected")
    if analysis['bugs']:
        summary_lines.append(f"- {len(analysis['bugs'])} bugs or logic problems flagged")
    if analysis['code_quality']:
        summary_lines.append(f"- {len(analysis['code_quality'])} code quality issues noted")
    if analysis['refactoring_suggestions']:
        summary_lines.append(f"- {len(analysis['refactoring_suggestions'])} suggestions for improving structure or performance")

    summary_lines.append("\n### 🧠 Suggested Action")
    if analysis['should_block']:
        summary_lines.append("Please address the issues listed in the AI Code Review Report before merging.")
    else:
        summary_lines.append("You may proceed with merging after reviewing minor improvements if desired.")

    return '\n'.join(summary_lines)



def process_pull_request_with_ai(payload: Dict):
    """Process GitHub pull request with AI analysis"""
    try:
        logger.info("🤖 Starting AI-powered PR processing...")
        
        action = payload.get('action')
        pr_data = payload.get('pull_request', {})
        repository = payload.get('repository', {})
        
        # Extract repository info
        owner = repository.get('owner', {}).get('login')
        repo_name = repository.get('name')
        pr_number = pr_data.get('number')
        
        logger.info(f"📋 PR Data - Repo: {owner}/{repo_name}, PR: {pr_number}, Action: {action}")
        
        # Only process on relevant actions
        if action not in ['opened', 'synchronize', 'reopened']:
            logger.info(f"⏭️ Skipping action: {action}")
            return
        
        # Extract code changes - NOW COMMIT SPECIFIC
        logger.info("📊 Extracting code changes from specific commit...")
        code_changes = extract_code_changes_from_webhook(payload, action)
        
        if not code_changes:
            logger.info("ℹ️ No code changes to analyze")
            return
        
        # Perform AI analysis
        logger.info("🔍 Starting AI analysis with Direct GGUF model...")
        analysis_result = analyze_files_with_ai(code_changes)
        
        logger.info(f"✅ AI analysis completed - Score: {analysis_result['average_score']:.1f}/100")
        logger.info(f"📊 Found: {len(analysis_result['security_issues'])} security, {len(analysis_result['bugs'])} bugs, {len(analysis_result['code_quality'])} quality issues")
        
        # Generate and post report
        logger.info("📝 Generating AI analysis report...")
        report = format_ai_analysis_report(analysis_result, pr_data)
        
        # Post both general comment and inline comments
        if github_client:
            commit_sha = pr_data.get('head', {}).get('sha')
            
            logger.info("💬 Posting AI analysis to GitHub...")
            logger.info(f"🔍 DEBUG: Commit SHA: {commit_sha}")
            logger.info(f"🔍 DEBUG: Inline comments to post: {len(analysis_result.get('inline_comments', []))}")
            
            # Post inline comments first
            if analysis_result.get('inline_comments') and commit_sha:
                logger.info(f"📝 Posting {len(analysis_result['inline_comments'])} inline comments...")
                
                successful_inline_comments = 0
                for i, comment in enumerate(analysis_result['inline_comments']):
                    logger.info(f"📝 DEBUG: Posting inline comment {i+1}: {comment['path']}:{comment['line']}")
                    logger.info(f"📝 DEBUG: Comment body: {comment['body'][:100]}...")
                    
                    
                    success = github_client.create_pr_review_comment(
                        owner, repo_name, pr_number, commit_sha,
                        comment['path'], comment['line'], comment['body']
                    )
                    if success:
                        successful_inline_comments += 1
                        logger.info(f"✅ DEBUG: Successfully posted inline comment {i+1}")
                    else:
                        logger.error(f"❌ DEBUG: Failed to post inline comment {i+1}")
                
                logger.info(f"✅ Posted {successful_inline_comments}/{len(analysis_result['inline_comments'])} inline comments")
            else:
                if not analysis_result.get('inline_comments'):
                    logger.warning("⚠️ DEBUG: No inline comments to post")
                if not commit_sha:
                    logger.warning("⚠️ DEBUG: No commit SHA available")
            
            # Post general summary comment
            logger.info("📋 Posting general summary comment...")
            mr_description = generate_merge_request_description(analysis_result, pr_data)

            github_client.create_pr_comment(owner, repo_name, pr_number, mr_description)
        else:
            logger.warning("⚠️ GitHub client not available, saving report locally")
            # Save to file
            report_file = f"reports/ai_review_pr_{pr_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            os.makedirs('reports', exist_ok=True)
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"💾 Report saved to: {report_file}")
        
        logger.info(f"🎉 AI analysis completed for PR {pr_number}")
        
    except Exception as e:
        logger.error(f"❌ Error in AI-powered PR processing: {e}", exc_info=True)

# Global instances
github_client = GitHubClient(GITHUB_TOKEN) if GITHUB_TOKEN else None

@app.route('/')
def home():
    """Root endpoint"""
    return jsonify({
        'message': 'AI Code Review Agent (Direct GGUF Powered)',
        'status': 'running',
        'version': '3.0.0',
        'ai_provider': 'direct_gguf',
        'model': MODEL_NAME,
        'model_path': MODEL_PATH,
        'endpoints': {
            'health': '/health',
            'webhook': '/webhook (POST)',
            'test_ai': '/test-ai (GET)',
            'analyze': '/analyze (POST)'
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'platform': 'github',
        'ai_provider': 'direct_gguf',
        'model': MODEL_NAME,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'use_gpu': USE_GPU,
        'gpu_layers': GPU_LAYERS,
        'context_size': CONTEXT_SIZE,
        'cpu_threads': CPU_THREADS,
        'github_configured': bool(GITHUB_TOKEN),
        'webhook_secret_configured': bool(WEBHOOK_SECRET)
    }
    
    # Check GPU availability
    if USE_GPU:
        try:
            import torch
            health_status['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                health_status['gpu_name'] = torch.cuda.get_device_name(0)
                health_status['gpu_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
        except ImportError:
            health_status['cuda_available'] = False
            health_status['pytorch_installed'] = False
    
    # Test GGUF model availability
    try:
        analyzer = ConfigurableModelAnalyzer()
        health_status['model_loaded'] = analyzer.llama is not None
        if analyzer.llama:
            health_status['model_type'] = analyzer._detect_model_type() if analyzer.model_type == 'auto' else analyzer.model_type
            health_status['chat_template'] = analyzer._get_chat_template()
    except Exception as e:
        logger.error(f"Health check error: {e}")
        health_status['model_loaded'] = False
    
    # Test GitHub connectivity
    if github_client:
        try:
            response = github_client.session.get(f"{GITHUB_URL}/user", timeout=5)
            health_status['github_connectivity'] = response.status_code == 200
            if response.status_code == 200:
                user_data = response.json()
                health_status['github_user'] = user_data.get('login')
        except:
            health_status['github_connectivity'] = False
    else:
        health_status['github_connectivity'] = False
    
    return jsonify(health_status)

@app.route('/test-ai')
def test_ai():
    """Test AI analysis with sample code"""
    try:
        analyzer = ConfigurableModelAnalyzer()
        
        # Test with sample problematic code
        test_code = [
            'const API_KEY = "sk-1234567890abcdef";',
            'const password = "admin123";',
            'console.log("Debug: API_KEY =", API_KEY);',
            'if (user.email = "test@example.com") {',
            '  return true;',
            '}',
            '// TODO: Fix this security issue'
        ]
        
        logger.info("🧪 Testing AI analysis with sample code...")
        result = analyzer.analyze_code_with_llama('test.js', test_code)
        
        return jsonify({
            'message': 'AI test completed (Configurable Model)',
            'model': MODEL_NAME,
            'model_type': analyzer._detect_model_type() if analyzer.model_type == 'auto' else analyzer.model_type,
            'chat_template': analyzer._get_chat_template(),
            'test_code': test_code,
            'analysis_result': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in AI test: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Handle GitHub webhook events"""
    try:
        # Verify webhook signature
        signature = request.headers.get('X-Hub-Signature', '')
        if not verify_github_signature(request.data, signature):
            logger.warning("Invalid webhook signature")
            return jsonify({'error': 'Invalid signature'}), 403
        
        payload = request.json
        event_type = request.headers.get('X-GitHub-Event')
        
        logger.info(f"Received GitHub webhook: {event_type}")
        
        # Handle pull request events
        if event_type == 'pull_request':
            # Process in background with AI
            threading.Thread(
                target=process_pull_request_with_ai,
                args=(payload,)
            ).start()
            
            return jsonify({
                'message': 'AI-powered PR analysis started', 
                'status': 'processing',
                'ai_provider': 'direct_gguf',
                'model': MODEL_NAME
            }), 202
        
        return jsonify({'message': 'Event not handled', 'event': event_type}), 200
        
    except Exception as e:
        logger.error(f"Error handling webhook: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyze', methods=['POST'])
def manual_analysis():
    """Manual AI analysis endpoint"""
    try:
        data = request.json
        code = data.get('code', '')
        filename = data.get('filename', 'test.js')
        
        if not code:
            return jsonify({'error': 'code parameter required'}), 400
        
        analyzer = ConfigurableModelAnalyzer()
        code_lines = code.split('\n')
        
        logger.info(f"🔍 Manual analysis of {filename} ({len(code_lines)} lines)")
        result = analyzer.analyze_code_with_llama(filename, code_lines)
        
        return jsonify({
            'message': 'Manual analysis completed',
            'filename': filename,
            'lines_analyzed': len(code_lines),
            'analysis': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in manual analysis: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Validate configuration
    if not GITHUB_TOKEN:
        logger.error("GITHUB_TOKEN environment variable is required")
        exit(1)
    
    # Log configuration
    logger.info("🤖 AI Code Review Agent Starting (Direct GGUF Powered)...")
    logger.info(f"GitHub URL: {GITHUB_URL}")
    logger.info(f"AI Provider: Direct GGUF")
    logger.info(f"AI Model: {MODEL_NAME}")
    logger.info(f"Model Path: {MODEL_PATH}")
    logger.info(f"Model Exists: {os.path.exists(MODEL_PATH)}")
    logger.info(f"GPU Enabled: {USE_GPU}")
    logger.info(f"GPU Layers: {GPU_LAYERS}")
    logger.info(f"Context Size: {CONTEXT_SIZE}")
    logger.info(f"CPU Threads: {CPU_THREADS}")
    logger.info(f"Security Threshold: {SECURITY_FAIL_THRESHOLD}")
    logger.info(f"Quality Threshold: {QUALITY_SCORE_THRESHOLD}")
    
    port = int(os.getenv('PORT', 5000))
    
    try:
        app.run(host='0.0.0.0', port=port, debug=DEBUG)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise
