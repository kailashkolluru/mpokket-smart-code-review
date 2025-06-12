# AI Code Review Agent - Technical Documentation

## 🎯 Problem Statement & Solution

### Problem Statement
Manual code reviews are time-consuming, inconsistent, and often miss critical security vulnerabilities and bugs. Development teams struggle with:
- **Security vulnerabilities** going undetected (hardcoded credentials, SQL injection, etc.)
- **Inconsistent review quality** across different reviewers
- **Time bottlenecks** in the development pipeline
- **Missing edge cases** and logic errors
- **Lack of automated quality enforcement**

### Our Solution: AI-Powered Code Review Agent
An intelligent GitHub integration that automatically analyzes pull requests using local AI models, providing:
- **Real-time security vulnerability detection**
- **Automated bug identification and suggestions**
- **Code quality assessment with scoring**
- **Inline comments** on specific problematic lines
- **Complete refactored code suggestions**
- **Commit-specific analysis** for incremental reviews

**Key Innovation**: Uses local GGUF models (like Llama, CodeLlama, Solar) for privacy-focused analysis without sending code to external APIs.

---

## 🛠️ Technical Specifications

### Core Technologies

#### Backend Framework
- **Language**: Python 3.8+
- **Web Framework**: Flask
- **Architecture**: Event-driven webhook processing

#### AI/ML Stack
- **Model Format**: GGUF (GPT-Generated Unified Format)
- **Model Library**: llama-cpp-python
- **Supported Models**: 
  - Llama 3/3.1 (General purpose)
  - CodeLlama (Code-specialized)
  - Solar 10.7B (Instruction-tuned)
  - DeepSeek Coder
  - StarCoder
  - Qwen Code models

#### GitHub Integration
- **API**: GitHub REST API v3
- **Authentication**: Personal Access Tokens
- **Webhooks**: Real-time PR event processing
- **Security**: HMAC signature verification

#### Dependencies
```python
# Core Dependencies
flask>=2.0.0
requests>=2.25.0
python-dotenv>=0.19.0
llama-cpp-python>=0.2.0

# Optional GPU Support
torch>=1.9.0  # For CUDA detection
```

### AI Model Configuration
- **Context Window**: 2048-8192 tokens (configurable)
- **Temperature**: 0.1 (low for deterministic analysis)
- **Max Tokens**: 1500 per analysis
- **GPU Acceleration**: Optional CUDA support
- **Quantization**: 4-bit (Q4_K_M) for efficiency

---

## 🏗️ Infrastructure Requirements

### Minimum System Requirements
```yaml
CPU: 4 cores (Intel/AMD x64)
RAM: 8GB (16GB recommended)
Storage: 10GB free space
OS: Linux/macOS/Windows
Python: 3.8 or higher
```

### Recommended Production Setup
```yaml
CPU: 8+ cores
RAM: 32GB
GPU: NVIDIA RTX 3060+ (optional, 8GB+ VRAM)
Storage: 50GB SSD
Network: Stable internet for GitHub API
```

### Model Storage Requirements
| Model Type | Size | RAM Usage | Description |
|------------|------|-----------|-------------|
| Llama 3 8B (Q4) | ~4.5GB | ~6GB | General code analysis |
| CodeLlama 7B (Q4) | ~4.0GB | ~5GB | Code-specialized |
| Solar 10.7B (Q4) | ~6.2GB | ~8GB | High-quality analysis |

### Network Requirements
- **GitHub API Access**: https://api.github.com
- **Webhook Endpoint**: Public IP or ngrok tunnel
- **Bandwidth**: ~1-10MB per PR analysis

### Security Considerations
- **Local Processing**: Models run locally, no code sent to external APIs
- **Token Security**: GitHub tokens stored as environment variables
- **Webhook Verification**: HMAC signature validation
- **Rate Limiting**: Built-in GitHub API rate limit handling

---

## 🚀 Setup Instructions

### Step 1: Environment Preparation

#### 1.1 Clone the Repository
```bash
git clone https://github.com/your-org/ai-code-review-agent.git
cd ai-code-review-agent
```

#### 1.2 Install Python Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 1.3 Download AI Model
```bash
# Create models directory
mkdir -p models/llama3/models/solar

# Download Solar 10.7B model (example)
wget -O models/llama3/models/solar/solar-10.7b-instruct-v1.0.Q4_K_M.gguf \
  "https://huggingface.co/TheBloke/SOLAR-10.7B-Instruct-v1.0-GGUF/resolve/main/solar-10.7b-instruct-v1.0.Q4_K_M.gguf"
```

### Step 2: GitHub Configuration

#### 2.1 Create GitHub Personal Access Token
1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Generate new token with scopes:
   - `repo` (Full repository access)
   - `pull_requests:write` (Comment on PRs)
   - `contents:read` (Read repository contents)

#### 2.2 Configure Webhook
1. Go to your repository Settings → Webhooks
2. Add webhook with:
   - **Payload URL**: `https://your-domain.com/webhook`
   - **Content type**: `application/json`
   - **Events**: "Pull requests" only
   - **Secret**: Generate a random secret key

### Step 3: Environment Configuration

#### 3.1 Create Environment File
```bash
# Create .env file
cat > .env << EOF
# GitHub Configuration
GITHUB_TOKEN=ghp_your_github_token_here
GITHUB_URL=https://api.github.com
WEBHOOK_SECRET=your_webhook_secret_here

# AI Model Configuration
MODEL_PATH=./models/llama3/models/solar/solar-10.7b-instruct-v1.0.Q4_K_M.gguf
MODEL_NAME=solar-10.7b-instruct
MODEL_TYPE=auto

# GPU Configuration (optional)
USE_GPU=false
GPU_LAYERS=32
CONTEXT_SIZE=4096
CPU_THREADS=8

# Analysis Configuration
MAX_TOKENS=1500
TEMPERATURE=0.1
TOP_P=0.9
ENABLE_REFACTORING=true
SECURITY_FAIL_THRESHOLD=1
QUALITY_SCORE_THRESHOLD=70

# Application Configuration
PORT=5000
DEBUG=false
EOF
```

#### 3.2 Verify Configuration
```bash
# Test model loading
python -c "
from llama_cpp import Llama
model = Llama('./models/llama3/models/solar/solar-10.7b-instruct-v1.0.Q4_K_M.gguf', verbose=False)
print('✅ Model loaded successfully!')
"
```

### Step 4: Application Deployment

#### 4.1 Local Development
```bash
# Start the application
python app.py

# Test endpoints
curl http://localhost:5000/health
curl http://localhost:5000/test-ai
```

#### 4.2 Production Deployment

##### Using Docker (Recommended)
```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000

CMD ["python", "app.py"]
```

```bash
# Build and run
docker build -t ai-code-review .
docker run -p 5000:5000 --env-file .env ai-code-review
```

##### Using systemd (Linux)
```bash
# Create service file
sudo tee /etc/systemd/system/ai-code-review.service << EOF
[Unit]
Description=AI Code Review Agent
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/path/to/ai-code-review-agent
Environment=PATH=/path/to/venv/bin
ExecStart=/path/to/venv/bin/python app.py
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl enable ai-code-review
sudo systemctl start ai-code-review
```

### Step 5: Tunnel Setup (Development)

#### Using ngrok
```bash
# Install ngrok
npm install -g ngrok

# Start tunnel
ngrok http 5000

# Use the https URL for GitHub webhook
```

#### Using CloudFlare Tunnel
```bash
# Install cloudflared
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o cloudflared
chmod +x cloudflared

# Start tunnel
./cloudflared tunnel --url http://localhost:5000
```

### Step 6: Testing & Verification

#### 6.1 Health Check
```bash
curl https://your-domain.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "github_connectivity": true,
  "cuda_available": false
}
```

#### 6.2 Manual Analysis Test
```bash
curl -X POST https://your-domain.com/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "filename": "test.js",
    "code": "const API_KEY = \"sk-1234567890\";\nconsole.log(API_KEY);"
  }'
```

#### 6.3 End-to-End Test
1. Create a test PR with security issues
2. Verify webhook triggers analysis
3. Check for inline comments and summary report

---

## 🔧 Configuration Options

### Model Selection Guide
| Use Case | Recommended Model | Size | Best For |
|----------|------------------|------|----------|
| General Reviews | Llama 3 8B | 4.5GB | Balanced analysis |
| Code-Focused | CodeLlama 7B | 4.0GB | Code understanding |
| High Quality | Solar 10.7B | 6.2GB | Detailed analysis |
| Security-First | DeepSeek Coder | 5.0GB | Vulnerability detection |

### Performance Tuning
```env
# For faster processing (lower quality)
CONTEXT_SIZE=2048
MAX_TOKENS=1000
CPU_THREADS=16

# For higher quality (slower)
CONTEXT_SIZE=8192
MAX_TOKENS=2000
CPU_THREADS=4
```

### Custom Prompts
```env
# Override system prompt
SYSTEM_PROMPT="You are a security-focused code reviewer..."

# Add custom prefixes
USER_PROMPT_PREFIX="Focus on authentication and authorization:"
```

---

## 📊 Monitoring & Logs

### Application Logs
```bash
# View real-time logs
tail -f app.log

# Search for errors
grep -i error app.log

# Monitor AI analysis performance
grep "AI analysis completed" app.log
```

### Key Metrics to Monitor
- **Analysis Time**: Average time per PR analysis
- **Model Memory Usage**: RAM consumption during analysis
- **GitHub API Rate Limits**: Remaining API calls
- **Error Rates**: Failed analyses or webhook processing
- **Issue Detection Accuracy**: Manual verification of findings

---

## 🔒 Security Best Practices

1. **Never commit tokens**: Use environment variables only
2. **Validate webhooks**: Always verify HMAC signatures
3. **Local processing**: Models run locally, code never leaves your infrastructure
4. **Regular updates**: Keep dependencies updated for security patches
5. **Access control**: Restrict webhook endpoint access
6. **Log sanitization**: Don't log sensitive code content

---

## 🤝 Contributing & Customization

### Adding New Models
1. Download GGUF model file
2. Update `MODEL_PATH` in configuration
3. Test with `/test-ai` endpoint
4. Adjust prompts if needed

### Custom Analysis Rules
Modify the `_create_analysis_prompt()` method to add:
- Language-specific rules
- Company coding standards
- Custom security patterns
- Domain-specific validations

### Extending Platform Support
The architecture supports adding:
- GitLab webhooks
- Bitbucket integration
- Azure DevOps
- Custom SCM systems

---

*Built for AI Hackathon 2025 - Transforming Code Review with Local AI*
