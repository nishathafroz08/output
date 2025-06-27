# Smart Quiz Generator Microservice

A containerized, goal-aligned quiz generator microservice that generates multiple-choice and short-answer questions dynamically from local data sources using offline AI models.

## 🎯 Overview

This microservice accepts user goals (e.g., "GATE ECE", "Amazon SDE") and returns JSON arrays of contextually relevant quiz questions. It operates entirely offline using retrieval-based methods and local pretrained models.

## ✨ Features

- **Goal-Aligned Generation**: Generates questions based on specific exam/interview goals
- **Offline Operation**: No external API dependencies - runs completely offline
- **Dual Generation Methods**: 
  - TF-IDF based retrieval from curated question banks
  - Local T5-small model for dynamic question generation
- **Flexible Question Types**: Supports both MCQ and short-answer formats
- **Production Ready**: Fully containerized with Docker
- **Configurable**: Runtime behavior controlled via `config.json`
- **Comprehensive Datasets**: Covers Amazon SDE, GATE CSE, GATE ECE, ML, Cyber Security, and AWS domains

## 📋 Requirements

- Python 3.10+
- Docker
- 4GB+ RAM (for local model inference)
- 2GB+ storage (for datasets and models)

## 🚀 Quick Start

### Using Docker (Recommended)

```bash
# Build the container
docker build -t smart-quiz .

# Run the service
docker run -p 8000:8000 smart-quiz

# Service will be available at http://localhost:8000
```

### Local Development

```bash
# Clone the repository
git clone <repository-url>
cd smart-quiz

# Install dependencies
pip install -r requirements.txt

# Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 📊 Supported Domains

- **Amazon SDE**: Software Development Engineer interview questions
- **GATE CSE**: Computer Science Engineering entrance exam
- **GATE ECE**: Electronics and Communication Engineering entrance exam  
- **ML**: Machine Learning concepts and algorithms
- **Cyber Security**: Information security and cybersecurity topics
- **AWS**: Amazon Web Services cloud computing questions

## 🔧 API Endpoints

### Generate Quiz Questions

**POST** `/generate`

Generate goal-aligned quiz questions based on specified parameters.

**Request Body:**
```json
{
  "goal": "Amazon SDE",
  "num_questions": 5,
  "difficulty": "intermediate"
}
```

**Supported Goals:**
- `"Amazon SDE"`
- `"GATE CSE"`
- `"GATE ECE"`
- `"ML"`  
- `"Cyber Security"`
- `"AWS"`

**Response:**
```json
{
  "quiz_id": "quiz_1234",
  "goal": "Amazon SDE",
  "questions": [
    {
      "type": "mcq",
      "question": "What is the time complexity of binary search?",
      "options": ["O(n)", "O(log n)", "O(n log n)", "O(1)"],
      "answer": "O(log n)",
      "difficulty": "intermediate",
      "topic": "Algorithms"
    }
  ]
}
```

### Health Check

**GET** `/health`

Returns service health status.

**Response:**
```json
{
  "status": "ok"
}
```

### Version Information

**GET** `/version`

Returns metadata about the model and configuration.

**Response:**
```json
{
  "version": "1.0.0",
  "generator_mode": "retrieval",
  "supported_goals": ["Amazon SDE", "GATE CSE", "GATE ECE", "ML", "Cyber Security", "AWS"]
}
```

## 📁 Project Structure

```
smart-quiz/
├── app/
│   ├── main.py              # FastAPI application
│   ├── generator.py         # Question generation logic
│   └── model/              # TF-IDF vectorizer and local models
├── data/
│   └── question_bank.json  # Curated question datasets
├── config.json             # Runtime configuration
├── Dockerfile              # Container setup
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── schema.json            # API input/output schemas
└── tests/
    └── test_generate.py   # API validation tests
```

## ⚙️ Configuration

The service behavior is controlled via `config.json`:

```json
{
  "generator_mode": "retrieval",
  "version": "1.0.0",
  "default_num_questions": 5,
  "max_questions": 10,
  "supported_difficulties": ["beginner", "intermediate", "advanced"],
  "supported_types": ["mcq", "short_answer"],
  "supported_goals": [
    "Amazon SDE",
    "GATE CSE", 
    "GATE ECE",
    "ML",
    "Cyber Security",
    "AWS"
  ]
}
```

### Configuration Options

| Parameter | Description | Values |
|-----------|-------------|---------|
| `generator_mode` | Generation method | `"retrieval"`, `"model"` |
| `max_questions` | Maximum questions per request | Integer (1-50) |
| `supported_difficulties` | Valid difficulty levels | Array of strings |
| `supported_goals` | Available domains | Array of goal strings |

## 🧪 Testing

Run the test suite to validate API functionality:

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python tests/test_generate.py
```

### Test Coverage

- Output structure validation
- Question count verification  
- Response time validation (< 1.5s for 5 questions)
- Error handling for invalid inputs
- Configuration validation

## 🔍 Generation Methods

### 1. Retrieval-Based (TF-IDF)

Uses TF-IDF vectorization to match user goals with relevant questions from the curated question bank.

**Advantages:**
- Fast response times
- High relevance to specified goals
- Consistent quality

### 2. Local Model (T5-Small)

Employs a locally hosted T5-small model for dynamic question generation.

**Advantages:**  
- Novel question generation
- Contextual understanding
- Adaptable to new domains

## 📈 Performance

- **Response Time**: < 1.5 seconds for 5 questions
- **Throughput**: ~50 requests/minute
- **Memory Usage**: ~2GB (including models)
- **CPU Usage**: Moderate (optimized for inference)

## 🚦 Error Handling

The API handles various error scenarios gracefully:

| Error Type | HTTP Status | Response |
|------------|-------------|----------|
| Invalid goal | 400 | `{"error": "Unsupported goal"}` |
| Invalid difficulty | 400 | `{"error": "Invalid difficulty level"}` |
| Exceeds max questions | 400 | `{"error": "Too many questions requested"}` |
| Missing config | 500 | `{"error": "Configuration not found"}` |

## 🐳 Docker Configuration

The Dockerfile uses Python 3.10 slim base image and includes:

- All model files and datasets
- Configuration files
- Optimized for production deployment
- Port 8000 exposure
- Uvicorn ASGI server

## 📝 Example Usage

```bash
# Generate Amazon SDE questions
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "goal": "Amazon SDE",
       "num_questions": 3,
       "difficulty": "intermediate"
     }'

# Generate GATE CSE questions  
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "goal": "GATE CSE", 
       "num_questions": 5,
       "difficulty": "advanced"
     }'

# Generate AWS questions
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "goal": "AWS",
       "num_questions": 4,
       "difficulty": "beginner"
     }'
```

## 🔒 Security Considerations

- No external network dependencies
- Input validation and sanitization
- Rate limiting (configurable)
- Error message sanitization
- Secure containerization

## 🛠️ Development

### Adding New Domains

1. Add questions to `data/question_bank.json`
2. Update `supported_goals` in `config.json`
3. Retrain TF-IDF vectorizer if using retrieval mode
4. Update tests and documentation

### Customizing Generation Logic

Modify `app/generator.py` to implement custom generation strategies while maintaining the API contract.

## 📊 Dataset Statistics

| Domain | Questions | Difficulty Levels | Topics Covered |
|--------|-----------|------------------|----------------|
| Amazon SDE | 300+ | All | DSA, System Design, OOP |
| GATE CSE | 250+ | All | OS, Networks, Algorithms |
| GATE ECE | 250+ | All | Signals, Communications |
| ML | 200+ | All | Supervised, Unsupervised |
| Cyber Security | 200+ | All | Cryptography, Networks |
| AWS | 200+ | All | Services, Architecture |

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-domain`)
3. Add questions and update configuration
4. Test thoroughly
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the API response for error details
2. Validate input against `schema.json`
3. Review logs in Docker container
4. Ensure `config.json` is properly formatted

---

**Note**: This microservice operates entirely offline and does not require internet connectivity after initial setup.
