# Chapter Quality Evaluation Framework

A comprehensive system for evaluating LLM-generated video chapters with automated metrics and manual review integration.

## **Core Features**

- **Chapter Generation**: Extract transcripts and generate chapters using LLMs
- **Automated Evaluation**: BERTScore, ROUGE, semantic similarity, and LLM error detection
- **Manual Review Integration**: Standardized scoring and quality assessment
- **Search Relevance**: User query-based evaluation and navigation utility
- **Visualization**: Comprehensive dashboards and quality metrics
- **Quality Assurance**: Inter-reviewer agreement and system calibration

## **Project Structure**

```
evaluation/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.py                          # Setup script
├── .gitignore                        # Git ignore rules
├── evaluator.py                      # Main evaluator script
│
├── src/                              # Source code
│   ├── vidtranscript2chapter.py      # Chapter generation from YouTube transcripts
│   ├── chapter_evaluator.py          # Core evaluation engine
│   └── visualization_dashboard.py    # Dashboard generation
│
├── docs/                             # Documentation
│   ├── MANUAL_REVIEW_WORKFLOW.md     # Manual review process guide
│   └── EVALUATION_FRAMEWORK.md       # Framework documentation
│
├── examples/                         # Example files
│   └── example_manual_review_format.json
│
├── data/                            # Generated data
│   ├── extracted_chapters/          # Chapter data
│   │   └── chapters_[video_id].json
│   ├── outputs/                     # Evaluation results and visualizations
│   │   ├── evaluation_[video_id].json
│   │   ├── reports_[video_id].json
│   │   └── dashboard_[video_id].png
│   └── manual_reviews/              # Manual review data
│       └── manual_review_[video_id].json
```

## **Quick Start**

### **Installation**
```bash
pip install -r requirements.txt
```

### **Setup**
Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
```

### **Basic Usage**
```bash
# Generate chapters and evaluate quality
python evaluator.py P127jhj-8-Y

# With custom user queries
python evaluator.py P127jhj-8-Y --queries "machine learning,tutorial"

# Generate visualization dashboard
python evaluator.py P127jhj-8-Y --dashboard

# Enable manual review workflow
python evaluator.py P127jhj-8-Y --manual-review
```

## **Manual Review Workflow**

1. **Generate evaluation data**: `python evaluator.py P127jhj-8-Y --manual-review`
2. **Review chapters**: Edit `data/manual_reviews/manual_review_P127jhj-8-Y.json`
3. **Process reviews**: `python evaluator.py P127jhj-8-Y --review-file data/manual_reviews/manual_review_P127jhj-8-Y.json`

See `docs/MANUAL_REVIEW_WORKFLOW.md` for detailed instructions.

## Interactive Manual Annotation Tool (Streamlit)

A comprehensive web-based interface for human annotations with automated evaluation integration.

### **Quick Start**
```bash
pip install -r requirements.txt

# Start the interactive annotation tool
streamlit run annotation/app.py
```

### ✨ **Key Features**

#### ** Video Processing**
- **Auto-Detection**: Automatically detects new video IDs and runs evaluator
- **One-Click Setup**: Enter video ID → Press Enter → Everything happens automatically
- **Real-Time Status**: Visual indicators show data availability and processing status
- **Seamless Integration**: Direct integration with evaluator pipeline

#### ** Multi-Modal Annotation Interface**
- **Chapter Review**: Rate chapters on 6 quality dimensions (1-5 scale)
- **Search & Rate**: Interactive search with relevance rating system
- **Dashboard View**: Real-time visualization of evaluation results
- **Progress Tracking**: Visual progress indicators and completion status

#### ** Intelligent Workflow**
- **Auto-Load**: Existing data loads automatically when video ID is provided
- **Auto-Evaluate**: New videos trigger evaluator automatically (powered by o4-mini)
- **Real-Time Updates**: Immediate UI refresh and data synchronization
- **Error Handling**: Graceful error recovery with helpful user guidance

#### ** Multi-Reviewer Support**
- **Reviewer Management**: Support for multiple reviewers with unique IDs
- **Namespace Isolation**: Each reviewer's annotations are stored separately
- **Aggregation**: Automatic score aggregation across reviewers
- **Consistency Tracking**: Progress monitoring and completion status

###  **User Interface**

#### **Sidebar Controls**
- **Video ID Input**: Smart input with auto-evaluation on Enter key
- **Reviewer ID**: Unique identifier for annotation tracking
- **Status Indicators**: Real-time data availability and processing status
- **Action Buttons**: Manual load and evaluator controls

#### **Main Interface Pages**
1. **Chapter Review**: Individual chapter scoring and annotation
2. **Search & Rate**: Query-based search with relevance rating
3. **Dashboard**: Comprehensive evaluation visualization

### **Advanced Features**

#### **Smart Data Management**
- **Session Persistence**: Maintains state across page refreshes
- **Incremental Loading**: Only loads data when needed
- **Conflict Resolution**: Handles data conflicts gracefully
- **Backup Protection**: Atomic file operations prevent data loss

#### **Real-Time Feedback**
- **Live Progress**: Real-time progress bars and status updates
- **Debug Information**: Detailed logging for troubleshooting
- **Error Messages**: Clear, actionable error messages
- **Success Confirmation**: Visual feedback for completed actions

### **Annotation Workflow**

#### **For New Videos:**
1. Enter video ID (e.g., `9vM4p9NN0Ts`)
2. Press Enter or click "⚙️ Run Evaluator"
3. System automatically runs evaluator (powered by o4-mini)
4. Data loads automatically when ready
5. Start annotating chapters and search results

#### **For Existing Videos:**
1. Enter video ID
2. Data loads automatically
3. Continue or add new annotations
4. Real-time dashboard updates

### **Annotation Metrics**

#### **Chapter Review Scores (1-5 scale)**
- **Overall Quality**: Comprehensive quality assessment
- **Content Accuracy**: Factual alignment with source material
- **Title Appropriateness**: Accuracy and clarity of chapter titles
- **Summary Quality**: Completeness and coherence of summaries
- **Search Relevance**: User query satisfaction and relevance
- **Navigation Utility**: Usability and structural effectiveness

#### **Search Rating System**
- **Relevant**: Search result matches user intent
- **Irrelevant**: Search result doesn't match user intent
- **Not Rated**: Pending or skipped ratings

### **Data Flow**

1. **Input**: Video ID → Auto-detection → Evaluator (if needed)
2. **Processing**: Chapter generation → Evaluation → Manual review template
3. **Annotation**: Human scoring → Real-time aggregation → Dashboard update
4. **Output**: Updated reports → Regenerated dashboard → Persistent storage

### **Technical Implementation**

- **Framework**: Streamlit for responsive web interface
- **Backend**: Python subprocess integration with evaluator
- **Storage**: JSON-based data persistence with atomic operations
- **Visualization**: Matplotlib/Seaborn for dashboard generation
- **AI Integration**: OpenAI o4-mini for chapter generation and evaluation

## **Command Line Options**

```bash
python evaluator.py <video_id> [options]

Options:
    --no-eval                    Skip evaluation
    --queries query1,query2,...  Custom user queries
    --manual-review              Save evaluation results for manual review
    --review-file file.json      Load your completed manual reviews
    --dashboard                  Generate visualization dashboard
    --help                       Show help message
```

## **Evaluation Metrics**

### **Automated Metrics (0-1 scale)**
- **Content Relevance**: Semantic alignment with transcript
- **Title Accuracy**: How well titles describe content
- **Summary Completeness**: Adequacy of summaries
- **BERTScore**: Advanced semantic similarity (precision, recall, F1)
- **ROUGE**: N-gram overlap and sentence structure (1, 2, L)
- **Search Relevance**: Query-based relevance scoring
- **Navigation Utility**: User navigation effectiveness

### **Manual Review Metrics (1-5 scale)**
- **Overall Quality**: Comprehensive quality assessment
- **Content Accuracy**: Factual alignment with source
- **Title Appropriateness**: Accuracy and clarity
- **Summary Quality**: Completeness and coherence
- **Search Relevance**: User query satisfaction
- **Navigation Utility**: Usability and structure

### **LLM Error Detection**
- **Hallucination Score**: Fabricated content detection
- **Factual Consistency**: Source material alignment
- **Coherence Score**: Logical flow and structure
- **Bias Score**: Subjective content detection
- **Completeness Score**: Information coverage

## **Quality Assurance**

- **Inter-reviewer Agreement**: Consistency measurement
- **Automated vs Manual Alignment**: System calibration
- **Error Severity Assessment**: High/Medium/Low categorization
- **Continuous Improvement**: Feedback loop integration

## **Output Files**

### **Chapter Data (data/extracted_chapters/)**
- `chapters_[video_id].json`: Chapter data with timestamps

### **Evaluation Results (data/outputs/)**
- `evaluation_[video_id].json`: Detailed evaluation metrics
- `reports_[video_id].json`: Comprehensive quality reports
- `dashboard_[video_id].png`: Comprehensive evaluation dashboard

### **Manual Reviews (data/manual_reviews/)**
- `manual_review_[video_id].json`: Manual review data (when enabled)

## **Workflow Integration**

The framework supports multiple workflows:

1. **Automated Only**: Generate and evaluate chapters automatically
2. **Manual Review**: Human expertise validation and quality assurance
3. **Hybrid Approach**: Automated pre-screening with targeted manual review
4. **Continuous Improvement**: System learning from human feedback

## **Customization**

### **Custom User Queries**
```bash
python evaluator.py P127jhj-8-Y --queries "domain,specific,queries"
```

### **Programmatic Usage**
```python
from src.vidtranscript2chapter import VideoChapterGenerator
from src.chapter_evaluator import ChapterEvaluator

# Initialize components
generator = VideoChapterGenerator()
evaluator = ChapterEvaluator()

# Generate and evaluate
chapters = generator.generate_chapters(segments, video_id)
results = evaluator.evaluate_chapters(chapters, transcript_data)
```

## **Documentation**

- **Manual Review Process**: `docs/MANUAL_REVIEW_WORKFLOW.md`
- **Framework Details**: `docs/EVALUATION_FRAMEWORK.md`
- **Example Format**: `examples/example_manual_review_format.json`

## **Key Benefits**

1. **Single Entry Point**: One main script for all operations
2. **Comprehensive Quality Assessment**: Multi-dimensional evaluation
3. **Human-AI Collaboration**: Automated efficiency + human expertise
4. **Scalable Process**: Priority-based review assignment
5. **Continuous Improvement**: Learning from feedback
6. **User-Centric**: Search relevance and navigation optimization

## **Example Workflows**

### **Complete Evaluation with Dashboard**
```bash
python evaluator.py P127jhj-8-Y --queries "tutorial,beginner" --dashboard
```

### **Manual Review Process**
```bash
# Step 1: Generate evaluation data
python evaluator.py P127jhj-8-Y --manual-review

# Step 2: Edit manual_review_P127jhj-8-Y.json in data/manual_reviews/

# Step 3: Process your reviews
python evaluator.py P127jhj-8-Y --review-file data/manual_reviews/manual_review_P127jhj-8-Y.json
```

### **Batch Processing**
```bash
# Process multiple videos
for video_id in P127jhj-8-Y dQw4w9WgXcQ another_video_id; do
    python evaluator.py $video_id --dashboard
done
```

This framework provides a complete solution for maintaining high-quality LLM-generated video chapters through systematic evaluation and continuous improvement.