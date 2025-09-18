#!/usr/bin/env python3
"""
Chapter Quality Evaluation Framework

This module provides comprehensive evaluation capabilities for LLM-generated video chapters,
focusing on quality, relevance, alignment, and user utility metrics.
"""

import json
import re
import math
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import torch
from bert_score import score as bert_score
from rouge_score import rouge_scorer

@dataclass
class EvaluationMetrics:
    """Comprehensive evaluation metrics for chapter quality."""
    # Content Quality Metrics
    content_relevance: float  # 0-1, how well chapter content matches transcript
    title_accuracy: float     # 0-1, how accurately title describes content
    summary_completeness: float  # 0-1, how complete the summary is
    
    # Advanced Text Quality Metrics
    bert_score_precision: float    # BERTScore precision (0-1)
    bert_score_recall: float       # BERTScore recall (0-1)
    bert_score_f1: float           # BERTScore F1 (0-1)
    rouge_1_f1: float              # ROUGE-1 F1 score (0-1)
    rouge_2_f1: float              # ROUGE-2 F1 score (0-1)
    rouge_l_f1: float              # ROUGE-L F1 score (0-1)
    
    # Structural Quality Metrics
    boundary_accuracy: float  # 0-1, how well chapter boundaries align with content
    temporal_consistency: float  # 0-1, logical flow of timestamps
    duration_appropriateness: float  # 0-1, appropriate chapter length
    
    # Uniqueness Metrics
    redundancy_score: float   # 0-1, lower is better (less redundant)
    distinctiveness: float    # 0-1, how unique this chapter is
    
    # Search Utility Metrics
    search_relevance: float   # 0-1, how well chapter serves search queries
    keyword_coverage: float   # 0-1, coverage of important keywords
    navigation_utility: float # 0-1, usefulness for user navigation
    
    # Overall Quality Score
    overall_score: float      # 0-1, weighted combination of all metrics
    
    # Metadata
    evaluation_timestamp: str
    chapter_index: int
    confidence_score: float

@dataclass
class UserQuery:
    """User search query with feedback."""
    query: str
    user_feedback: Optional[Dict[str, Any]] = None  # {chapter_id: relevance_score, navigation_utility_score}
    timestamp: Optional[str] = None

@dataclass
class SearchEvaluation:
    """Search relevance evaluation results."""
    query: str
    chapter_relevance_scores: Dict[int, float]  # chapter_index -> relevance_score
    navigation_utility_scores: Dict[int, float]  # chapter_index -> navigation_score
    user_feedback: Optional[Dict[int, Dict[str, float]]] = None

@dataclass
class ManualReview:
    """Manual review data structure."""
    reviewer_id: str
    chapter_index: int
    overall_quality_score: float  # 1-5 scale
    content_accuracy_score: float  # 1-5 scale
    title_appropriateness_score: float  # 1-5 scale
    summary_quality_score: float  # 1-5 scale
    search_relevance_score: float  # 1-5 scale
    navigation_utility_score: float  # 1-5 scale
    issues_identified: List[str]
    recommendations: List[str]
    review_timestamp: str
    confidence_level: float  # 0-1, reviewer confidence in their assessment

@dataclass
class StandardizedScoringGuide:
    """Standardized scoring criteria for manual reviews."""
    content_accuracy_criteria: Dict[str, str]
    title_appropriateness_criteria: Dict[str, str]
    summary_quality_criteria: Dict[str, str]
    search_relevance_criteria: Dict[str, str]
    navigation_utility_criteria: Dict[str, str]
    common_llm_errors: Dict[str, str]

@dataclass
class LLMErrorAnalysis:
    """Analysis of common LLM errors and issues."""
    hallucination_score: float  # 0-1, likelihood of fabricated content
    factual_consistency_score: float  # 0-1, consistency with transcript
    coherence_score: float  # 0-1, logical flow and structure
    relevance_score: float  # 0-1, relevance to user query
    completeness_score: float  # 0-1, completeness of information
    bias_score: float  # 0-1, potential bias in content
    detected_errors: List[str]
    error_severity: Dict[str, str]  # error_type -> severity_level

@dataclass
class ChapterAnalysis:
    """Detailed analysis of a single chapter."""
    chapter_data: Dict[str, Any]
    transcript_segment: str
    evaluation_metrics: EvaluationMetrics
    issues_detected: List[str]
    recommendations: List[str]
    semantic_keywords: List[str]
    content_themes: List[str]
    search_evaluations: List[SearchEvaluation] = None
    manual_reviews: List[ManualReview] = None
    llm_error_analysis: Optional[LLMErrorAnalysis] = None
    quality_trends: Dict[str, List[float]] = None  # metric -> historical_scores

class ChapterEvaluator:
    """Main evaluation class for chapter quality assessment."""
    
    def __init__(self, openai_api_key: Optional[str] = None, embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the chapter evaluator."""
        self.client = OpenAI(api_key=openai_api_key)
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Initialize advanced embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embedding_model.to(self.device)
        
        # Initialize ROUGE scorer
        print("Loading ROUGE scorer...")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # User queries for evaluation
        self.user_queries: List[UserQuery] = []
        
        # Initialize standardized scoring guide
        self.scoring_guide = self._initialize_standardized_scoring_guide()
    
    def _initialize_standardized_scoring_guide(self) -> StandardizedScoringGuide:
        """Initialize standardized scoring criteria for manual reviews."""
        return StandardizedScoringGuide(
            content_accuracy_criteria={
                "5": "Perfect alignment with transcript content, no factual errors",
                "4": "High alignment with minor inaccuracies or omissions",
                "3": "Good alignment with some inaccuracies or missing key points",
                "2": "Moderate alignment with significant inaccuracies",
                "1": "Poor alignment, major factual errors or fabrications"
            },
            title_appropriateness_criteria={
                "5": "Title perfectly captures chapter content and is engaging",
                "4": "Title accurately describes content, minor clarity issues",
                "3": "Title generally appropriate with some ambiguity",
                "2": "Title partially accurate but misleading or unclear",
                "1": "Title inaccurate or completely misleading"
            },
            summary_quality_criteria={
                "5": "Comprehensive, coherent summary covering all key points",
                "4": "Good summary with minor gaps or clarity issues",
                "3": "Adequate summary missing some important details",
                "2": "Incomplete summary with significant gaps",
                "1": "Poor summary, missing critical information"
            },
            search_relevance_criteria={
                "5": "Highly relevant to user queries, excellent discoverability",
                "4": "Good relevance with minor gaps in query coverage",
                "3": "Moderately relevant, covers main query aspects",
                "2": "Limited relevance, misses key query elements",
                "1": "Poor relevance, doesn't serve user search needs"
            },
            navigation_utility_criteria={
                "5": "Excellent navigation aid, clear timestamps and structure",
                "4": "Good navigation with minor usability issues",
                "3": "Adequate navigation, some clarity improvements needed",
                "2": "Limited navigation utility, usability concerns",
                "1": "Poor navigation aid, confusing or unhelpful"
            },
            common_llm_errors={
                "hallucination": "Fabricated information not present in source",
                "factual_inconsistency": "Contradictory information within content",
                "temporal_misalignment": "Incorrect timestamps or content placement",
                "redundancy": "Repetitive or overlapping content",
                "incompleteness": "Missing key information or context",
                "bias": "Inappropriate bias or subjective content",
                "coherence": "Poor logical flow or structure",
                "relevance": "Content not relevant to user query or topic"
            }
        )
    
    def add_user_query(self, query: str, feedback: Dict[str, Any] = None) -> None:
        """Add a user query for search relevance evaluation."""
        user_query = UserQuery(query=query, user_feedback=feedback)
        self.user_queries.append(user_query)
    
    def generate_sample_queries(self, transcript_data: List[Dict], num_queries: int = 5) -> List[str]:
        """Generate sample search queries based on transcript content."""
        try:
            # Extract key topics and concepts from transcript
            full_transcript = ' '.join([item['text'] for item in transcript_data])
            
            # Use OpenAI to generate realistic search queries
            prompt = f"""
            Based on this video transcript, generate {num_queries} realistic search queries that users might use to find relevant content in this video.
            
            Transcript excerpt: {full_transcript[:2000]}...
            
            Generate queries that cover different aspects like:
            - Specific topics or concepts
            - How-to questions
            - Problem-solving queries
            - General topic searches
            
            Return only the queries, one per line, without numbering or explanation.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at understanding user search behavior. Generate realistic search queries that users would use to find video content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            queries = [q.strip() for q in response.choices[0].message.content.strip().split('\n') if q.strip()]
            return queries[:num_queries]
            
        except Exception as e:
            print(f"Error generating sample queries: {e}")
            # Fallback to simple keyword-based queries
            words = re.findall(r'\b\w{4,}\b', full_transcript.lower())
            word_freq = defaultdict(int)
            for word in words:
                word_freq[word] += 1
            
            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            return [f"what is {word}" for word, _ in top_words[:num_queries]]
    
    def evaluate_search_relevance(self, 
                                 query: str, 
                                 chapters: List[Dict], 
                                 transcript_data: List[Dict]) -> SearchEvaluation:
        """Evaluate search relevance for a specific query."""
        
        # Get embeddings for query and chapter content
        query_embedding = self.embedding_model.encode([query])
        
        relevance_scores = {}
        navigation_scores = {}
        
        for i, chapter in enumerate(chapters):
            # Combine chapter title and summary for embedding
            chapter_text = f"{chapter['title']} {chapter['summary']}"
            chapter_embedding = self.embedding_model.encode([chapter_text])
            
            # Calculate semantic similarity using embeddings
            similarity = cosine_similarity(query_embedding, chapter_embedding)[0][0]
            relevance_scores[i] = float(similarity)
            
            # Calculate navigation utility based on timestamp accuracy
            navigation_scores[i] = self._calculate_navigation_utility_for_query(
                query, chapter, transcript_data
            )
        
        return SearchEvaluation(
            query=query,
            chapter_relevance_scores=relevance_scores,
            navigation_utility_scores=navigation_scores
        )
    
    def _calculate_navigation_utility_for_query(self, 
                                               query: str, 
                                               chapter: Dict, 
                                               transcript_data: List[Dict]) -> float:
        """Navigation utility: does the chapter help a user with this query?
        Defined as embedding similarity between the user query and:
        (1) the chapter's transcript segment, (2) the generated summary.
        Returns a 0..1 score as the average of the two cosine similarities.
        """
        # Get transcript segment for this chapter
        transcript_segment = self._get_transcript_segment(chapter, transcript_data)
        
        try:
            embeddings = self.embedding_model.encode([query, transcript_segment, chapter['summary']])
            query_vec = embeddings[0]
            transcript_vec = embeddings[1]
            summary_vec = embeddings[2]
            sim_q_tr = cosine_similarity([query_vec], [transcript_vec])[0][0]
            sim_q_sum = cosine_similarity([query_vec], [summary_vec])[0][0]
            return float(max(0.0, min(1.0, (sim_q_tr + sim_q_sum) / 2.0)))
        except Exception as e:
            print(f"Error calculating navigation utility (embeddings): {e}")
            return 0.0
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using embeddings."""
        try:
            embeddings = self.embedding_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error in semantic similarity calculation: {e}")
            return 0.0
    
    def _calculate_bert_score(self, candidate: str, reference: str) -> Dict[str, float]:
        """Calculate BERTScore between candidate and reference text."""
        try:
            # BERTScore expects lists of strings
            P, R, F1 = bert_score([candidate], [reference], lang="en", verbose=False)
            return {
                'precision': float(P.item()),
                'recall': float(R.item()),
                'f1': float(F1.item())
            }
        except Exception as e:
            print(f"Error calculating BERTScore: {e}")
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    
    def _calculate_rouge_scores(self, candidate: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores between candidate and reference text."""
        try:
            scores = self.rouge_scorer.score(reference, candidate)
            return {
                'rouge_1_f1': scores['rouge1'].fmeasure,
                'rouge_2_f1': scores['rouge2'].fmeasure,
                'rouge_l_f1': scores['rougeL'].fmeasure
            }
        except Exception as e:
            print(f"Error calculating ROUGE scores: {e}")
            return {'rouge_1_f1': 0.0, 'rouge_2_f1': 0.0, 'rouge_l_f1': 0.0}
    
    def analyze_llm_errors(self, 
                          chapter: Dict, 
                          transcript_segment: str, 
                          user_queries: List[str]) -> LLMErrorAnalysis:
        """Analyze common LLM errors and issues."""
        
        # Calculate hallucination score (content not in transcript)
        hallucination_score = self._calculate_hallucination_score(chapter, transcript_segment)
        
        # Calculate factual consistency score
        factual_consistency_score = self._calculate_factual_consistency(chapter, transcript_segment)
        
        # Calculate coherence score
        coherence_score = self._calculate_coherence_score(chapter)
        
        # Calculate relevance score based on user queries
        relevance_score = self._calculate_query_relevance(chapter, user_queries)
        
        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(chapter, transcript_segment)
        
        # Calculate bias score
        bias_score = self._calculate_bias_score(chapter)
        
        # Detect specific errors
        detected_errors = self._detect_specific_errors(chapter, transcript_segment)
        
        # Assess error severity
        error_severity = self._assess_error_severity(detected_errors)
        
        return LLMErrorAnalysis(
            hallucination_score=hallucination_score,
            factual_consistency_score=factual_consistency_score,
            coherence_score=coherence_score,
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            bias_score=bias_score,
            detected_errors=detected_errors,
            error_severity=error_severity
        )
    
    def _calculate_hallucination_score(self, chapter: Dict, transcript_segment: str) -> float:
        """Calculate likelihood of fabricated content."""
        try:
            # Use BERTScore to detect content not present in transcript
            summary = chapter['summary']
            title = chapter['title']
            
            # Calculate similarity between chapter content and transcript
            summary_similarity = self._calculate_semantic_similarity(summary, transcript_segment)
            title_similarity = self._calculate_semantic_similarity(title, transcript_segment)
            
            # Lower similarity indicates potential hallucination
            avg_similarity = (summary_similarity + title_similarity) / 2
            hallucination_score = 1.0 - avg_similarity  # Inverse of similarity
            
            return float(min(1.0, max(0.0, hallucination_score)))
        except:
            return 0.5  # Default neutral score
    
    def _calculate_factual_consistency(self, chapter: Dict, transcript_segment: str) -> float:
        """Calculate consistency with transcript facts."""
        try:
            # Check for contradictory statements
            summary = chapter['summary']
            
            # Use BERTScore for factual consistency
            bert_scores = self._calculate_bert_score(summary, transcript_segment)
            consistency_score = bert_scores['f1']  # F1 score indicates consistency
            
            return float(consistency_score)
        except:
            return 0.5
    
    def _calculate_coherence_score(self, chapter: Dict) -> float:
        """Calculate logical flow and structure."""
        try:
            summary = chapter['summary']
            
            # Check for logical flow indicators
            coherence_indicators = [
                'first', 'second', 'then', 'next', 'finally', 'however', 'therefore',
                'because', 'although', 'meanwhile', 'consequently'
            ]
            
            # Count coherence indicators
            indicator_count = sum(1 for indicator in coherence_indicators if indicator in summary.lower())
            
            # Check sentence structure
            sentences = summary.split('.')
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
            
            # Score based on indicators and sentence structure
            coherence_score = min(1.0, (indicator_count * 0.1 + min(1.0, avg_sentence_length / 20)))
            
            return float(coherence_score)
        except:
            return 0.5
    
    def _calculate_query_relevance(self, chapter: Dict, user_queries: List[str]) -> float:
        """Calculate relevance to user queries."""
        if not user_queries:
            return 0.5
        
        relevance_scores = []
        chapter_text = f"{chapter['title']} {chapter['summary']}"
        
        for query in user_queries:
            similarity = self._calculate_semantic_similarity(query, chapter_text)
            relevance_scores.append(similarity)
        
        return float(np.mean(relevance_scores))
    
    def _calculate_completeness_score(self, chapter: Dict, transcript_segment: str) -> float:
        """Calculate completeness of information."""
        try:
            summary = chapter['summary']
            
            # Check if summary covers key aspects of transcript
            transcript_length = len(transcript_segment.split())
            summary_length = len(summary.split())
            
            # Ideal summary length ratio
            ideal_ratio = 0.15  # 15% of original length
            actual_ratio = summary_length / transcript_length if transcript_length > 0 else 0
            
            # Score based on how close to ideal ratio
            completeness_score = 1.0 - abs(actual_ratio - ideal_ratio) / ideal_ratio
            
            return float(min(1.0, max(0.0, completeness_score)))
        except:
            return 0.5
    
    def _calculate_bias_score(self, chapter: Dict) -> float:
        """Calculate potential bias in content."""
        try:
            summary = chapter['summary'].lower()
            title = chapter['title'].lower()
            
            # Check for bias indicators
            bias_indicators = [
                'obviously', 'clearly', 'undoubtedly', 'definitely', 'absolutely',
                'everyone knows', 'nobody would', 'all experts agree'
            ]
            
            bias_count = sum(1 for indicator in bias_indicators if indicator in summary or indicator in title)
            
            # Higher bias count indicates more bias
            bias_score = min(1.0, bias_count * 0.2)
            
            return float(bias_score)
        except:
            return 0.0
    
    def _detect_specific_errors(self, chapter: Dict, transcript_segment: str) -> List[str]:
        """Detect specific types of errors."""
        errors = []
        
        # Check for temporal misalignment
        if self._check_temporal_misalignment(chapter, transcript_segment):
            errors.append("temporal_misalignment")
        
        # Check for redundancy
        if self._check_redundancy(chapter):
            errors.append("redundancy")
        
        # Check for incompleteness
        if self._check_incompleteness(chapter, transcript_segment):
            errors.append("incompleteness")
        
        # Check for coherence issues
        if self._check_coherence_issues(chapter):
            errors.append("coherence")
        
        return errors
    
    def _check_temporal_misalignment(self, chapter: Dict, transcript_segment: str) -> bool:
        """Check if timestamps align with content."""
        # This is a simplified check - in practice, you'd do more sophisticated analysis
        start_time = chapter.get('start_time', 0)
        end_time = chapter.get('end_time', 0)
        
        # Check if duration is reasonable
        duration = end_time - start_time
        if duration < 10 or duration > 1800:  # Too short or too long
            return True
        
        return False
    
    def _check_redundancy(self, chapter: Dict) -> bool:
        """Check for redundant content."""
        summary = chapter['summary']
        title = chapter['title']
        
        # Check if title and summary are too similar
        similarity = self._calculate_semantic_similarity(title, summary)
        if similarity > 0.8:
            return True
        
        return False
    
    def _check_incompleteness(self, chapter: Dict, transcript_segment: str) -> bool:
        """Check for incomplete information."""
        summary = chapter['summary']
        
        # Check if summary is too short relative to content
        if len(summary.split()) < 20:
            return True
        
        return False
    
    def _check_coherence_issues(self, chapter: Dict) -> bool:
        """Check for coherence issues."""
        summary = chapter['summary']
        
        # Check for sentence structure issues
        sentences = summary.split('.')
        if len(sentences) < 2:
            return True
        
        return False
    
    def _assess_error_severity(self, errors: List[str]) -> Dict[str, str]:
        """Assess severity of detected errors."""
        severity_levels = {
            "hallucination": "high",
            "factual_inconsistency": "high",
            "temporal_misalignment": "medium",
            "redundancy": "low",
            "incompleteness": "medium",
            "bias": "medium",
            "coherence": "low",
            "relevance": "medium"
        }
        
        return {error: severity_levels.get(error, "low") for error in errors}
    
    def add_manual_review(self, review: ManualReview) -> None:
        """Add a manual review to the evaluation system."""
        # Validate review data
        if not (1 <= review.overall_quality_score <= 5):
            raise ValueError("Overall quality score must be between 1 and 5")
        
        # Validate all scores are in range
        scores = [
            review.content_accuracy_score,
            review.title_appropriateness_score,
            review.summary_quality_score,
            review.search_relevance_score,
            review.navigation_utility_score
        ]
        
        for score in scores:
            if not (1 <= score <= 5):
                raise ValueError("All scores must be between 1 and 5")
        
        # Validate confidence level
        if not (0 <= review.confidence_level <= 1):
            raise ValueError("Confidence level must be between 0 and 1")
        
        # In a real implementation, this would store in database
        print(f"Manual review added for chapter {review.chapter_index} by reviewer {review.reviewer_id}")
        print(f"Overall quality: {review.overall_quality_score}/5")
        print(f"Confidence level: {review.confidence_level:.2f}")
    
    def create_manual_review_template(self, chapter_index: int, reviewer_id: str) -> Dict[str, Any]:
        """Create a template for manual review input."""
        return {
            "reviewer_id": reviewer_id,
            "chapter_index": chapter_index,
            "overall_quality_score": None,  # 1-5
            "content_accuracy_score": None,  # 1-5
            "title_appropriateness_score": None,  # 1-5
            "summary_quality_score": None,  # 1-5
            "search_relevance_score": None,  # 1-5
            "navigation_utility_score": None,  # 1-5
            "issues_identified": [],  # List of strings
            "recommendations": [],  # List of strings
            "review_timestamp": None,  # Will be auto-generated
            "confidence_level": None  # 0-1
        }
    
    def process_manual_review(self, review_data: Dict[str, Any]) -> ManualReview:
        """Process manual review data and create ManualReview object."""
        from datetime import datetime
        
        # Auto-generate timestamp if not provided
        if not review_data.get("review_timestamp"):
            review_data["review_timestamp"] = datetime.now().isoformat()
        
        # Create ManualReview object
        review = ManualReview(
            reviewer_id=review_data["reviewer_id"],
            chapter_index=review_data["chapter_index"],
            overall_quality_score=review_data["overall_quality_score"],
            content_accuracy_score=review_data["content_accuracy_score"],
            title_appropriateness_score=review_data["title_appropriateness_score"],
            summary_quality_score=review_data["summary_quality_score"],
            search_relevance_score=review_data["search_relevance_score"],
            navigation_utility_score=review_data["navigation_utility_score"],
            issues_identified=review_data["issues_identified"],
            recommendations=review_data["recommendations"],
            review_timestamp=review_data["review_timestamp"],
            confidence_level=review_data["confidence_level"]
        )
        
        return review
    
    def calculate_inter_reviewer_agreement(self, reviews: List[ManualReview]) -> float:
        """Calculate inter-reviewer agreement score."""
        if len(reviews) < 2:
            return 1.0
        
        # Calculate agreement on overall quality scores
        scores = [review.overall_quality_score for review in reviews]
        
        # Simple agreement calculation (can be enhanced with more sophisticated methods)
        score_variance = np.var(scores)
        agreement_score = max(0.0, 1.0 - score_variance / 4.0)  # Normalize by max variance
        
        return float(agreement_score)
        
    def evaluate_chapters(self, 
                         chapters: List[Dict], 
                         transcript_data: List[Dict],
                         video_metadata: Dict = None,
                         user_queries: List[str] = None) -> List[ChapterAnalysis]:
        """
        Comprehensive evaluation of all chapters.
        
        Args:
            chapters: List of chapter dictionaries
            transcript_data: Original transcript data
            video_metadata: Optional video metadata (title, description, etc.)
            user_queries: Optional list of user search queries for evaluation
            
        Returns:
            List of ChapterAnalysis objects with detailed evaluations
        """
        print("Starting comprehensive chapter evaluation...")
        
        # Step 1: Generate or use user queries for search evaluation
        print("Step 1: Setting up search evaluation queries...")
        if user_queries is None:
            print("Generating sample user queries...")
            user_queries = self.generate_sample_queries(transcript_data, num_queries=5)
        
        print(f"Using {len(user_queries)} queries for evaluation: {user_queries}")
        
        # Step 2: Content Quality Analysis
        print("Step 2: Analyzing content quality...")
        content_analyses = self._analyze_content_quality(chapters, transcript_data)
        
        # Step 3: Structural Analysis
        print("Step 3: Analyzing structural quality...")
        structural_analyses = self._analyze_structural_quality(chapters, transcript_data)
        
        # Step 4: Redundancy Detection
        print("Step 4: Detecting redundancy and overlaps...")
        redundancy_analyses = self._detect_redundancy(chapters)
        
        # Step 5: Search Utility Analysis with user queries
        print("Step 5: Analyzing search utility with user queries...")
        search_analyses = self._analyze_search_utility_with_queries(chapters, transcript_data, user_queries)
        
        # Step 6: LLM Error Analysis
        print("Step 6: Analyzing LLM errors and issues...")
        llm_error_analyses = self._analyze_llm_errors_comprehensive(chapters, transcript_data, user_queries)
        
        # Step 7: Combine all analyses
        print("Step 7: Combining analyses into final evaluation...")
        final_analyses = self._combine_analyses(
            chapters, transcript_data, content_analyses, 
            structural_analyses, redundancy_analyses, search_analyses, 
            user_queries, llm_error_analyses
        )
        
        return final_analyses
    
    def _analyze_content_quality(self, 
                                chapters: List[Dict], 
                                transcript_data: List[Dict]) -> List[Dict]:
        """Analyze content quality metrics for each chapter."""
        analyses = []
        
        for i, chapter in enumerate(chapters):
            # Get relevant transcript segment
            transcript_segment = self._get_transcript_segment(
                chapter, transcript_data
            )
            
            # Content relevance analysis
            content_relevance = self._calculate_content_relevance(
                chapter['summary'], transcript_segment
            )
            
            # Title accuracy analysis (context similarity to transcript and summary)
            title_accuracy = self._calculate_title_accuracy(
                chapter['title'], transcript_segment, chapter['summary']
            )
            
            # Summary completeness analysis
            summary_completeness = self._calculate_summary_completeness(
                chapter['summary'], transcript_segment
            )
            
            # Calculate BERTScore between summary and transcript segment
            bert_scores = self._calculate_bert_score(chapter['summary'], transcript_segment)
            
            # Calculate ROUGE scores between summary and transcript segment
            rouge_scores = self._calculate_rouge_scores(chapter['summary'], transcript_segment)
            
            analyses.append({
                'chapter_index': i,
                'content_relevance': content_relevance,
                'title_accuracy': title_accuracy,
                'summary_completeness': summary_completeness,
                'transcript_segment': transcript_segment,
                'bert_scores': bert_scores,
                'rouge_scores': rouge_scores
            })
        
        return analyses
    
    def _analyze_structural_quality(self, 
                                  chapters: List[Dict], 
                                  transcript_data: List[Dict]) -> List[Dict]:
        """Analyze structural quality metrics."""
        analyses = []
        
        for i, chapter in enumerate(chapters):
            # Boundary accuracy
            boundary_accuracy = self._calculate_boundary_accuracy(
                chapter, transcript_data
            )
            
            # Temporal consistency
            temporal_consistency = self._calculate_temporal_consistency(
                chapter, chapters
            )
            
            # Duration appropriateness
            duration_appropriateness = self._calculate_duration_appropriateness(
                chapter
            )
            
            analyses.append({
                'chapter_index': i,
                'boundary_accuracy': boundary_accuracy,
                'temporal_consistency': temporal_consistency,
                'duration_appropriateness': duration_appropriateness
            })
        
        return analyses
    
    def _detect_redundancy(self, chapters: List[Dict]) -> List[Dict]:
        """Detect redundant and overlapping chapters using advanced embeddings."""
        analyses = []
        
        # Extract text content for similarity analysis
        chapter_texts = []
        for chapter in chapters:
            text = f"{chapter['title']} {chapter['summary']}"
            chapter_texts.append(text)
        
        # Calculate similarity matrix using sentence transformers
        if len(chapter_texts) > 1:
            try:
                # Use advanced embeddings for better semantic similarity
                embeddings = self.embedding_model.encode(chapter_texts)
                similarity_matrix = cosine_similarity(embeddings)
            except Exception as e:
                print(f"Error with embeddings, falling back to TF-IDF: {e}")
                # Fallback to TF-IDF
                tfidf_matrix = self.vectorizer.fit_transform(chapter_texts)
                similarity_matrix = cosine_similarity(tfidf_matrix)
        else:
            similarity_matrix = np.array([[1.0]])
        
        for i, chapter in enumerate(chapters):
            # Calculate redundancy score (average similarity to other chapters)
            if len(chapters) > 1:
                similarities = [similarity_matrix[i][j] for j in range(len(chapters)) if j != i]
                redundancy_score = float(np.mean(similarities))
            else:
                redundancy_score = 0.0
            
            # Calculate distinctiveness (inverse of redundancy)
            distinctiveness = 1.0 - redundancy_score
            
            # Detect overlapping content with lower threshold for embeddings
            overlap_threshold = 0.6  # Lower threshold for semantic similarity
            overlapping_chapters = []
            for j, other_chapter in enumerate(chapters):
                if i != j and similarity_matrix[i][j] > overlap_threshold:
                    overlapping_chapters.append({
                        'chapter_index': j,
                        'similarity': float(similarity_matrix[i][j]),
                        'title': other_chapter['title']
                    })
            
            analyses.append({
                'chapter_index': i,
                'redundancy_score': redundancy_score,
                'distinctiveness': distinctiveness,
                'overlapping_chapters': overlapping_chapters
            })
        
        return analyses
    
    def _analyze_search_utility_with_queries(self, 
                                           chapters: List[Dict], 
                                           transcript_data: List[Dict],
                                           user_queries: List[str]) -> List[Dict]:
        """Analyze search utility and navigation value using user queries."""
        analyses = []
        
        # Extract key terms from transcript for search relevance
        full_transcript = ' '.join([item['text'] for item in transcript_data])
        
        # Calculate average scores across all user queries
        all_search_relevances = []
        all_navigation_utilities = []
        
        for i, chapter in enumerate(chapters):
            chapter_search_scores = []
            chapter_navigation_scores = []
            
            # Evaluate chapter against each user query
            for query in user_queries:
                search_eval = self.evaluate_search_relevance(query, [chapter], transcript_data)
                chapter_search_scores.append(search_eval.chapter_relevance_scores.get(0, 0.0))
                chapter_navigation_scores.append(search_eval.navigation_utility_scores.get(0, 0.0))
            
            # Average scores across queries (convert to Python float)
            avg_search_relevance = float(np.mean(chapter_search_scores)) if chapter_search_scores else 0.0
            avg_navigation_utility = float(np.mean(chapter_navigation_scores)) if chapter_navigation_scores else 0.0
            
            # Traditional keyword coverage (kept for comparison)
            keyword_coverage = self._calculate_keyword_coverage(chapter, full_transcript)
            
            analyses.append({
                'chapter_index': i,
                'search_relevance': avg_search_relevance,
                'keyword_coverage': keyword_coverage,
                'navigation_utility': avg_navigation_utility,
                'query_scores': {
                    'search_relevances': chapter_search_scores,
                    'navigation_utilities': chapter_navigation_scores
                }
            })
        
        return analyses
    
    def _analyze_llm_errors_comprehensive(self, 
                                        chapters: List[Dict], 
                                        transcript_data: List[Dict],
                                        user_queries: List[str]) -> List[LLMErrorAnalysis]:
        """Comprehensive LLM error analysis for all chapters."""
        error_analyses = []
        
        for i, chapter in enumerate(chapters):
            # Get transcript segment for this chapter
            transcript_segment = self._get_transcript_segment(chapter, transcript_data)
            
            # Analyze LLM errors
            error_analysis = self.analyze_llm_errors(chapter, transcript_segment, user_queries)
            error_analyses.append(error_analysis)
        
        return error_analyses
    
    def _combine_analyses(self, 
                         chapters: List[Dict],
                         transcript_data: List[Dict],
                         content_analyses: List[Dict],
                         structural_analyses: List[Dict],
                         redundancy_analyses: List[Dict],
                         search_analyses: List[Dict],
                         user_queries: List[str],
                         llm_error_analyses: List[LLMErrorAnalysis]) -> List[ChapterAnalysis]:
        """Combine all analyses into final ChapterAnalysis objects."""
        final_analyses = []
        
        for i, chapter in enumerate(chapters):
            # Get corresponding analyses
            content = content_analyses[i]
            structural = structural_analyses[i]
            redundancy = redundancy_analyses[i]
            search = search_analyses[i]
            llm_errors = llm_error_analyses[i]
            
            # Create evaluation metrics
            metrics = EvaluationMetrics(
                content_relevance=content['content_relevance'],
                title_accuracy=content['title_accuracy'],
                summary_completeness=content['summary_completeness'],
                bert_score_precision=content['bert_scores']['precision'],
                bert_score_recall=content['bert_scores']['recall'],
                bert_score_f1=content['bert_scores']['f1'],
                rouge_1_f1=content['rouge_scores']['rouge_1_f1'],
                rouge_2_f1=content['rouge_scores']['rouge_2_f1'],
                rouge_l_f1=content['rouge_scores']['rouge_l_f1'],
                boundary_accuracy=structural['boundary_accuracy'],
                temporal_consistency=structural['temporal_consistency'],
                duration_appropriateness=structural['duration_appropriateness'],
                redundancy_score=redundancy['redundancy_score'],
                distinctiveness=redundancy['distinctiveness'],
                search_relevance=search['search_relevance'],
                keyword_coverage=search['keyword_coverage'],
                navigation_utility=search['navigation_utility'],
                overall_score=0.0,  # Will be calculated below
                evaluation_timestamp="",
                chapter_index=i,
                confidence_score=0.0
            )
            
            # Calculate overall score (weighted average)
            weights = {
                'content_relevance': 0.15,
                'title_accuracy': 0.10,
                'summary_completeness': 0.10,
                'bert_score_f1': 0.15,        # BERTScore F1 as primary quality metric
                'rouge_l_f1': 0.10,           # ROUGE-L F1 for summary quality
                'boundary_accuracy': 0.10,
                'temporal_consistency': 0.08,
                'duration_appropriateness': 0.05,
                'distinctiveness': 0.10,
                'search_relevance': 0.07
            }
            
            overall_score = sum(
                getattr(metrics, metric) * weight 
                for metric, weight in weights.items()
            )
            metrics.overall_score = overall_score
            
            # Detect issues
            issues = self._detect_issues(metrics, redundancy['overlapping_chapters'], llm_errors)
            # Generate query-based chapter recommendations (per user queries)
            # using the search evaluations prepared below
            # (Recommend this chapter for queries where both relevance and navigation are strong)
            recommendations = []
            
            # Extract semantic information
            semantic_keywords = self._extract_semantic_keywords(chapter)
            content_themes = self._extract_content_themes(chapter)
            
            # Create search evaluations for this chapter
            search_evaluations = []
            for query in user_queries:
                search_eval = self.evaluate_search_relevance(query, [chapter], transcript_data)
                search_evaluations.append(search_eval)

            # Build recommendations from search evaluations
            recommendations = self._generate_query_recommendations(search_evaluations, chapter)
            
            # Create final analysis
            analysis = ChapterAnalysis(
                chapter_data=chapter,
                transcript_segment=content['transcript_segment'],
                evaluation_metrics=metrics,
                issues_detected=issues,
                recommendations=recommendations,
                semantic_keywords=semantic_keywords,
                content_themes=content_themes,
                search_evaluations=search_evaluations,
                manual_reviews=None,  # Will be populated when manual reviews are added
                llm_error_analysis=llm_errors,
                quality_trends=None  # Will be populated with historical data
            )
            
            final_analyses.append(analysis)
        
        return final_analyses
    
    # Helper methods for individual metric calculations
    def _get_transcript_segment(self, chapter: Dict, transcript_data: List[Dict]) -> str:
        """Extract the transcript segment corresponding to a chapter."""
        start_time = chapter['start_time']
        end_time = chapter['end_time']
        
        segment_texts = []
        for item in transcript_data:
            if start_time <= item['start'] <= end_time:
                segment_texts.append(item['text'])
        
        return ' '.join(segment_texts)
    
    def _calculate_content_relevance(self, summary: str, transcript_segment: str) -> float:
        """Calculate how well the summary matches the transcript content using advanced embeddings."""
        try:
            # Use sentence transformers for better semantic similarity
            embeddings = self.embedding_model.encode([summary, transcript_segment])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error calculating content relevance: {e}")
            # Fallback to TF-IDF
            try:
                texts = [summary, transcript_segment]
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(similarity)
            except:
                return 0.5  # Default neutral score
    
    def _calculate_title_accuracy(self, title: str, transcript_segment: str, chapter_summary: str) -> float:
        """Calculate title accuracy via semantic similarity to transcript and summary.
        Uses sentence-transformer embeddings; falls back to TF-IDF similarity on error.
        """
        try:
            # Compute embeddings for title, transcript, and summary
            embeddings = self.embedding_model.encode([title, transcript_segment, chapter_summary])
            title_vec, transcript_vec, summary_vec = embeddings[0], embeddings[1], embeddings[2]
            sim_title_transcript = cosine_similarity([title_vec], [transcript_vec])[0][0]
            sim_title_summary = cosine_similarity([title_vec], [summary_vec])[0][0]
            # Weighted combination prioritizing transcript context
            combined = 0.6 * sim_title_transcript + 0.4 * sim_title_summary
            return float(max(0.0, min(1.0, combined)))
        except Exception as e:
            print(f"Error calculating title accuracy: {e}")
            print("Fallback to TF-IDF similarity if embeddings fail")  
            try:
                texts = [title, transcript_segment, chapter_summary]
                tfidf_matrix = self.vectorizer.fit_transform(texts)
                sim_title_transcript = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                sim_title_summary = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[2:3])[0][0]
                combined = 0.6 * sim_title_transcript + 0.4 * sim_title_summary
                return float(max(0.0, min(1.0, combined)))
            except:
                return 0.0
    
    def _calculate_summary_completeness(self, summary: str, transcript_segment: str) -> float:
        """Assess summary completeness by semantic coverage, not just length.
        Primary signal: embedding-based semantic similarity between summary and transcript segment.
        Secondary signal: keyword coverage from transcript in summary.
        Returns a 0..1 score.
        """
        if not transcript_segment.strip() or not summary.strip():
            return 0.0
        
        # 1) Semantic similarity (embeddings), fallback to TF-IDF cosine
        try:
            emb = self.embedding_model.encode([summary, transcript_segment])
            sim_semantic = cosine_similarity([emb[0]], [emb[1]])[0][0]
        except Exception as e:
            print(f"Error calculating summary completeness (embeddings): {e}")
            try:
                tfidf = self.vectorizer.fit_transform([summary, transcript_segment])
                sim_semantic = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
            except Exception as e2:
                print(f"Fallback TF-IDF failed for completeness: {e2}")
                sim_semantic = 0.0
        
        # 2) Lexical diversity alignment instead of hardcoded keyword lists
        def _clean_tokens(text: str) -> List[str]:
            return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        tr_tokens = _clean_tokens(transcript_segment)
        sm_tokens = _clean_tokens(summary)
        coverage = 0.0
        if tr_tokens and sm_tokens:
            tr_types = set(tr_tokens)
            sm_types = set(sm_tokens)
            # Type-Token Ratio (TTR)
            tr_ttr = len(tr_types) / len(tr_tokens)
            sm_ttr = len(sm_types) / len(sm_tokens)
            diversity_ratio = (sm_ttr / tr_ttr) if tr_ttr > 0 else 0.0
            diversity_ratio = float(max(0.0, min(1.0, diversity_ratio)))
            # Lexical overlap of unique types
            overlap = len(tr_types.intersection(sm_types)) / len(tr_types) if tr_types else 0.0
            # Combine diversity and overlap for coverage proxy
            coverage = 0.5 * diversity_ratio + 0.5 * overlap
        
        # Combine: prioritize semantic similarity, then coverage
        score = 0.8 * float(sim_semantic) + 0.2 * float(coverage)
        return float(max(0.0, min(1.0, score)))
    
    def _calculate_boundary_accuracy(self, chapter: Dict, transcript_data: List[Dict]) -> float:
        """Calculate how well chapter boundaries align with content transitions."""
        # This is a simplified version - in practice, you'd use more sophisticated
        # topic modeling or semantic analysis
        return 0.8  # Placeholder - implement topic boundary detection
    
    def _calculate_temporal_consistency(self, chapter: Dict, all_chapters: List[Dict]) -> float:
        """Calculate temporal consistency with other chapters."""
        # Check if timestamps are in logical order
        start_time = chapter['start_time']
        end_time = chapter['end_time']
        
        # Basic checks
        if start_time >= end_time:
            return 0.0
        
        # Check for reasonable duration (not too short, not too long)
        duration = end_time - start_time
        if duration < 10:  # Too short
            return 0.5
        elif duration > 1800:  # Too long (>30 minutes)
            return 0.5
        else:
            return 1.0
    
    def _calculate_duration_appropriateness(self, chapter: Dict) -> float:
        """Calculate if chapter duration is appropriate for the content type."""
        duration = chapter['duration']
        
        # Optimal duration ranges for different content types
        if duration < 30:  # Very short
            return 0.3
        elif 30 <= duration <= 300:  # Good range (30 seconds to 5 minutes)
            return 1.0
        elif 300 < duration <= 600:  # Acceptable (5-10 minutes)
            return 0.8
        else:  # Too long
            return 0.5
    
    def _calculate_search_relevance(self, chapter: Dict, full_transcript: str) -> float:
        """Calculate search relevance score."""
        # Extract common search terms and check chapter coverage
        chapter_text = f"{chapter['title']} {chapter['summary']}"
        
        # Use TF-IDF to find important terms in full transcript
        try:
            tfidf_matrix = self.vectorizer.fit_transform([full_transcript, chapter_text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top terms from full transcript
            transcript_scores = tfidf_matrix[0].toarray()[0]
            top_indices = np.argsort(transcript_scores)[-10:]  # Top 10 terms
            important_terms = [feature_names[i] for i in top_indices]
            
            # Check how many important terms are covered in chapter
            chapter_lower = chapter_text.lower()
            covered_terms = sum(1 for term in important_terms if term in chapter_lower)
            
            return covered_terms / len(important_terms) if important_terms else 0.0
        except:
            return 0.5
    
    def _calculate_keyword_coverage(self, chapter: Dict, full_transcript: str) -> float:
        """Calculate keyword coverage score."""
        # Similar to search relevance but focused on keyword density
        chapter_text = f"{chapter['title']} {chapter['summary']}"
        
        # Extract keywords using simple frequency analysis
        transcript_words = re.findall(r'\b\w{4,}\b', full_transcript.lower())
        chapter_words = re.findall(r'\b\w{4,}\b', chapter_text.lower())
        
        if not transcript_words:
            return 0.0
        
        # Calculate coverage of frequent terms
        word_freq = defaultdict(int)
        for word in transcript_words:
            word_freq[word] += 1
        
        # Get top 20% most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        top_words = [word for word, freq in sorted_words[:len(sorted_words)//5]]
        
        chapter_word_set = set(chapter_words)
        covered_keywords = sum(1 for word in top_words if word in chapter_word_set)
        
        return covered_keywords / len(top_words) if top_words else 0.0
    
    def _calculate_navigation_utility(self, chapter: Dict) -> float:
        """Calculate navigation utility score."""
        # Factors: clear title, appropriate duration, good summary
        title_score = 1.0 if len(chapter['title']) > 10 and len(chapter['title']) < 100 else 0.5
        summary_score = 1.0 if len(chapter['summary']) > 50 else 0.5
        duration_score = self._calculate_duration_appropriateness(chapter)
        
        return (title_score + summary_score + duration_score) / 3.0
    
    def _detect_issues(self, metrics: EvaluationMetrics, overlapping_chapters: List[Dict], llm_errors: LLMErrorAnalysis) -> List[str]:
        """Detect issues with the chapter based on metrics."""
        issues = []
        
        if metrics.content_relevance < 0.5:
            issues.append("Low content relevance - summary doesn't match transcript well")
        
        if metrics.title_accuracy < 0.3:
            issues.append("Poor title accuracy - title doesn't describe content accurately")
        
        if metrics.summary_completeness < 0.4:
            issues.append("Incomplete summary - missing key content points")
        
        if metrics.bert_score_f1 < 0.3:
            issues.append("Low BERTScore F1 - poor semantic alignment between summary and content")
        
        if metrics.rouge_l_f1 < 0.2:
            issues.append("Low ROUGE-L F1 - summary lacks proper sentence structure alignment")
        
        if metrics.boundary_accuracy < 0.6:
            issues.append("Poor boundary alignment - chapter boundaries don't match content transitions")
        
        if metrics.redundancy_score > 0.7:
            issues.append("High redundancy - chapter overlaps significantly with others")
        
        if metrics.distinctiveness < 0.3:
            issues.append("Low distinctiveness - chapter lacks unique content")
        
        if metrics.duration_appropriateness < 0.6:
            issues.append("Inappropriate duration - chapter is too short or too long")
        
        if overlapping_chapters:
            issues.append(f"Overlapping content detected with {len(overlapping_chapters)} other chapters")
        
        # Add LLM error issues
        if llm_errors.hallucination_score > 0.7:
            issues.append("High hallucination risk - content may contain fabricated information")
        
        if llm_errors.factual_consistency_score < 0.3:
            issues.append("Low factual consistency - content contradicts source material")
        
        if llm_errors.coherence_score < 0.4:
            issues.append("Poor coherence - logical flow and structure issues")
        
        if llm_errors.bias_score > 0.5:
            issues.append("Potential bias detected - content may be subjective or biased")
        
        if llm_errors.detected_errors:
            for error in llm_errors.detected_errors:
                severity = llm_errors.error_severity.get(error, "low")
                issues.append(f"LLM Error ({severity}): {self.scoring_guide.common_llm_errors.get(error, error)}")
        
        return issues
    
    def _generate_query_recommendations(self, search_evaluations: List[SearchEvaluation], chapter: Dict) -> List[str]:
        """Generate chapter recommendations tied to user queries.
        Shows top 5 queries by combined relevance and navigation scores.
        Returns human-readable strings per qualifying query, including title and summary.
        """
        if not search_evaluations:
            return []
        
        # Calculate combined scores and sort
        query_scores = []
        for se in search_evaluations:
            rel = float(se.chapter_relevance_scores.get(0, 0.0))
            nav = float(se.navigation_utility_scores.get(0, 0.0))
            combined_score = (rel + nav) / 2.0  # Average of relevance and navigation
            query_scores.append((se.query, rel, nav, combined_score))
        
        # Sort by combined score and take top 5
        query_scores.sort(key=lambda x: x[3], reverse=True)
        top_queries = query_scores[:5]
        
        recs: List[str] = []
        for query, rel, nav, combined in top_queries:
            recs.append(f"Recommended for query '{query}' (relevance {rel:.2f}, navigation {nav:.2f}):\n  Title: {chapter['title']}\n  Summary: {chapter['summary']}")
        
        return recs
    
    def _extract_semantic_keywords(self, chapter: Dict) -> List[str]:
        """Extract semantic keywords from chapter content."""
        # Simple keyword extraction - in practice, use more sophisticated NLP
        text = f"{chapter['title']} {chapter['summary']}"
        words = re.findall(r'\b\w{4,}\b', text.lower())
        
        # Remove common stop words
        stop_words = {'this', 'that', 'with', 'from', 'they', 'them', 'their', 'there', 'these', 'those'}
        keywords = [word for word in words if word not in stop_words]
        
        # Return top keywords
        return list(set(keywords))[:10]
    
    def _extract_content_themes(self, chapter: Dict) -> List[str]:
        """Extract content themes from chapter."""
        # Placeholder for theme extraction
        return ["general content"]  # Implement topic modeling for better theme extraction

def main():
    """Example usage of the ChapterEvaluator."""
    # This would be used with actual chapter data
    print("Chapter Evaluator initialized successfully!")
    print("Use evaluate_chapters() method to analyze chapter quality.")

if __name__ == "__main__":
    main()
