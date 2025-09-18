#!/usr/bin/env python3
"""
Main Chapter Quality Evaluator

A comprehensive system for evaluating LLM-generated video chapters with automated metrics and manual review integration.

Usage:
    python evaluator.py <video_id> [options]

Options:
    --no-eval                    Skip evaluation
    --queries query1,query2,...  Custom user queries
    --manual-review              Save evaluation results for manual review
    --review-file file.json      Load your completed manual reviews
    --dashboard                  Generate visualization dashboard
    --help                       Show this help message

Examples:
    python evaluator.py P127jhj-8-Y
    python evaluator.py P127jhj-8-Y --queries "machine learning,tutorial"
    python evaluator.py P127jhj-8-Y --manual-review
    python evaluator.py P127jhj-8-Y --dashboard
    python evaluator.py P127jhj-8-Y --review-file data/extracted_chapters/manual_review_P127jhj-8-Y.json
"""

import sys
import os
import json
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings("ignore")
os.environ['CURL_CA_BUNDLE'] = ''
# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from vidtranscript2chapter import VideoChapterGenerator, Chapter
from chapter_evaluator import ChapterEvaluator, ChapterAnalysis, ManualReview
from visualization_dashboard import ChapterEvaluationDashboard

class MainEvaluator:
    """Main evaluator class that orchestrates the complete pipeline."""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the main evaluator with all components."""
        self.generator = VideoChapterGenerator(openai_api_key)
        self.evaluator = ChapterEvaluator(openai_api_key)
        self.dashboard = ChapterEvaluationDashboard()
    
    def run_evaluation(self, 
                      video_id: str,
                      evaluate: bool = True,
                      custom_queries: Optional[List[str]] = None,
                      manual_reviews: Optional[List[Dict]] = None,
                      enable_manual_review: bool = False,
                      generate_dashboard: bool = False) -> Dict:
        """
        Run the complete evaluation pipeline.
        
        Args:
            video_id: YouTube video ID
            evaluate: Whether to run quality evaluation
            custom_queries: Optional list of user search queries
            manual_reviews: Optional list of manual review data
            enable_manual_review: Whether to save data for manual review
            generate_dashboard: Whether to generate visualization dashboard
            
        Returns:
            Dictionary with all results
        """
        print(f"Chapter Quality Evaluator")
        print(f"Video ID: {video_id}")
        print("=" * 60)
        
        try:
            # Step 1: Extract transcript
            print("Step 1: Extracting transcript...")
            formatted_text, transcript_data = self.generator.extract_transcript(video_id)
            print(f"SUCCESS: Transcript extracted ({len(transcript_data)} segments)")
            
            # Step 2: Segment transcript
            print("Step 2: Segmenting transcript...")
            segments = self.generator.segment_transcript(transcript_data)
            print(f"SUCCESS: Transcript segmented into {len(segments)} chunks")
            
            # Step 3: Generate chapters
            print("Step 3: Generating chapters...")
            chapters = self.generator.generate_chapters(segments, video_id=video_id)
            print(f"SUCCESS: Generated {len(chapters)} chapters")
            
            # Step 4: Evaluate chapters (if requested)
            evaluation_results = None
            if evaluate:
                print("Step 4: Evaluating chapter quality...")
                evaluation_results = self.evaluator.evaluate_chapters(
                    chapters=[self._chapter_to_dict(ch) for ch in chapters],
                    transcript_data=transcript_data,
                    user_queries=custom_queries
                )
                print(f"SUCCESS: Evaluation completed for {len(evaluation_results)} chapters")
            
            # Step 5: Process manual reviews (if enabled)
            manual_review_results = None
            if enable_manual_review:
                print("👥 Step 5: Processing manual reviews...")
                if manual_reviews:
                    manual_review_results = self._process_manual_reviews(manual_reviews, evaluation_results)
                    print(f"SUCCESS: Processed {len(manual_reviews)} manual reviews")
                else:
                    # Save evaluation results for manual review
                    self._save_for_manual_review(video_id, chapters, evaluation_results)
                    print(f"SUCCESS: Saved evaluation results for manual review")
                    print(f"  Run manual review and then use --review-file to load your annotations")
            
            # Step 6: Generate reports
            step_num = 6 if enable_manual_review else 5
            print(f"📋 Step {step_num}: Generating reports...")
            reports = self._generate_reports(chapters, evaluation_results, video_id, manual_review_results)
            print("SUCCESS: Reports generated")
            
            # Step 7: Save results
            print("💾 Step 7: Saving results...")
            self._save_results(video_id, chapters, evaluation_results, reports)
            print("SUCCESS: Results saved")
            
            # Step 8: Generate dashboard (if requested)
            if generate_dashboard:
                print("Step 8: Generating visualization dashboard...")
                self._generate_dashboard(video_id)
                print("SUCCESS: Dashboard generated")
            
            return {
                'video_id': video_id,
                'chapters': chapters,
                'evaluation_results': evaluation_results,
                'manual_review_results': manual_review_results,
                'reports': reports,
                'transcript_data': transcript_data
            }
            
        except Exception as e:
            print(f"ERROR: Error processing video {video_id}: {str(e)}")
            raise
    
    def _chapter_to_dict(self, chapter: Chapter) -> Dict:
        """Convert Chapter object to dictionary."""
        return {
            'title': chapter.title,
            'summary': chapter.summary,
            'start_time': chapter.start_time,
            'end_time': chapter.end_time,
            'duration': chapter.duration,
            'start_timestamp': chapter.start_timestamp,
            'end_timestamp': chapter.end_timestamp,
            'youtube_timestamp': chapter.youtube_timestamp
        }
    
    def _save_results(self, video_id: str, chapters: List[Chapter], 
                     evaluation_results: List[ChapterAnalysis], reports: Dict) -> None:
        """Save all results to appropriate directories."""
        # Create directories if they don't exist
        os.makedirs('data/extracted_chapters', exist_ok=True)
        os.makedirs('data/outputs', exist_ok=True)
        
        # Save chapters to extracted_chapters directory
        chapters_data = {
            'video_id': video_id,
            'chapters': [self._chapter_to_dict(ch) for ch in chapters]
        }
        chapters_file = f"data/extracted_chapters/chapters_{video_id}.json"
        with open(chapters_file, 'w') as f:
            json.dump(chapters_data, f, indent=2)
        
        # Save evaluation results to outputs directory
        if evaluation_results:
            eval_data = []
            for result in evaluation_results:
                eval_dict = {
                    'chapter_data': result.chapter_data,
                    'transcript_segment': result.transcript_segment,
                    'evaluation_metrics': self._convert_metrics_for_json(result.evaluation_metrics.__dict__),
                    'issues_detected': result.issues_detected,
                    'recommendations': result.recommendations,
                    'semantic_keywords': result.semantic_keywords,
                    'content_themes': result.content_themes
                }
                
                # Add search evaluations if present
                if result.search_evaluations:
                    eval_dict['search_evaluations'] = [
                        {
                            'query': se.query,
                            'chapter_relevance_scores': se.chapter_relevance_scores,
                            'navigation_utility_scores': se.navigation_utility_scores,
                            'user_feedback': se.user_feedback
                        } for se in result.search_evaluations
                    ]
                
                eval_data.append(eval_dict)
            
            eval_file = f"data/outputs/evaluation_{video_id}.json"
            with open(eval_file, 'w') as f:
                json.dump(eval_data, f, indent=2)
        
        # Save reports to outputs directory
        reports_file = f"data/outputs/reports_{video_id}.json"
        with open(reports_file, 'w') as f:
            json.dump(reports, f, indent=2)
        
        print(f"SUCCESS: Results saved:")
        print(f"  - Chapters: {chapters_file} (extracted_chapters)")
        if evaluation_results:
            print(f"  - Evaluation: {eval_file} (outputs)")
        print(f"  - Reports: {reports_file} (outputs)")
    
    def _convert_metrics_for_json(self, metrics_dict: Dict) -> Dict:
        """Convert numpy types to Python native types for JSON serialization."""
        for key, value in metrics_dict.items():
            if hasattr(value, 'item'):  # numpy scalar
                metrics_dict[key] = value.item()
            elif hasattr(value, '__iter__') and not isinstance(value, (str, list, dict)):
                # numpy array or similar
                metrics_dict[key] = [float(x) for x in value]
        return metrics_dict
    
    def _save_for_manual_review(self, video_id: str, chapters: List[Chapter], 
                               evaluation_results: List[ChapterAnalysis]) -> None:
        """Save evaluation results in a format suitable for manual review."""
        from datetime import datetime
        
        manual_review_data = {
            "video_id": video_id,
            "timestamp": datetime.now().isoformat(),
            "chapters_for_review": []
        }
        
        for i, (chapter, result) in enumerate(zip(chapters, evaluation_results)):
            chapter_data = {
                "chapter_index": i,
                "chapter_info": {
                    "title": chapter.title,
                    "summary": chapter.summary,
                    "start_timestamp": chapter.start_timestamp,
                    "end_timestamp": chapter.end_timestamp,
                    "youtube_timestamp": chapter.youtube_timestamp
                },
                "automated_evaluation": self._convert_metrics_for_json(result.evaluation_metrics.__dict__),
                "detected_issues": result.issues_detected,
                "automated_recommendations": result.recommendations,
                "query_recommendations": result.recommendations,  # Query-based recommendations for rating
                "llm_error_analysis": result.llm_error_analysis.__dict__ if result.llm_error_analysis else None,
                "transcript_segment": result.transcript_segment
            }
            
            manual_review_data["chapters_for_review"].append(chapter_data)
        
        # Save to manual_reviews directory
        os.makedirs('data/manual_reviews', exist_ok=True)
        filename = f"data/manual_reviews/manual_review_{video_id}.json"
        with open(filename, 'w') as f:
            json.dump(manual_review_data, f, indent=2)
        
        print(f"SUCCESS: Manual review data saved to: {filename}")
        print(f"SUCCESS: {len(chapters)} chapters ready for your manual review")
        print(f"INFO: Use the Streamlit app under annotation/ to submit reviews")
        print(f"INFO: Or provide reviews programmatically and then run: python evaluator.py {video_id} --review-file {filename}")
    
    def _process_manual_reviews(self, manual_reviews: List[Dict], 
                               evaluation_results: List[ChapterAnalysis]) -> Dict:
        """Process manual review data and integrate with evaluation results."""
        processed_reviews = []
        review_statistics = {
            'total_reviews': len(manual_reviews),
            'reviewers': set(),
            'chapters_reviewed': set(),
            'average_scores': {},
            'inter_reviewer_agreement': {}
        }
        
        for review_data in manual_reviews:
            try:
                # Process review data into ManualReview object
                review = self.evaluator.process_manual_review(review_data)
                processed_reviews.append(review)
                
                # Add to evaluator
                self.evaluator.add_manual_review(review)
                
                # Update statistics
                review_statistics['reviewers'].add(review.reviewer_id)
                review_statistics['chapters_reviewed'].add(review.chapter_index)
                
                # Calculate average scores
                for metric in ['overall_quality_score', 'content_accuracy_score', 
                              'title_appropriateness_score', 'summary_quality_score',
                              'search_relevance_score', 'navigation_utility_score']:
                    if metric not in review_statistics['average_scores']:
                        review_statistics['average_scores'][metric] = []
                    review_statistics['average_scores'][metric].append(getattr(review, metric))
                
            except Exception as e:
                print(f"Warning: Error processing manual review: {e}")
                continue
        
        # Calculate final averages
        for metric, scores in review_statistics['average_scores'].items():
            review_statistics['average_scores'][metric] = sum(scores) / len(scores)
        
        return {
            'processed_reviews': [
                {
                    'reviewer_id': review.reviewer_id,
                    'chapter_index': review.chapter_index,
                    'overall_quality_score': review.overall_quality_score,
                    'content_accuracy_score': review.content_accuracy_score,
                    'title_appropriateness_score': review.title_appropriateness_score,
                    'summary_quality_score': review.summary_quality_score,
                    'search_relevance_score': review.search_relevance_score,
                    'navigation_utility_score': review.navigation_utility_score,
                    'issues_identified': review.issues_identified,
                    'recommendations': review.recommendations,
                    'review_timestamp': review.review_timestamp,
                    'confidence_level': review.confidence_level
                } for review in processed_reviews
            ],
            'statistics': review_statistics
        }
    
    def _generate_reports(self, chapters: List[Chapter], 
                         evaluation_results: List[ChapterAnalysis],
                         video_id: str,
                         manual_review_results: Dict = None) -> Dict:
        """Generate comprehensive evaluation reports."""
        reports = {
            'summary_report': self._generate_summary_report(chapters, evaluation_results),
            'quality_metrics': self._generate_quality_metrics_report(evaluation_results),
            'issues_report': self._generate_issues_report(evaluation_results),
            'recommendations_report': self._generate_recommendations_report(evaluation_results)
        }
        
        # Add manual review results to reports
        if manual_review_results:
            reports['manual_review_summary'] = self._generate_manual_review_summary(manual_review_results)
        
        return reports
    
    def _generate_summary_report(self, chapters: List[Chapter], 
                                evaluation_results: List[ChapterAnalysis]) -> Dict:
        """Generate high-level summary report."""
        if not evaluation_results:
            return {'status': 'No evaluation results available'}
        
        scores = [result.evaluation_metrics.overall_score for result in evaluation_results]
        
        return {
            'total_chapters': len(chapters),
            'quality_statistics': {
                'average_quality_score': sum(scores) / len(scores),
                'high_quality_chapters': sum(1 for score in scores if score >= 0.8),
                'medium_quality_chapters': sum(1 for score in scores if 0.5 <= score < 0.8),
                'low_quality_chapters': sum(1 for score in scores if score < 0.5)
            },
            'issues_summary': {
                'total_issues_detected': sum(len(result.issues_detected) for result in evaluation_results),
                'chapters_with_issues': sum(1 for result in evaluation_results if result.issues_detected)
            }
        }
    
    def _generate_quality_metrics_report(self, evaluation_results: List[ChapterAnalysis]) -> Dict:
        """Generate quality metrics report."""
        if not evaluation_results:
            return {}
        
        metrics = ['content_relevance', 'title_accuracy', 'summary_completeness',
                  'bert_score_f1', 'rouge_l_f1', 'search_relevance', 'navigation_utility']
        
        report = {}
        for metric in metrics:
            scores = [getattr(result.evaluation_metrics, metric) for result in evaluation_results]
            report[metric] = {
                'average': sum(scores) / len(scores),
                'min': min(scores),
                'max': max(scores)
            }
        
        return report
    
    def _generate_issues_report(self, evaluation_results: List[ChapterAnalysis]) -> Dict:
        """Generate issues report."""
        if not evaluation_results:
            return {}
        
        all_issues = []
        for result in evaluation_results:
            all_issues.extend(result.issues_detected)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        return {
            'total_issues': len(all_issues),
            'unique_issues': len(issue_counts),
            'most_common_issues': dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        }
    
    def _generate_recommendations_report(self, evaluation_results: List[ChapterAnalysis]) -> Dict:
        """Generate recommendations report."""
        if not evaluation_results:
            return {}
        
        all_recommendations = []
        for result in evaluation_results:
            all_recommendations.extend(result.recommendations)
        
        rec_counts = {}
        for rec in all_recommendations:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        return {
            'total_recommendations': len(all_recommendations),
            'unique_recommendations': len(rec_counts),
            'priority_recommendations': [
                {'recommendation': rec, 'frequency': count}
                for rec, count in sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
        }
    
    def _generate_manual_review_summary(self, manual_review_results: Dict) -> Dict:
        """Generate summary of manual review results."""
        if 'statistics' not in manual_review_results:
            return {'status': 'No manual reviews processed'}
        
        stats = manual_review_results['statistics']
        
        return {
            'review_overview': {
                'total_reviews': stats['total_reviews'],
                'unique_reviewers': len(stats['reviewers']),
                'chapters_reviewed': len(stats['chapters_reviewed'])
            },
            'average_scores': stats['average_scores']
        }
    
    def _generate_dashboard(self, video_id: str) -> None:
        """Generate visualization dashboard."""
        os.makedirs('data/outputs', exist_ok=True)
        save_path = f"data/outputs/dashboard_{video_id}.png"
        self.dashboard.create_comprehensive_dashboard(video_id, save_path=save_path)
        print(f"SUCCESS: Dashboard saved to: {save_path}")

def main():
    """Main function to run the evaluator."""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    video_id = sys.argv[1]
    evaluate = "--no-eval" not in sys.argv
    enable_manual_review = "--manual-review" in sys.argv
    generate_dashboard = "--dashboard" in sys.argv
    
    # Parse custom queries if provided
    custom_queries = None
    for i, arg in enumerate(sys.argv):
        if arg == "--queries" and i + 1 < len(sys.argv):
            custom_queries = [q.strip() for q in sys.argv[i + 1].split(',')]
            break
    
    # Parse manual review file if provided
    manual_reviews = None
    for i, arg in enumerate(sys.argv):
        if arg == "--review-file" and i + 1 < len(sys.argv):
            review_file = sys.argv[i + 1]
            try:
                with open(review_file, 'r') as f:
                    manual_reviews = json.load(f)
                print(f"SUCCESS: Loaded {len(manual_reviews)} manual reviews from {review_file}")
            except FileNotFoundError:
                print(f"ERROR: Review file not found: {review_file}")
                sys.exit(1)
            except json.JSONDecodeError:
                print(f"ERROR: Invalid JSON in review file: {review_file}")
                sys.exit(1)
            break
    
    try:
        evaluator = MainEvaluator()
        results = evaluator.run_evaluation(
            video_id=video_id,
            evaluate=evaluate,
            custom_queries=custom_queries,
            manual_reviews=manual_reviews,
            enable_manual_review=enable_manual_review,
            generate_dashboard=generate_dashboard
        )
        
        # Print summary
        if evaluate and results['evaluation_results']:
            print("\n" + "="*60)
            print("EVALUATION SUMMARY")
            print("="*60)
            
            summary = results['reports']['summary_report']
            print(f"Total Chapters: {summary['total_chapters']}")
            print(f"Average Quality Score: {summary['quality_statistics']['average_quality_score']:.2f}")
            print(f"Total Issues Detected: {summary['issues_summary']['total_issues_detected']}")
            
            # Show top 5 chapters by search relevance score
            print("\n" + "-"*60)
            print("TOP 5 CHAPTERS BY SEARCH RELEVANCE")
            print("-"*60)
            if results['evaluation_results']:
                # Sort chapters by search relevance score in descending order
                sorted_chapters = sorted(
                    enumerate(results['evaluation_results']), 
                    key=lambda x: x[1].evaluation_metrics.search_relevance, 
                    reverse=True
                )
                
                # Show top 5 chapters
                top_5_chapters = sorted_chapters[:5]
                for i, (original_idx, result) in enumerate(top_5_chapters, 1):
                    chapter_num = original_idx + 1
                    title = result.chapter_data['title']
                    relevance_score = result.evaluation_metrics.search_relevance
                    overall_score = result.evaluation_metrics.overall_score
                    
                    print(f"\n{i}. Chapter {chapter_num}: {title}")
                    print(f"   Search Relevance: {relevance_score:.3f}")
                    print(f"   Overall Score: {overall_score:.3f}")
                    
                    # Show top recommendation for this chapter if available
                    if result.recommendations:
                        top_rec = result.recommendations[0]
                        print(f"   Top Recommendation: {top_rec}")
                
                if len(results['evaluation_results']) > 5:
                    remaining = len(results['evaluation_results']) - 5
                    print(f"\n   ... and {remaining} more chapters")
            else:
                print("No evaluation results available")
        
        print(f"\n" + "="*60)
        print(f"EVALUATION COMPLETED: {video_id}")
        print("="*60)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
