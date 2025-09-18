#!/usr/bin/env python3
"""
Chapter Evaluation Visualization Dashboard

This module provides comprehensive visualization capabilities for chapter evaluation results,
including quality metrics, issues analysis, and recommendations.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import os

class ChapterEvaluationDashboard:
    """Dashboard for visualizing chapter evaluation results."""
    
    def __init__(self, style: str = 'whitegrid'):
        """Initialize the dashboard with styling."""
        plt.style.use('default')
        sns.set_style(style)
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'success': '#F18F01',
            'warning': '#C73E1D',
            'info': '#6C5CE7'
        }
    
    def create_comprehensive_dashboard(self, 
                                     video_id: str, 
                                     evaluation_file: str = None,
                                     reports_file: str = None,
                                     save_path: str = None) -> None:
        """Create a comprehensive dashboard with all visualizations."""
        
        # Load data
        evaluation_data = self._load_evaluation_data(video_id, evaluation_file)
        reports_data = self._load_reports_data(video_id, reports_file)
        
        if not evaluation_data:
            print("No evaluation data found. Please run evaluation first.")
            return
        
        # Create figure with subplots (4x4 grid)
        fig = plt.figure(figsize=(24, 20))
        
        # 1. Overall Quality Score Distribution
        ax1 = plt.subplot(4, 4, 1)
        self._plot_quality_distribution(evaluation_data, ax1)
        
        # 2. Quality Metrics Radar Chart (Automated vs Human averages)
        ax2 = plt.subplot(4, 4, 2, projection='polar')
        self._plot_metrics_radar(evaluation_data, reports_data, ax2)
        
        # 3. BERTScore and ROUGE Metrics Comparison
        ax3 = plt.subplot(4, 4, 3)
        self._plot_bert_rouge_comparison(evaluation_data, ax3)
        
        # 4. Issues Analysis (refined categories)
        ax4 = plt.subplot(4, 4, 4)
        self._plot_issues_analysis(evaluation_data, ax4)
        
        # 5. Per-Chapter Difference (Human - Automated) Heatmap
        ax5 = plt.subplot(4, 4, 5)
        self._plot_per_chapter_human_vs_auto(evaluation_data, reports_data, ax5)
        
        # 6. Redundancy Analysis
        ax6 = plt.subplot(4, 4, 6)
        self._plot_redundancy_analysis(evaluation_data, ax6)
        
        # 7. Search Relevance Heatmap
        ax7 = plt.subplot(4, 4, 7)
        self._plot_search_relevance_heatmap(evaluation_data, ax7)
        
        # 8. Content Quality Trends (with Human baselines)
        ax8 = plt.subplot(4, 4, 8)
        self._plot_content_quality_trends(evaluation_data, reports_data, ax8)
        
        # 9. BERTScore Trends
        ax9 = plt.subplot(4, 4, 9)
        self._plot_bert_score_trends(evaluation_data, ax9)
        
        # 10. ROUGE Score Trends
        ax10 = plt.subplot(4, 4, 10)
        self._plot_rouge_score_trends(evaluation_data, ax10)
        
        # 11. Search & Navigation Trends (with Human baselines)
        ax11 = plt.subplot(4, 4, 11)
        self._plot_search_nav_trends(evaluation_data, reports_data, ax11)
        
        # 12. Chapter Timeline
        ax12 = plt.subplot(4, 4, 12)
        self._plot_chapter_timeline(evaluation_data, ax12)
        
        # 13. Metric Difference Summary (Human - Automated averages)
        ax13 = plt.subplot(4, 4, 13)
        self._plot_metric_difference_summary(evaluation_data, reports_data, ax13)
        
        # 14. Recommendation Ratings
        ax14 = plt.subplot(4, 4, 14)
        self._plot_recommendation_ratings(reports_data, ax14)
        
        # 15. Human vs Automated Metric Averages
        ax15 = plt.subplot(4, 4, 15)
        self._plot_human_vs_auto_averages(evaluation_data, reports_data, ax15)
        
        # 16. Per-Chapter Overall: Automated vs Human
        ax16 = plt.subplot(4, 4, 16)
        self._plot_overall_vs_human(evaluation_data, reports_data, ax16)
        
        # Add title and adjust layout
        fig.suptitle(f'Chapter Evaluation Dashboard - Video {video_id}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Dashboard saved to: {save_path}")
        else:
            plt.show()

    def _get_human_aggregates(self, reports_data: Dict) -> Dict:
        """Extract human aggregate scores from reports data. Returns dict with averages and per_chapter."""
        summary = reports_data.get('manual_review_summary') if isinstance(reports_data, dict) else None
        if not summary:
            return {'average_scores': {}, 'per_chapter': []}
        return {
            'average_scores': summary.get('average_scores', {}),
            'per_chapter': summary.get('per_chapter', [])
        }

    def _plot_human_vs_auto_averages(self, evaluation_data: List[Dict], reports_data: Dict, ax: plt.Axes) -> None:
        """Compare automated metric averages (0-1) with human averages (1-5 -> normalized)."""
        # Metrics with human counterparts
        metric_pairs = [
            ('content_relevance', 'content_accuracy_score'),
            ('title_accuracy', 'title_appropriateness_score'),
            ('summary_completeness', 'summary_quality_score'),
            ('search_relevance', 'search_relevance_score'),
            ('navigation_utility', 'navigation_utility_score')
        ]

        # Automated averages (0-1)
        auto_avgs = {}
        if evaluation_data:
            for auto_metric, _ in metric_pairs:
                vals = [item['evaluation_metrics'][auto_metric] for item in evaluation_data]
                auto_avgs[auto_metric] = float(np.mean(vals)) if vals else 0.0

        # Human averages (1-5 -> 0-1)
        human = self._get_human_aggregates(reports_data)
        human_avg = {}
        for _, human_metric in metric_pairs:
            val = human['average_scores'].get(human_metric)
            human_avg[human_metric] = (float(val) / 5.0) if isinstance(val, (int, float)) else 0.0

        labels = [
            'Content', 'Title', 'Summary', 'Search', 'Navigation'
        ]
        auto_vals = [auto_avgs.get(m, 0.0) for m, _ in metric_pairs]
        human_vals = [human_avg.get(hm, 0.0) for _, hm in metric_pairs]

        x = np.arange(len(labels))
        width = 0.35
        ax.bar(x - width/2, auto_vals, width, label='Automated', color=self.colors['primary'])
        ax.bar(x + width/2, human_vals, width, label='Human (normalized)', color=self.colors['success'])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Human vs Automated Metric Averages')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_overall_vs_human(self, evaluation_data: List[Dict], reports_data: Dict, ax: plt.Axes) -> None:
        """Compare per-chapter automated overall (0-1) vs human overall (avg of 6 metrics, normalized)."""
        human = self._get_human_aggregates(reports_data)
        per_chapter = {int(item['chapter_index']): item for item in human.get('per_chapter', []) if 'chapter_index' in item}

        auto_overall = [item['evaluation_metrics']['overall_score'] for item in evaluation_data] if evaluation_data else []

        # Human overall per chapter: average of six metrics normalized to 0..1
        human_overall = []
        for i in range(len(auto_overall)):
            entry = per_chapter.get(i)
            if not entry:
                human_overall.append(0.0)
                continue
            avgs = entry.get('averages', {})
            metrics = [
                avgs.get('overall_quality_score'),
                avgs.get('content_accuracy_score'),
                avgs.get('title_appropriateness_score'),
                avgs.get('summary_quality_score'),
                avgs.get('search_relevance_score'),
                avgs.get('navigation_utility_score')
            ]
            vals = [v for v in metrics if isinstance(v, (int, float))]
            human_overall.append((sum(vals)/len(vals)/5.0) if vals else 0.0)

        x = np.arange(len(auto_overall))
        width = 0.35
        ax.bar(x - width/2, auto_overall, width, label='Automated Overall', color=self.colors['primary'])
        ax.bar(x + width/2, human_overall, width, label='Human Overall (normalized)', color=self.colors['success'])
        ax.set_xticks(x)
        ax.set_xticklabels([f'Ch{i+1}' for i in x], rotation=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Per-Chapter: Automated vs Human Overall')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _load_evaluation_data(self, video_id: str, evaluation_file: str = None) -> List[Dict]:
        """Load evaluation data from file."""
        if evaluation_file is None:
            evaluation_file = f"data/outputs/evaluation_{video_id}.json"
        
        if not os.path.exists(evaluation_file):
            return []
        
        with open(evaluation_file, 'r') as f:
            return json.load(f)
    
    def _load_reports_data(self, video_id: str, reports_file: str = None) -> Dict:
        """Load reports data from file."""
        if reports_file is None:
            reports_file = f"data/outputs/reports_{video_id}.json"
        
        if not os.path.exists(reports_file):
            return {}
        
        with open(reports_file, 'r') as f:
            return json.load(f)
    
    def _plot_quality_distribution(self, data: List[Dict], ax: plt.Axes) -> None:
        """Plot overall quality score distribution."""
        scores = [item['evaluation_metrics']['overall_score'] for item in data]
        
        ax.hist(scores, bins=10, color=self.colors['primary'], alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(scores), color=self.colors['warning'], 
                  linestyle='--', linewidth=2, label=f'Mean: {np.mean(scores):.2f}')
        ax.set_title('Overall Quality Score Distribution')
        ax.set_xlabel('Quality Score')
        ax.set_ylabel('Number of Chapters')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_metrics_radar(self, data: List[Dict], reports_data: Dict, ax: plt.Axes) -> None:
        """Plot metrics radar chart: automated averages vs human averages (normalized).
        Human line is shown only for mapped metrics to avoid misleading zeros.
        """
        metrics = ['content_relevance', 'title_accuracy', 'summary_completeness',
                  'boundary_accuracy', 'distinctiveness', 'search_relevance']
        # Automated averages (0..1)
        auto_avg = []
        for metric in metrics:
            scores = [item['evaluation_metrics'][metric] for item in data]
            auto_avg.append(float(np.mean(scores)) if scores else 0.0)
        # Human averages mapped to closest counterparts
        human_map = {
            'content_relevance': 'content_accuracy_score',
            'title_accuracy': 'title_appropriateness_score',
            'summary_completeness': 'summary_quality_score',
            'boundary_accuracy': None,  # no direct human metric
            'distinctiveness': None,    # no direct human metric
            'search_relevance': 'search_relevance_score'
        }
        human_summary = self._get_human_aggregates(reports_data)
        human_vals_all = []
        for metric in metrics:
            hm = human_map[metric]
            if hm is None:
                human_vals_all.append(np.nan)
            else:
                val = human_summary['average_scores'].get(hm)
                human_vals_all.append((float(val)/5.0) if isinstance(val, (int, float)) else np.nan)
        # Radar chart prep
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles_closed = angles + angles[:1]
        auto_vals_closed = auto_avg + auto_avg[:1]
        # Plot automated
        ax.plot(angles_closed, auto_vals_closed, 'o-', linewidth=2, color=self.colors['primary'], label='Automated Avg')
        ax.fill(angles_closed, auto_vals_closed, alpha=0.15, color=self.colors['primary'])
        # Plot human only on mapped metrics (skip NaNs)
        mapped_indices = [i for i, v in enumerate(human_vals_all) if not np.isnan(v)]
        if mapped_indices:
            angles_h = [angles[i] for i in mapped_indices] + [angles[mapped_indices[0]]]
            human_h = [human_vals_all[i] for i in mapped_indices] + [human_vals_all[mapped_indices[0]]]
            ax.plot(angles_h, human_h, 'o--', linewidth=2, color=self.colors['success'], label='Human Avg (norm)')
        # Axes labels
        ax.set_xticks(angles)
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics])
        ax.set_ylim(0, 1)
        ax.set_title('Average Metrics: Automated vs Human')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        ax.grid(True)
    
    def _plot_bert_rouge_comparison(self, data: List[Dict], ax: plt.Axes) -> None:
        """Plot BERTScore and ROUGE metrics comparison."""
        bert_f1_scores = [item['evaluation_metrics']['bert_score_f1'] for item in data]
        rouge_l_scores = [item['evaluation_metrics']['rouge_l_f1'] for item in data]
        rouge_1_scores = [item['evaluation_metrics']['rouge_1_f1'] for item in data]
        rouge_2_scores = [item['evaluation_metrics']['rouge_2_f1'] for item in data]
        
        # Create grouped bar chart
        x = np.arange(len(data))
        width = 0.2
        
        ax.bar(x - 1.5*width, bert_f1_scores, width, label='BERTScore F1', color=self.colors['primary'], alpha=0.8)
        ax.bar(x - 0.5*width, rouge_l_scores, width, label='ROUGE-L F1', color=self.colors['secondary'], alpha=0.8)
        ax.bar(x + 0.5*width, rouge_1_scores, width, label='ROUGE-1 F1', color=self.colors['success'], alpha=0.8)
        ax.bar(x + 1.5*width, rouge_2_scores, width, label='ROUGE-2 F1', color=self.colors['warning'], alpha=0.8)
        
        ax.set_xlabel('Chapter')
        ax.set_ylabel('Score')
        ax.set_title('BERTScore vs ROUGE Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Ch{i+1}' for i in range(len(data))])
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    def _plot_issues_analysis(self, data: List[Dict], ax: plt.Axes) -> None:
        """Plot issues analysis with refined categories (avoid 'Other')."""
        issue_counts = {
            'Content Quality': 0,
            'Advanced Metrics': 0,
            'Structural': 0,
            'Redundancy': 0,
            'Duration': 0,
            'Coherence': 0,
            'Factual Consistency': 0,
            'Hallucination': 0,
            'Bias': 0
        }
        for item in data:
            for issue in item['issues_detected']:
                lower = issue.lower()
                if 'content relevance' in lower or 'summary' in lower or 'title accuracy' in lower:
                    issue_counts['Content Quality'] += 1
                elif 'bert' in lower or 'rouge' in lower:
                    issue_counts['Advanced Metrics'] += 1
                elif 'boundary' in lower or 'temporal' in lower or 'timestamp' in lower:
                    issue_counts['Structural'] += 1
                elif 'redundancy' in lower or 'distinctiveness' in lower:
                    issue_counts['Redundancy'] += 1
                elif 'duration' in lower:
                    issue_counts['Duration'] += 1
                elif 'coherence' in lower:
                    issue_counts['Coherence'] += 1
                elif 'factual' in lower or 'consistency' in lower:
                    issue_counts['Factual Consistency'] += 1
                elif 'hallucination' in lower:
                    issue_counts['Hallucination'] += 1
                elif 'bias' in lower:
                    issue_counts['Bias'] += 1
        
        categories = [k for k, v in issue_counts.items() if v > 0]
        counts = [issue_counts[k] for k in categories]
        
        if categories:
            bars = ax.bar(categories, counts, color=self.colors['warning'])
            ax.set_title('Issues by Category (Refined)')
            ax.set_ylabel('Number of Issues')
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            for bar, count in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
        else:
            ax.text(0.5, 0.5, 'No Issues Detected', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12)
            ax.set_title('Issues Analysis')
    
    def _plot_duration_analysis(self, data: List[Dict], ax: plt.Axes) -> None:
        """Deprecated: duration-based chart removed per requirements."""
        ax.axis('off')
    
    def _plot_redundancy_analysis(self, data: List[Dict], ax: plt.Axes) -> None:
        """Plot redundancy analysis."""
        redundancy_scores = [item['evaluation_metrics']['redundancy_score'] for item in data]
        distinctiveness_scores = [item['evaluation_metrics']['distinctiveness'] for item in data]
        
        ax.scatter(redundancy_scores, distinctiveness_scores, 
                  c=self.colors['primary'], alpha=0.7, s=100)
        
        # Add chapter labels
        for i, (red, dist) in enumerate(zip(redundancy_scores, distinctiveness_scores)):
            ax.annotate(f'Ch{i+1}', (red, dist), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Redundancy Score')
        ax.set_ylabel('Distinctiveness Score')
        ax.set_title('Redundancy vs Distinctiveness')
        ax.grid(True, alpha=0.3)
        
        # Add ideal zone
        ax.axhspan(0.7, 1.0, alpha=0.2, color='green', label='Ideal Zone')
        ax.legend()
    
    def _plot_search_relevance_heatmap(self, data: List[Dict], ax: plt.Axes) -> None:
        """Plot search relevance heatmap."""
        metrics = ['search_relevance', 'keyword_coverage', 'navigation_utility']
        
        # Create matrix
        matrix = []
        for item in data:
            row = [item['evaluation_metrics'][metric] for metric in metrics]
            matrix.append(row)
        
        matrix = np.array(matrix)
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([m.replace('_', '\n') for m in metrics])
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([f'Ch{i+1}' for i in range(len(data))])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Score')
        
        ax.set_title('Search Utility Metrics')
        
        # Add text annotations
        for i in range(len(data)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f'{matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

    def _plot_per_chapter_human_vs_auto(self, data: List[Dict], reports_data: Dict, ax: plt.Axes) -> None:
        """Heatmap: per-chapter difference (Human - Automated) for key metrics.
        Missing human metrics are treated as 0 (i.e., 0.0 normalized).
        """
        metric_pairs = [
            ('content_relevance', 'content_accuracy_score'),
            ('title_accuracy', 'title_appropriateness_score'),
            ('summary_completeness', 'summary_quality_score'),
            ('search_relevance', 'search_relevance_score'),
            ('navigation_utility', 'navigation_utility_score')
        ]
        human = self._get_human_aggregates(reports_data)
        per_chapter = {int(item['chapter_index']): item for item in human.get('per_chapter', []) if 'chapter_index' in item}
        rows = []
        for i, item in enumerate(data):
            row = []
            for auto_metric, human_metric in metric_pairs:
                auto_val = float(item['evaluation_metrics'].get(auto_metric, 0.0))
                entry = per_chapter.get(i)
                human_avg = entry.get('averages', {}).get(human_metric) if entry else 0.0
                human_val = float(human_avg)/5.0 if isinstance(human_avg, (int, float)) else 0.0
                row.append(human_val - auto_val)
            rows.append(row)
        matrix = np.array(rows)
        if matrix.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Per-Chapter Difference (Human - Auto)')
            return
        im = ax.imshow(matrix, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(metric_pairs)))
        ax.set_xticklabels([m.replace('_', '\n') for m, _ in metric_pairs], fontsize=9)
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([f'Ch{i+1}' for i in range(len(data))])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Human - Auto (0..1)')
        ax.set_title('Per-Chapter Difference (Human - Auto)')

    def _plot_per_chapter_diff(self, data: List[Dict], reports_data: Dict, ax: plt.Axes) -> None:
        """Heatmap: per-chapter difference (Automated - Human normalized)."""
        metric_pairs = [
            ('content_relevance', 'content_accuracy_score'),
            ('title_accuracy', 'title_appropriateness_score'),
            ('summary_completeness', 'summary_quality_score'),
            ('search_relevance', 'search_relevance_score'),
            ('navigation_utility', 'navigation_utility_score')
        ]
        human = self._get_human_aggregates(reports_data)
        per_chapter = {int(item['chapter_index']): item for item in human.get('per_chapter', []) if 'chapter_index' in item}
        rows = []
        for i, item in enumerate(data):
            row = []
            for auto_metric, human_metric in metric_pairs:
                auto_val = float(item['evaluation_metrics'].get(auto_metric, 0.0))
                entry = per_chapter.get(i)
                human_avg = entry.get('averages', {}).get(human_metric) if entry else 0.0
                human_val = float(human_avg)/5.0 if isinstance(human_avg, (int, float)) else 0.0
                row.append(human_val - auto_val)
            rows.append(row)
        matrix = np.array(rows)
        if matrix.size == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Per-Chapter Difference (Human - Auto)')
            return
        im = ax.imshow(matrix, cmap='bwr', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(metric_pairs)))
        ax.set_xticklabels([m.replace('_', '\n') for m, _ in metric_pairs])
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([f'Ch{i+1}' for i in range(len(data))])
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Human - Auto (0..1)')
        ax.set_title('Per-Chapter Difference (Human - Auto)')
        # Add zero reference line
        ax.axhline(y=-0.5, color='k', linewidth=0.5, alpha=0.3)

    def _plot_metric_difference_summary(self, data: List[Dict], reports_data: Dict, ax: plt.Axes) -> None:
        """Bar chart: average (Human - Automated) per metric for quick summary."""
        metric_pairs = [
            ('content_relevance', 'content_accuracy_score', 'Content'),
            ('title_accuracy', 'title_appropriateness_score', 'Title'),
            ('summary_completeness', 'summary_quality_score', 'Summary'),
            ('search_relevance', 'search_relevance_score', 'Search'),
            ('navigation_utility', 'navigation_utility_score', 'Navigation')
        ]
        human = self._get_human_aggregates(reports_data)
        per_chapter = {int(item['chapter_index']): item for item in human.get('per_chapter', []) if 'chapter_index' in item}
        diffs = []
        labels = []
        for auto_metric, human_metric, label in metric_pairs:
            auto_vals = [float(item['evaluation_metrics'].get(auto_metric, 0.0)) for item in data]
            human_vals = []
            for i in range(len(data)):
                entry = per_chapter.get(i)
                human_avg = entry.get('averages', {}).get(human_metric) if entry else 0.0
                human_vals.append((float(human_avg)/5.0) if isinstance(human_avg, (int, float)) else 0.0)
            if auto_vals:
                diffs.append(float(np.mean([h - a for h, a in zip(human_vals, auto_vals)])))
                labels.append(label)
        x = np.arange(len(diffs))
        colors = [self.colors['success'] if d >= 0 else self.colors['warning'] for d in diffs]
        bars = ax.bar(x, diffs, color=colors)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Human - Auto (avg)')
        ax.set_title('Metric Difference Summary (Human - Automated)')
        for rect, d in zip(bars, diffs):
            ax.text(rect.get_x() + rect.get_width()/2, rect.get_height() + (0.02 if d >= 0 else -0.04), f'{d:.2f}',
                    ha='center', va='bottom' if d >= 0 else 'top', fontsize=9)
    
    def _plot_content_quality_trends(self, data: List[Dict], reports_data: Dict, ax: plt.Axes) -> None:
        """Plot content quality trends over chapters with human baselines."""
        chapters = range(1, len(data) + 1)
        content_relevance = [item['evaluation_metrics']['content_relevance'] for item in data]
        title_accuracy = [item['evaluation_metrics']['title_accuracy'] for item in data]
        summary_completeness = [item['evaluation_metrics']['summary_completeness'] for item in data]
        
        ax.plot(chapters, content_relevance, 'o-', label='Content Relevance (Auto)', 
               color=self.colors['primary'])
        ax.plot(chapters, title_accuracy, 's-', label='Title Accuracy (Auto)', 
               color=self.colors['secondary'])
        ax.plot(chapters, summary_completeness, '^-', label='Summary Completeness (Auto)', 
               color=self.colors['success'])
        
        # Human baselines (flat lines)
        human = self._get_human_aggregates(reports_data)
        def hline(avg_key, label, color):
            val = human['average_scores'].get(avg_key)
            if isinstance(val, (int, float)):
                ax.axhline(y=float(val)/5.0, linestyle='--', color=color, alpha=0.7, label=label)
        hline('content_accuracy_score', 'Human Content Accuracy (avg)', self.colors['primary'])
        hline('title_appropriateness_score', 'Human Title Appropriateness (avg)', self.colors['secondary'])
        hline('summary_quality_score', 'Human Summary Quality (avg)', self.colors['success'])
        
        ax.set_xlabel('Chapter Number')
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Content Quality Trends (with Human Baselines)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_bert_score_trends(self, data: List[Dict], ax: plt.Axes) -> None:
        """Plot BERTScore trends over chapters."""
        chapters = range(1, len(data) + 1)
        bert_precision = [item['evaluation_metrics']['bert_score_precision'] for item in data]
        bert_recall = [item['evaluation_metrics']['bert_score_recall'] for item in data]
        bert_f1 = [item['evaluation_metrics']['bert_score_f1'] for item in data]
        
        ax.plot(chapters, bert_precision, 'o-', label='BERTScore Precision', 
               color=self.colors['primary'], linewidth=2)
        ax.plot(chapters, bert_recall, 's-', label='BERTScore Recall', 
               color=self.colors['secondary'], linewidth=2)
        ax.plot(chapters, bert_f1, '^-', label='BERTScore F1', 
               color=self.colors['success'], linewidth=2)
        
        ax.set_xlabel('Chapter Number')
        ax.set_ylabel('BERTScore')
        ax.set_title('BERTScore Trends Across Chapters')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_rouge_score_trends(self, data: List[Dict], ax: plt.Axes) -> None:
        """Plot ROUGE score trends over chapters."""
        chapters = range(1, len(data) + 1)
        rouge_1 = [item['evaluation_metrics']['rouge_1_f1'] for item in data]
        rouge_2 = [item['evaluation_metrics']['rouge_2_f1'] for item in data]
        rouge_l = [item['evaluation_metrics']['rouge_l_f1'] for item in data]
        
        ax.plot(chapters, rouge_1, 'o-', label='ROUGE-1 F1', 
               color=self.colors['primary'], linewidth=2)
        ax.plot(chapters, rouge_2, 's-', label='ROUGE-2 F1', 
               color=self.colors['secondary'], linewidth=2)
        ax.plot(chapters, rouge_l, '^-', label='ROUGE-L F1', 
               color=self.colors['success'], linewidth=2)
        
        ax.set_xlabel('Chapter Number')
        ax.set_ylabel('ROUGE Score')
        ax.set_title('ROUGE Score Trends Across Chapters')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

    def _plot_search_nav_trends(self, data: List[Dict], reports_data: Dict, ax: plt.Axes) -> None:
        """Plot search relevance and navigation utility trends with human baselines."""
        chapters = range(1, len(data) + 1)
        search_relevance = [item['evaluation_metrics']['search_relevance'] for item in data]
        navigation_utility = [item['evaluation_metrics']['navigation_utility'] for item in data]
        ax.plot(chapters, search_relevance, 'o-', label='Search Relevance (Auto)', color=self.colors['primary'])
        ax.plot(chapters, navigation_utility, 's-', label='Navigation Utility (Auto)', color=self.colors['secondary'])
        # Human baselines
        human = self._get_human_aggregates(reports_data)
        if isinstance(human.get('average_scores', {}).get('search_relevance_score'), (int, float)):
            ax.axhline(y=float(human['average_scores']['search_relevance_score'])/5.0, linestyle='--', color=self.colors['primary'], alpha=0.7, label='Human Search Relevance (avg)')
        if isinstance(human.get('average_scores', {}).get('navigation_utility_score'), (int, float)):
            ax.axhline(y=float(human['average_scores']['navigation_utility_score'])/5.0, linestyle='--', color=self.colors['secondary'], alpha=0.7, label='Human Navigation Utility (avg)')
        ax.set_xlabel('Chapter Number')
        ax.set_ylabel('Score (0-1)')
        ax.set_title('Search & Navigation Trends (with Human Baselines)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_chapter_timeline(self, data: List[Dict], ax: plt.Axes) -> None:
        """Plot chapter timeline with quality scores."""
        start_times = [item['chapter_data']['start_time'] for item in data]
        durations = [item['chapter_data']['duration'] for item in data]
        quality_scores = [item['evaluation_metrics']['overall_score'] for item in data]
        
        # Create timeline bars
        for i, (start, duration, score) in enumerate(zip(start_times, durations, quality_scores)):
            color = plt.cm.RdYlGn(score)  # Green for high quality, red for low
            ax.barh(i, duration, left=start, height=0.8, color=color, alpha=0.7)
            
            # Add chapter number
            ax.text(start + duration/2, i, f'Ch{i+1}', ha='center', va='center',
                   fontweight='bold', fontsize=10)
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Chapter')
        ax.set_title('Chapter Timeline with Quality Scores')
        ax.set_yticks(range(len(data)))
        ax.set_yticklabels([f'Ch{i+1}' for i in range(len(data))])
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, 
                                 norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Quality Score')
    
    def _plot_quality_vs_duration(self, data: List[Dict], ax: plt.Axes) -> None:
        """Deprecated: duration-based comparison removed per requirements."""
        ax.axis('off')
    
    def _plot_metrics_correlation(self, data: List[Dict], ax: plt.Axes) -> None:
        """Plot metrics correlation heatmap."""
        metrics = ['content_relevance', 'title_accuracy', 'summary_completeness',
                  'bert_score_precision', 'bert_score_recall', 'bert_score_f1',
                  'rouge_1_f1', 'rouge_2_f1', 'rouge_l_f1',
                  'boundary_accuracy', 'temporal_consistency', 'duration_appropriateness',
                  'redundancy_score', 'distinctiveness', 'search_relevance',
                  'keyword_coverage', 'navigation_utility', 'overall_score']
        
        # Create correlation matrix
        matrix = []
        for item in data:
            row = [item['evaluation_metrics'][metric] for metric in metrics]
            matrix.append(row)
        
        df = pd.DataFrame(matrix, columns=metrics)
        correlation_matrix = df.corr()
        
        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, ax=ax, fmt='.2f', cbar_kws={'shrink': 0.8})
        
        ax.set_title('Metrics Correlation Matrix')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)

    def _plot_advanced_metrics_summary(self, data: List[Dict], ax: plt.Axes) -> None:
        """Deprecated: kept for potential future use."""
        ax.axis('off')

    def _plot_summary_statistics(self, reports_data: Dict, ax: plt.Axes) -> None:
        """Plot summary statistics."""
        if not reports_data or 'summary_report' not in reports_data:
            ax.text(0.5, 0.5, 'No Summary Data Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Summary Statistics')
            return
        
        summary = reports_data['summary_report']
        
        # Create summary text
        stats_text = f"""
Total Chapters: {summary['total_chapters']}
Avg Duration: {summary['average_duration']:.1f}s
Total Duration: {summary['total_video_duration']:.1f}s

Quality Statistics:
Avg Score: {summary['quality_statistics']['average_quality_score']:.2f}
High Quality: {summary['quality_statistics']['high_quality_chapters']}
Low Quality: {summary['quality_statistics']['low_quality_chapters']}

Issues Summary:
Total Issues: {summary['issues_summary']['total_issues_detected']}
Chapters with Issues: {summary['issues_summary']['chapters_with_issues']}
        """
        
        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Summary Statistics')

    def _plot_recommendation_ratings(self, reports_data: Dict, ax: plt.Axes) -> None:
        """Plot search rating statistics."""
        if not reports_data or 'search_ratings_summary' not in reports_data:
            ax.text(0.5, 0.5, 'No Search Rating Data Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Search Ratings')
            return
        
        search_ratings_data = reports_data['search_ratings_summary']
        
        if not search_ratings_data:
            ax.text(0.5, 0.5, 'No Search Ratings Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Search Ratings')
            return
        
        # Collect search rating data
        query_stats = []
        for query_data in search_ratings_data:
            query = query_data['query']
            relevant_count = query_data['relevant_count']
            irrelevant_count = query_data['irrelevant_count']
            total_rated = relevant_count + irrelevant_count
            
            if total_rated > 0:
                relevance_ratio = relevant_count / total_rated
                query_stats.append({
                    'query': query[:20] + '...' if len(query) > 20 else query,
                    'relevance_ratio': relevance_ratio,
                    'total_rated': total_rated,
                    'relevant_count': relevant_count,
                    'irrelevant_count': irrelevant_count
                })
        
        if not query_stats:
            ax.text(0.5, 0.5, 'No Rated Search Results Available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Search Ratings')
            return
        
        # Create bar plot
        x_labels = [stat['query'] for stat in query_stats]
        relevance_ratios = [stat['relevance_ratio'] for stat in query_stats]
        total_rated = [stat['total_rated'] for stat in query_stats]
        
        bars = ax.bar(range(len(x_labels)), relevance_ratios, color='lightgreen', alpha=0.7)
        
        # Add total rated count on top of bars
        for i, (bar, total) in enumerate(zip(bars, total_rated)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'n={total}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Search Query')
        ax.set_ylabel('Relevance Ratio')
        ax.set_title('Search Result Relevance Ratings')
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        # Add average line
        avg_relevance = sum(relevance_ratios) / len(relevance_ratios)
        ax.axhline(y=avg_relevance, color='red', linestyle='--', alpha=0.7, 
                  label=f'Avg: {avg_relevance:.2f}')
        ax.legend()

def main():
    """Main function to create dashboard."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualization_dashboard.py <video_id> [--save]")
        print("Example: python visualization_dashboard.py P127jhj-8-Y --save")
        sys.exit(1)
    
    video_id = sys.argv[1]
    save_path = f"data/dashboard_{video_id}.png" if "--save" in sys.argv else None
    
    try:
        dashboard = ChapterEvaluationDashboard()
        dashboard.create_comprehensive_dashboard(video_id, save_path=save_path)
        print("Dashboard created successfully!")
        
    except Exception as e:
        print(f"Error creating dashboard: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
