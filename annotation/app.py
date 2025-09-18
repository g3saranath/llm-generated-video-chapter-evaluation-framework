#!/usr/bin/env python3
"""
Streamlit-based Manual Annotation Tool

Features:
- Auto-loads default input file `data/manual_reviews/manual_review_{video_id}.json`
- Displays one chapter at a time with prefilled manual review template
- Saves annotations back to the same file, namespaced by `reviewer_id`
- Computes aggregated human annotation scores and updates `data/outputs/reports_{video_id}.json`
- Regenerates evaluation dashboard `data/outputs/dashboard_{video_id}.png`

Run:
    streamlit run annotation/app.py -- --video_id P127jhj-8-Y --reviewer_id alice
"""

import os
import json
import argparse
from typing import Dict, Any, List

import streamlit as st
import pandas as pd
import time

# Allow importing from src/
import sys
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from visualization_dashboard import ChapterEvaluationDashboard

# Import evaluator components
import subprocess
import threading
import time


def parse_recommendation(rec_text: str) -> Dict[str, Any]:
    """Parse recommendation text to extract query, scores, title, and summary."""
    try:
        # Extract query
        query_start = rec_text.find("'") + 1
        query_end = rec_text.find("'", query_start)
        query = rec_text[query_start:query_end] if query_start > 0 and query_end > query_start else "Unknown Query"
        
        # Extract relevance and navigation scores
        relevance_match = rec_text.find("relevance ")
        navigation_match = rec_text.find("navigation ")
        
        relevance = 0.0
        navigation = 0.0
        
        if relevance_match != -1:
            rel_start = relevance_match + len("relevance ")
            rel_end = rec_text.find(",", rel_start)
            if rel_end == -1:
                rel_end = rec_text.find(")", rel_start)
            if rel_end != -1:
                relevance = float(rec_text[rel_start:rel_end])
        
        if navigation_match != -1:
            nav_start = navigation_match + len("navigation ")
            nav_end = rec_text.find(")", nav_start)
            if nav_end != -1:
                navigation = float(rec_text[nav_start:nav_end])
        
        # Extract title and summary
        title_start = rec_text.find("Title: ") + len("Title: ")
        summary_start = rec_text.find("Summary: ") + len("Summary: ")
        
        title = "Unknown Title"
        summary = "Unknown Summary"
        
        if title_start > len("Title: ") - 1:
            title_end = rec_text.find("\n", title_start)
            if title_end == -1:
                title_end = len(rec_text)
            title = rec_text[title_start:title_end].strip()
        
        if summary_start > len("Summary: ") - 1:
            summary = rec_text[summary_start:].strip()
        
        return {
            'query': query,
            'relevance': relevance,
            'navigation': navigation,
            'title': title,
            'summary': summary
        }
    except Exception as e:
        print(f"Error parsing recommendation: {e}")
        return {
            'query': "Parse Error",
            'relevance': 0.0,
            'navigation': 0.0,
            'title': "Parse Error",
            'summary': rec_text
        }


def load_manual_review_file(video_id: str) -> Dict[str, Any]:
    path = os.path.join(PROJECT_ROOT, 'data', 'manual_reviews', f'manual_review_{video_id}.json')
    if not os.path.exists(path):
        st.error(f"Manual review file not found: {path}. Generate it via `python evaluator.py {video_id} --manual-review`." )
        st.stop()
    
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        # Create backup of corrupted file
        import time
        backup_path = f"{path}.corrupted.{int(time.time())}"
        os.rename(path, backup_path)
        st.warning(f"Corrupted JSON file detected. Created backup at {backup_path}. Starting with fresh data.")
        return create_empty_manual_review_data(video_id)
    except Exception as e:
        st.error(f"Error loading manual review file: {e}")
        return create_empty_manual_review_data(video_id)

    # Ensure container for user annotations (may not exist in older files)
    if 'user_annotations' not in data:
        data['user_annotations'] = {}  # reviewer_id -> {chapter_index -> review}
    return data


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    elif hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def save_manual_review_file(video_id: str, data: Dict[str, Any]) -> None:
    path = os.path.join(PROJECT_ROOT, 'data', 'manual_reviews', f'manual_review_{video_id}.json')
    
    # Convert numpy types before saving
    serializable_data = convert_numpy_types(data)
    
    # Write to temporary file first, then rename to prevent corruption
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        # Atomic rename to prevent corruption
        os.rename(temp_path, path)
    except Exception as e:
        # Clean up temp file if something went wrong
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def aggregate_human_scores(manual_data: Dict[str, Any]) -> Dict[str, Any]:
    # Aggregate across all reviewers per chapter on the 1-5 scale metrics
    metrics = [
        'overall_quality_score',
        'content_accuracy_score',
        'title_appropriateness_score',
        'summary_quality_score',
        'search_relevance_score',
        'navigation_utility_score'
    ]
    user_annotations: Dict[str, Dict[str, Dict[str, Any]]] = manual_data.get('user_annotations', {})
    chapters_for_review: List[Dict[str, Any]] = manual_data.get('chapters_for_review', [])

    chapter_index_to_stats: Dict[int, Dict[str, Any]] = {}
    for chapter in chapters_for_review:
        idx = chapter['chapter_index']
        chapter_index_to_stats[idx] = {
            'chapter_index': idx,
            'num_reviews': 0,
            'averages': {m: None for m in metrics},
            'recommendation_ratings': {}
        }

    # Collect scores and recommendation ratings
    per_chapter_scores: Dict[int, Dict[str, List[float]]] = {c['chapter_index']: {m: [] for m in metrics} for c in chapters_for_review}
    per_chapter_rec_ratings: Dict[int, Dict[str, List[str]]] = {c['chapter_index']: {} for c in chapters_for_review}
    
    for reviewer_id, per_reviewer in user_annotations.items():
        for chapter_index_str, review in per_reviewer.items():
            try:
                idx = int(chapter_index_str)
            except ValueError:
                continue
            if idx not in per_chapter_scores:
                per_chapter_scores[idx] = {m: [] for m in metrics}
                per_chapter_rec_ratings[idx] = {}
            found_any = False
            for m in metrics:
                val = review.get(m)
                if isinstance(val, (int, float)):
                    per_chapter_scores[idx][m].append(float(val))
                    found_any = True
            if found_any:
                chapter_index_to_stats[idx]['num_reviews'] += 1
            
            # Collect recommendation ratings (both old chapter-based and new query-based)
            rec_ratings = review.get('recommendation_ratings', {})
            for rec_idx, rating in rec_ratings.items():
                if rec_idx not in per_chapter_rec_ratings[idx]:
                    per_chapter_rec_ratings[idx][rec_idx] = []
                if rating != "Not Rated":
                    per_chapter_rec_ratings[idx][rec_idx].append(rating)
            
            # Also collect query-based ratings
            query_ratings = review.get('query_ratings', {})
            for query, query_rec_ratings in query_ratings.items():
                for rating_key, rating in query_rec_ratings.items():
                    if rating != "Not Rated":
                        # Extract chapter index from rating key (format: "chapter_idx_rec_idx")
                        chapter_idx_str = rating_key.split('_')[0]
                        try:
                            chapter_idx = int(chapter_idx_str)
                            if chapter_idx not in per_chapter_rec_ratings:
                                per_chapter_rec_ratings[chapter_idx] = {}
                            if rating_key not in per_chapter_rec_ratings[chapter_idx]:
                                per_chapter_rec_ratings[chapter_idx][rating_key] = []
                            per_chapter_rec_ratings[chapter_idx][rating_key].append(rating)
                        except ValueError:
                            continue

    # Compute averages for metrics
    for idx, metric_map in per_chapter_scores.items():
        for m, vals in metric_map.items():
            if vals:
                chapter_index_to_stats[idx]['averages'][m] = sum(vals) / len(vals)

    # Compute recommendation rating statistics
    for idx, rec_ratings in per_chapter_rec_ratings.items():
        for rec_idx, ratings in rec_ratings.items():
            if ratings:
                relevant_count = ratings.count("Relevant")
                irrelevant_count = ratings.count("Irrelevant")
                total_rated = len(ratings)
                chapter_index_to_stats[idx]['recommendation_ratings'][rec_idx] = {
                    'relevant_count': relevant_count,
                    'irrelevant_count': irrelevant_count,
                    'total_rated': total_rated,
                    'relevance_ratio': relevant_count / total_rated if total_rated > 0 else 0.0
                }

    # Collect search ratings
    search_ratings_summary = {}
    for reviewer_id, per_reviewer in user_annotations.items():
        search_ratings = per_reviewer.get('search_ratings', {})
        for search_key, search_data in search_ratings.items():
            query = search_data.get('query', 'Unknown')
            ratings = search_data.get('ratings', {})
            
            if query not in search_ratings_summary:
                search_ratings_summary[query] = {
                    'query': query,
                    'total_searches': 0,
                    'total_results': 0,
                    'relevant_count': 0,
                    'irrelevant_count': 0,
                    'not_rated_count': 0
                }
            
            search_ratings_summary[query]['total_searches'] += 1
            search_ratings_summary[query]['total_results'] += len(ratings)
            
            for rating in ratings.values():
                if rating == "Relevant":
                    search_ratings_summary[query]['relevant_count'] += 1
                elif rating == "Irrelevant":
                    search_ratings_summary[query]['irrelevant_count'] += 1
                else:
                    search_ratings_summary[query]['not_rated_count'] += 1

    # Overall summary
    total_reviews = sum(v['num_reviews'] for v in chapter_index_to_stats.values())
    overall_avgs = {m: None for m in metrics}
    for m in metrics:
        vals = [v['averages'][m] for v in chapter_index_to_stats.values() if v['averages'][m] is not None]
        if vals:
            overall_avgs[m] = sum(vals) / len(vals)

    return {
        'manual_review_summary': {
            'total_reviews': total_reviews,
            'chapters_reviewed': sum(1 for v in chapter_index_to_stats.values() if v['num_reviews'] > 0),
            'average_scores': overall_avgs,
            'per_chapter': list(chapter_index_to_stats.values())
        },
        'search_ratings_summary': list(search_ratings_summary.values())
    }


def update_reports_with_manual_aggregates(video_id: str, aggregates: Dict[str, Any]) -> None:
    outputs_dir = os.path.join(PROJECT_ROOT, 'data', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    reports_path = os.path.join(outputs_dir, f'reports_{video_id}.json')
    reports: Dict[str, Any] = {}
    if os.path.exists(reports_path):
        with open(reports_path, 'r') as f:
            try:
                reports = json.load(f)
            except json.JSONDecodeError:
                reports = {}
    reports['manual_review_summary'] = aggregates.get('manual_review_summary', {})
    reports['search_ratings_summary'] = aggregates.get('search_ratings_summary', [])
    
    # Convert numpy types before saving
    serializable_reports = convert_numpy_types(reports)
    
    # Write to temporary file first, then rename to prevent corruption
    temp_path = f"{reports_path}.tmp"
    try:
        with open(temp_path, 'w') as f:
            json.dump(serializable_reports, f, indent=2)
        
        # Atomic rename to prevent corruption
        os.rename(temp_path, reports_path)
    except Exception as e:
        # Clean up temp file if something went wrong
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e


def regenerate_dashboard(video_id: str) -> None:
    outputs_dir = os.path.join(PROJECT_ROOT, 'data', 'outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    save_path = os.path.join(outputs_dir, f'dashboard_{video_id}.png')
    dashboard = ChapterEvaluationDashboard()
    dashboard.create_comprehensive_dashboard(video_id, save_path=save_path)


def get_session_data():
    """Get session data from sidebar or session state."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--video_id', type=str, default=None)
    parser.add_argument('--reviewer_id', type=str, default=None)
    args, _ = parser.parse_known_args()

    # Controls in sidebar
    with st.sidebar:
        st.header('Session')
        video_id = st.text_input('Video ID (e.g., 9vM4p9NN0Ts from https://www.youtube.com/watch?v=9vM4p9NN0Ts)', 
                                value=args.video_id or st.session_state.get('video_id', ''),
                                help='Press Enter to automatically run evaluator if no data exists')
        reviewer_id = st.text_input('Reviewer ID', value=args.reviewer_id or st.session_state.get('reviewer_id', ''))
        
        col1, col2 = st.columns(2)
        with col1:
            load_btn = st.button('🔄 Load Data', help='Manually load data for current video ID')
        with col2:
            run_evaluator_btn = st.button('⚙️ Run Evaluator', help='Run evaluator for current video ID')
        
        # Show status if video ID is provided
        if video_id:
            manual_review_path = os.path.join(PROJECT_ROOT, 'data', 'manual_reviews', f'manual_review_{video_id}.json')
            if os.path.exists(manual_review_path):
                st.success(f"✅ Data available for video {video_id}")
            else:
                st.warning(f"⚠️ No data found for video {video_id}")
                st.info("💡 Press ⚙️ Run Evaluator to auto-run evaluator")
        
        # Store in session state
        if video_id:
            st.session_state['video_id'] = video_id
        if reviewer_id:
            st.session_state['reviewer_id'] = reviewer_id

    return video_id, reviewer_id, load_btn, run_evaluator_btn


def run_evaluator_for_video(video_id):
    """Run the evaluator for a given video ID."""
    try:
        # Use absolute paths to avoid any path issues
        evaluator_path = os.path.abspath(os.path.join(PROJECT_ROOT, 'evaluator.py'))
        project_root_abs = os.path.abspath(PROJECT_ROOT)
        
        # # Debug: Show paths
        # st.write(f"**Debug:** PROJECT_ROOT = {PROJECT_ROOT}")
        # st.write(f"**Debug:** Absolute PROJECT_ROOT = {project_root_abs}")
        # st.write(f"**Debug:** Evaluator path = {evaluator_path}")
        # st.write(f"**Debug:** Evaluator exists = {os.path.exists(evaluator_path)}")
        
        if not os.path.exists(evaluator_path):
            return False, f"Evaluator file not found at: {evaluator_path}"
        
        # Run the evaluator with manual review enabled
        cmd = [
            'python', 
            evaluator_path, 
            video_id, 
            '--manual-review',
            '--dashboard'
        ]
        
        # st.write(f"**Debug:** Command = {cmd}")
        # st.write(f"**Debug:** Working directory = {project_root_abs}")
        
        # Run the evaluator
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=project_root_abs)
        
        if result.returncode == 0:
            # Show evaluator output in expandable section
            with st.expander("Evaluator Output", expanded=False):
                st.text(result.stdout)
            return True, "Evaluator completed successfully"
        else:
            # Show error output
            with st.expander("Evaluator Error Output", expanded=True):
                st.text(result.stderr)
            return False, f"Evaluator failed: {result.stderr}"
            
    except Exception as e:
        return False, f"Error running evaluator: {str(e)}"


def load_data(video_id, load_btn, run_evaluator_btn=False):
    """Load manual review data."""
    if not video_id:
        st.info('Enter a Video ID to begin.')
        return None, None

    # Clear session state if video ID changed
    video_id_changed = 'video_id' in st.session_state and st.session_state['video_id'] != video_id
    if video_id_changed:
        # Video ID changed, clear session state to force fresh load
        if 'manual_data' in st.session_state:
            del st.session_state['manual_data']
        st.session_state['video_id'] = video_id

    # Check if manual review file exists
    manual_review_path = os.path.join(PROJECT_ROOT, 'data', 'manual_reviews', f'manual_review_{video_id}.json')
    
    # Auto-run evaluator if new video ID entered and no data exists
    should_auto_run_evaluator = (
        video_id_changed and 
        not os.path.exists(manual_review_path) and 
        not load_btn and 
        not run_evaluator_btn
    )
    
    if should_auto_run_evaluator:
        st.info(f"🔄 New video ID detected: {video_id}. No data found, running evaluator...")
        with st.spinner("Running evaluator to auto-generate chapters powered by o4-mini, and evaluation..."):
            success, message = run_evaluator_for_video(video_id)
            
            if success:
                st.success("Evaluator completed successfully! Loading data...")
            else:
                st.error(f"Evaluator failed: {message}")
                st.info("You can try running the evaluator manually or check the video ID.")
                return None, None
        
        # Clear session state to force fresh load after evaluator
        if 'manual_data' in st.session_state:
            del st.session_state['manual_data']
        
        # Force page refresh to load new data
        st.rerun()
    
    # Handle Run Evaluator button click immediately
    if run_evaluator_btn:
        # Set a flag to prevent multiple runs
        if 'evaluator_running' not in st.session_state or not st.session_state['evaluator_running']:
            st.session_state['evaluator_running'] = True
            
            st.info(f"Starting evaluator for video {video_id}...")
            with st.spinner("Running evaluator to generate chapters and evaluation data..."):
                success, message = run_evaluator_for_video(video_id)
                
                if success:
                    st.success(" Evaluator completed successfully! Loading data...")
                else:
                    st.error(f" Evaluator failed: {message}")
                    st.info("You can try running the evaluator manually or check the video ID.")
                    st.session_state['evaluator_running'] = False
                    return None, None
            
            # Clear session state to force fresh load after evaluator
            if 'manual_data' in st.session_state:
                del st.session_state['manual_data']
            
            # Reset the flag
            st.session_state['evaluator_running'] = False
            
            # Force page refresh to load new data
            st.rerun()
    
    # Auto-load if data exists and no manual action requested
    should_auto_load = (
        os.path.exists(manual_review_path) and 
        not load_btn and 
        not run_evaluator_btn and 
        'manual_data' not in st.session_state
    )
    
    # Load data (either on button, auto-load, or on initial render if present)
    if load_btn or should_auto_load or 'manual_data' not in st.session_state:
        if should_auto_load:
            st.info(f"📁 Auto-loading existing data for video {video_id}...")
        
        if not os.path.exists(manual_review_path):
            # No manual review file exists, run the evaluator
            st.info(f"Manual review file not found for video {video_id}. Running evaluator...")
            
            with st.spinner("Running evaluator to generate chapters and evaluation data..."):
                success, message = run_evaluator_for_video(video_id)
                
                if success:
                    st.success("Evaluator completed successfully! Loading data...")
                else:
                    st.error(f"Evaluator failed: {message}")
                    st.info("You can try running the evaluator manually or check the video ID.")
                    return None, None
        
        # Now load the manual review data
        manual_data = load_manual_review_file(video_id)
        st.session_state['manual_data'] = manual_data
        st.session_state['video_id'] = video_id

    if 'manual_data' not in st.session_state:
        return None, None

    manual_data = st.session_state['manual_data']
    chapters_for_review = manual_data.get('chapters_for_review', [])
    user_annotations = manual_data.get('user_annotations', {})
    
    return manual_data, chapters_for_review, user_annotations


def render_chapter_navigation(chapters_for_review, user_annotations, reviewer_id):
    """Render chapter navigation with progress indicator."""
    idx_options = [c['chapter_index'] for c in chapters_for_review]
    if not idx_options:
        st.warning('No chapters found in the manual review file.')
        return None, None
    
    # Initialize session state for current chapter index
    if 'current_chapter_idx' not in st.session_state:
        st.session_state['current_chapter_idx'] = 0
    
    # Ensure current chapter index is within bounds
    if st.session_state['current_chapter_idx'] >= len(idx_options):
        st.session_state['current_chapter_idx'] = 0
    
    # Get current chapter first
    chapter_idx = idx_options[st.session_state['current_chapter_idx']]
    chapter = next(c for c in chapters_for_review if c['chapter_index'] == chapter_idx)
    
    # Show annotation progress
    if reviewer_id:
        annotated_chapters = set()
        for chapter_data in chapters_for_review:
            current_chapter_idx = chapter_data['chapter_index']
            if str(current_chapter_idx) in user_annotations.get(reviewer_id, {}):
                annotated_chapters.add(current_chapter_idx)
        
        total_chapters = len(chapters_for_review)
        annotated_count = len(annotated_chapters)
        st.write(f"Progress: {annotated_count}/{total_chapters} chapters annotated")
        
        # Show progress bar
        progress = annotated_count / total_chapters if total_chapters > 0 else 0
        st.progress(progress)
    
    # Chapter selector dropdown
    chapter_options = []
    for i, c in enumerate(chapters_for_review):
        current_chapter_idx = c['chapter_index']
        status = "✓" if reviewer_id and str(current_chapter_idx) in user_annotations.get(reviewer_id, {}) else "○"
        chapter_options.append(f"{status} Chapter {i+1}: {c['chapter_info']['title']}")
    
    selected_chapter = st.selectbox(
        "Select Chapter:",
        options=chapter_options,
        index=st.session_state['current_chapter_idx']
    )
    
    # Update chapter index if selection changed
    if selected_chapter != chapter_options[st.session_state['current_chapter_idx']]:
        st.session_state['current_chapter_idx'] = chapter_options.index(selected_chapter)
        st.rerun()
    
    # Navigation buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    
    with col1:
        if st.button('◀ Previous', disabled=st.session_state['current_chapter_idx'] == 0):
            st.session_state['current_chapter_idx'] = max(0, st.session_state['current_chapter_idx'] - 1)
            st.rerun()
    
    with col2:
        if st.button('Next ▶', disabled=st.session_state['current_chapter_idx'] >= len(idx_options) - 1):
            st.session_state['current_chapter_idx'] = min(len(idx_options) - 1, st.session_state['current_chapter_idx'] + 1)
            st.rerun()
    
    with col3:
        if st.button('Reset to First'):
            st.session_state['current_chapter_idx'] = 0
            st.rerun()
    
    with col4:
        st.write(f"Chapter {st.session_state['current_chapter_idx'] + 1} of {len(idx_options)}: {chapter['chapter_info']['title']}")
    
    return chapter_idx, chapter


def chapter_review_page():
    """Chapter Review page."""
    st.title('Chapter Review')
    
    video_id, reviewer_id, load_btn, run_evaluator_btn = get_session_data()
    result = load_data(video_id, load_btn, run_evaluator_btn)
    if result[0] is None:
        return
    
    manual_data, chapters_for_review, user_annotations = result
    
    if reviewer_id and reviewer_id not in user_annotations:
        user_annotations[reviewer_id] = {}
        manual_data['user_annotations'] = user_annotations

    # Chapter navigation
    result = render_chapter_navigation(chapters_for_review, user_annotations, reviewer_id)
    if result[0] is None:
        return
    chapter_idx, chapter = result
    
    # Debug: Show current chapter being processed
    st.write(f"**Debug:** Currently processing Chapter {chapter_idx}: {chapter['chapter_info']['title']}")

    # Show context
    with st.expander('Chapter Info', expanded=True):
        st.write(chapter['chapter_info'])
        st.write({'automated_evaluation': chapter.get('automated_evaluation', {})})
        st.write({'detected_issues': chapter.get('detected_issues', [])})
        st.write({'automated_recommendations': chapter.get('automated_recommendations', [])})

    # Prefill defaults
    template = {}
    existing = {}
    if reviewer_id:
        existing = user_annotations.get(reviewer_id, {}).get(str(chapter_idx), {})

    st.subheader('Your Review')
    col1, col2, col3 = st.columns(3)
    with col1:
        overall_quality_score = st.number_input('Overall Quality (1-5)', min_value=1.0, max_value=5.0, step=1.0,
                                                value=float(existing.get('overall_quality_score', template.get('overall_quality_score') or 3)))
        content_accuracy_score = st.number_input('Content Accuracy (1-5)', min_value=1.0, max_value=5.0, step=1.0,
                                                 value=float(existing.get('content_accuracy_score', template.get('content_accuracy_score') or 3)))
    with col2:
        title_appropriateness_score = st.number_input('Title Appropriateness (1-5)', min_value=1.0, max_value=5.0, step=1.0,
                                                      value=float(existing.get('title_appropriateness_score', template.get('title_appropriateness_score') or 3)))
        summary_quality_score = st.number_input('Summary Quality (1-5)', min_value=1.0, max_value=5.0, step=1.0,
                                                value=float(existing.get('summary_quality_score', template.get('summary_quality_score') or 3)))
    with col3:
        search_relevance_score = st.number_input('Search Relevance (1-5)', min_value=1.0, max_value=5.0, step=1.0,
                                                 value=float(existing.get('search_relevance_score', template.get('search_relevance_score') or 3)))
        navigation_utility_score = st.number_input('Navigation Utility (1-5)', min_value=1.0, max_value=5.0, step=1.0,
                                                   value=float(existing.get('navigation_utility_score', template.get('navigation_utility_score') or 3)))

    issues_identified = st.tags(label='Issues Identified',
                                text='Type and press enter',
                                value=existing.get('issues_identified', [])) if hasattr(st, 'tags') else st.text_area('Issues Identified (comma-separated)', value=','.join(existing.get('issues_identified', [])))

    recommendations = st.tags(label='Recommendations',
                              text='Type and press enter',
                              value=existing.get('recommendations', [])) if hasattr(st, 'tags') else st.text_area('Recommendations (comma-separated)', value=','.join(existing.get('recommendations', [])))

    confidence_level = st.slider('Confidence (0-1)', min_value=0.0, max_value=1.0, step=0.05,
                                 value=float(existing.get('confidence_level', 0.8)))

    # Save button and logic
    col1, col2 = st.columns([1, 4])
    with col1:
        save_clicked = st.button('Save Review', type='primary', disabled=not bool(reviewer_id))
    with col2:
        if not reviewer_id:
            st.info('Enter a Reviewer ID in the sidebar to enable saving.')

    if save_clicked:
        # Show saving status
        with st.spinner("Saving annotation..."):
            # Normalize tags fallback
            if isinstance(issues_identified, str):
                issues_list = [s.strip() for s in issues_identified.split(',') if s.strip()]
            else:
                issues_list = list(issues_identified)
            if isinstance(recommendations, str):
                recs_list = [s.strip() for s in recommendations.split(',') if s.strip()]
            else:
                recs_list = list(recommendations)

        # Save under reviewer namespace
        if reviewer_id not in user_annotations:
            user_annotations[reviewer_id] = {}
        user_annotations[reviewer_id][str(chapter_idx)] = {
            'reviewer_id': reviewer_id,
            'chapter_index': chapter_idx,
            'overall_quality_score': float(overall_quality_score),
            'content_accuracy_score': float(content_accuracy_score),
            'title_appropriateness_score': float(title_appropriateness_score),
            'summary_quality_score': float(summary_quality_score),
            'search_relevance_score': float(search_relevance_score),
            'navigation_utility_score': float(navigation_utility_score),
            'issues_identified': issues_list,
            'recommendations': recs_list,
            'confidence_level': float(confidence_level)
        }

        manual_data['user_annotations'] = user_annotations
        
        # Debug: Show what's being saved
        st.write(f"**Debug:** Saving annotation for chapter {chapter_idx} by reviewer {reviewer_id}")
        
        save_manual_review_file(video_id, manual_data)
        
        # Update session state with new data
        st.session_state['manual_data'] = manual_data
        
        # Reload data from file to ensure consistency
        manual_data = load_manual_review_file(video_id)
        st.session_state['manual_data'] = manual_data

        # Aggregate and update reports
        aggregates = aggregate_human_scores(manual_data)
        update_reports_with_manual_aggregates(video_id, aggregates)

        # Regenerate dashboard immediately
        try:
            regenerate_dashboard(video_id)
            st.success('✅ Saved! Review data updated and dashboard regenerated.')
            st.balloons()  # Visual feedback for successful save
        except Exception as e:
            st.success('✅ Saved! Review data updated.')
            st.warning(f'⚠️ Dashboard regeneration failed: {e}')
            st.info('You can manually regenerate the dashboard from the Dashboard page.')
        
        st.rerun()  # Refresh the page to show updated progress and checkmarks


def search_and_rate_page():
    """Search and Rate page - Search chapters by query and rate results."""
    st.title('Search & Rate Chapters')
    
    video_id, reviewer_id, load_btn, run_evaluator_btn = get_session_data()
    result = load_data(video_id, load_btn, run_evaluator_btn)
    if result[0] is None:
        return
    
    manual_data, chapters_for_review, user_annotations = result
    
    if reviewer_id and reviewer_id not in user_annotations:
        user_annotations[reviewer_id] = {}
        manual_data['user_annotations'] = user_annotations

    # Debug info
    st.write(f"**Debug Info:** Loaded {len(chapters_for_review)} chapters for review")
    
    # Show annotation progress
    if reviewer_id and user_annotations:
        total_annotations = sum(len(chapters) for chapters in user_annotations.values())
        st.write(f"**Annotations:** {total_annotations} total annotations by {len(user_annotations)} reviewers")
    
    # Add refresh button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button('🔄 Refresh Data'):
            # Clear session state to force reload
            if 'manual_data' in st.session_state:
                del st.session_state['manual_data']
            st.rerun()
    with col2:
        st.info('Click "Refresh Data" if annotations are not showing up properly.')
    
    # Test search functionality
    if st.button("Test Search (Simple)"):
        test_results = simple_text_search(chapters_for_review, "transformers", 0.0, 0.0, 3)
        st.write(f"Test search found {len(test_results)} results")
        if test_results:
            for i, result in enumerate(test_results):
                st.write(f"{i+1}. Chapter {result['chapter_idx']}: {result['chapter_info']['title']} (Score: {result['combined_score']:.3f})")

    # Search interface
    st.subheader('Search Chapters')
    
    # Search query input
    search_query = st.text_input(
        "Enter search query:",
        placeholder="e.g., transformers, deep learning, neural networks",
        key="search_query"
    )
    
    # Search parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_relevance = st.slider(
            "Minimum Relevance Score:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum relevance score for search results"
        )
    
    with col2:
        min_navigation = st.slider(
            "Minimum Navigation Score:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Minimum navigation utility score for search results"
        )
    
    with col3:
        max_results = st.slider(
            "Maximum Results:",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Maximum number of results to return"
        )
    
    # Search buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        search_clicked = st.button('Search Chapters', type='primary')
    with col2:
        clear_clicked = st.button('Clear Results', type='secondary')
    
    # Initialize session state for search results
    if 'search_results' not in st.session_state:
        st.session_state.search_results = []
    if 'current_search_query' not in st.session_state:
        st.session_state.current_search_query = ""
    
    # Handle clear results
    if clear_clicked:
        st.session_state.search_results = []
        st.session_state.current_search_query = ""
        st.rerun()
    
    # Perform search
    if search_clicked and search_query:
        with st.spinner("Searching chapters..."):
            search_results = perform_chapter_search(
                chapters_for_review, 
                search_query, 
                min_relevance, 
                min_navigation, 
                max_results
            )
        
        # Store search results in session state
        st.session_state.search_results = search_results
        st.session_state.current_search_query = search_query
        
        if search_results:
            st.subheader(f'Search Results for "{search_query}"')
            st.write(f"Found {len(search_results)} relevant chapters:")
            
            # Display search results for rating
            display_search_results_for_rating(search_results, user_annotations, reviewer_id, video_id, manual_data)
        else:
            st.warning(f"No chapters found matching your criteria for query: '{search_query}'")
            st.info("Try lowering the minimum scores or using different search terms.")
            
            # Show some debugging info
            with st.expander("Debug Info"):
                st.write(f"""
                **Search Parameters:**
                - Query: {search_query}
                - Min Relevance: {min_relevance}
                - Min Navigation: {min_navigation}
                - Max Results: {max_results}
                - Total Chapters: {len(chapters_for_review)}
                """)
                
                # Show sample chapter data
                if chapters_for_review:
                    sample_chapter = chapters_for_review[0]
                    st.write("**Sample Chapter Data:**")
                    st.json({
                        "chapter_index": sample_chapter.get('chapter_index'),
                        "title": sample_chapter.get('chapter_info', {}).get('title', 'No title'),
                        "summary": sample_chapter.get('chapter_info', {}).get('summary', 'No summary')[:100] + '...'
                    })
    
    # Display existing search results if available
    elif st.session_state.search_results and st.session_state.current_search_query:
        st.subheader(f'Search Results for "{st.session_state.current_search_query}"')
        st.write(f"Found {len(st.session_state.search_results)} relevant chapters:")
        
        # Display search results for rating
        display_search_results_for_rating(st.session_state.search_results, user_annotations, reviewer_id, video_id, manual_data)
    
    elif search_clicked and not search_query:
        st.warning("Please enter a search query.")
    
    # Show search history and ratings
    if reviewer_id:
        show_search_history(user_annotations, reviewer_id)


def perform_chapter_search(chapters, query, min_relevance, min_navigation, max_results):
    """Search chapters based on query and score thresholds."""
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        # Load the model
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Encode the query
        query_embedding = model.encode([query])
        
        search_results = []
        
        for chapter in chapters:
            try:
                chapter_idx = chapter['chapter_index']
                chapter_info = chapter['chapter_info']
                title = chapter_info.get('title', '')
                summary = chapter_info.get('summary', '')
                
                # Skip if no content
                if not title and not summary:
                    continue
                
                # Combine title and summary for search
                search_text = f"{title} {summary}".strip()
                
                if not search_text:
                    continue
                
                # Encode the chapter content
                chapter_embedding = model.encode([search_text])
                
                # Calculate relevance score (query vs chapter content)
                relevance_score = float(cosine_similarity(query_embedding, chapter_embedding)[0][0])
                
                # Calculate navigation utility (query vs title + summary)
                nav_text = f"{title} {summary}".strip()
                nav_embedding = model.encode([nav_text])
                navigation_score = float(cosine_similarity(query_embedding, nav_embedding)[0][0])
                
                # Check if chapter meets criteria
                if relevance_score >= min_relevance and navigation_score >= min_navigation:
                    search_results.append({
                        'chapter_idx': chapter_idx,
                        'chapter_info': chapter_info,
                        'relevance_score': relevance_score,
                        'navigation_score': navigation_score,
                        'combined_score': (relevance_score + navigation_score) / 2
                    })
            except Exception as chapter_error:
                st.warning(f"Error processing chapter {chapter.get('chapter_index', 'unknown')}: {str(chapter_error)}")
                continue
        
        # Sort by combined score and return top results
        search_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return search_results[:max_results]
        
    except ImportError as e:
        st.warning(f"Semantic search unavailable: {str(e)}")
        st.info("Falling back to simple text matching...")
        return simple_text_search(chapters, query, min_relevance, min_navigation, max_results)
    except Exception as e:
        st.warning(f"Semantic search failed: {str(e)}")
        st.info("Falling back to simple text matching...")
        return simple_text_search(chapters, query, min_relevance, min_navigation, max_results)


def simple_text_search(chapters, query, min_relevance, min_navigation, max_results):
    """Fallback simple text search when semantic search fails."""
    query_lower = query.lower()
    search_results = []
    
    for chapter in chapters:
        try:
            chapter_idx = chapter['chapter_index']
            chapter_info = chapter['chapter_info']
            title = chapter_info.get('title', '').lower()
            summary = chapter_info.get('summary', '').lower()
            
            # Simple text matching
            title_matches = query_lower in title
            summary_matches = query_lower in summary
            
            # Calculate simple scores
            relevance_score = 0.0
            if title_matches:
                relevance_score += 0.7
            if summary_matches:
                relevance_score += 0.3
            
            navigation_score = 0.0
            if title_matches:
                navigation_score += 0.8
            if summary_matches:
                navigation_score += 0.2
            
            # Normalize scores and convert to Python float
            relevance_score = float(min(relevance_score, 1.0))
            navigation_score = float(min(navigation_score, 1.0))
            
            # Check if chapter meets criteria
            if relevance_score >= min_relevance and navigation_score >= min_navigation:
                search_results.append({
                    'chapter_idx': chapter_idx,
                    'chapter_info': chapter_info,
                    'relevance_score': relevance_score,
                    'navigation_score': navigation_score,
                    'combined_score': (relevance_score + navigation_score) / 2
                })
        except Exception as chapter_error:
            continue
    
    # Sort by combined score and return top results
    search_results.sort(key=lambda x: x['combined_score'], reverse=True)
    return search_results[:max_results]


def display_search_results_for_rating(search_results, user_annotations, reviewer_id, video_id, manual_data):
    """Display search results for rating by the annotator."""
    if not search_results:
        return
    
    # Get existing search ratings for this specific search
    existing_ratings = {}
    if reviewer_id:
        search_ratings_data = user_annotations.get(reviewer_id, {}).get('search_ratings', {})
        # Get the most recent search ratings
        if search_ratings_data:
            latest_search = max(search_ratings_data.keys(), key=lambda k: search_ratings_data[k].get('timestamp', ''))
            existing_ratings = search_ratings_data[latest_search].get('ratings', {})
    
    st.write("Rate each search result as relevant or irrelevant:")
    
    search_ratings = {}
    
    for i, result in enumerate(search_results):
        chapter_idx = result['chapter_idx']
        chapter_info = result['chapter_info']
        relevance_score = result['relevance_score']
        navigation_score = result['navigation_score']
        combined_score = result['combined_score']
        
        # Create a container for each search result
        with st.container():
            st.markdown(f"**Search Result {i+1}:**")
            
            # Display search result details in columns
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"**Chapter {chapter_idx + 1}:** {chapter_info['title']}")
                st.markdown(f"**Summary:** {chapter_info['summary'][:200]}{'...' if len(chapter_info['summary']) > 200 else ''}")
                st.markdown(f"**Duration:** {chapter_info['start_timestamp']} - {chapter_info['end_timestamp']}")
            
            with col2:
                st.metric("Relevance", f"{relevance_score:.3f}")
                st.metric("Navigation", f"{navigation_score:.3f}")
            
            with col3:
                st.metric("Combined", f"{combined_score:.3f}")
            
            # Rating selection
            rating_options = ["Relevant", "Irrelevant", "Not Rated"]
            default_idx = 2  # "Not Rated" by default
            
            # Check if we have existing rating
            rating_key = f"{chapter_idx}_{i}"
            existing_rating = existing_ratings.get(rating_key, "Not Rated")
            if existing_rating in rating_options:
                default_idx = rating_options.index(existing_rating)
            
            search_ratings[rating_key] = st.selectbox(
                f"Rate this search result:",
                options=rating_options,
                index=default_idx,
                key=f"search_rating_{chapter_idx}_{i}"
            )
            
            st.markdown("---")
    
    # Save button for search ratings
    col1, col2 = st.columns([1, 4])
    with col1:
        save_clicked = st.button('Save Search Ratings', type='primary', disabled=not bool(reviewer_id))
    with col2:
        if not reviewer_id:
            st.info('Enter a Reviewer ID in the sidebar to enable saving.')

    if save_clicked:
        # Show saving status
        with st.spinner("Saving search ratings..."):
            # Save search ratings
            if reviewer_id not in user_annotations:
                user_annotations[reviewer_id] = {}
            if 'search_ratings' not in user_annotations[reviewer_id]:
                user_annotations[reviewer_id]['search_ratings'] = {}
        
        # Store search query and results metadata
        search_metadata = {
            'query': st.session_state.get('current_search_query', ''),
            'timestamp': str(pd.Timestamp.now()),
            'num_results': len(search_results),
            'ratings': search_ratings,
            'search_results': search_results  # Store the search results (numpy types will be converted by save function)
        }
        
        search_key = f"search_{len(user_annotations[reviewer_id].get('search_ratings', {}))}"
        user_annotations[reviewer_id]['search_ratings'][search_key] = search_metadata

        manual_data['user_annotations'] = user_annotations
        save_manual_review_file(video_id, manual_data)
        
        # Update session state with new data
        st.session_state['manual_data'] = manual_data

        # Aggregate and update reports
        aggregates = aggregate_human_scores(manual_data)
        update_reports_with_manual_aggregates(video_id, aggregates)

        # Regenerate dashboard immediately
        try:
            regenerate_dashboard(video_id)
            st.success('✅ Saved! Search ratings updated and dashboard regenerated.')
            st.balloons()  # Visual feedback for successful save
        except Exception as e:
            st.success('✅ Saved! Search ratings updated.')
            st.warning(f'⚠️ Dashboard regeneration failed: {e}')
            st.info('You can manually regenerate the dashboard from the Dashboard page.')
        
        st.rerun()  # Refresh the page to show updated ratings


def show_search_history(user_annotations, reviewer_id):
    """Show search history and ratings for the reviewer."""
    if reviewer_id not in user_annotations:
        return
    
    search_ratings = user_annotations[reviewer_id].get('search_ratings', {})
    if not search_ratings:
        return
    
    st.subheader('Search History')
    
    for search_key, search_data in search_ratings.items():
        with st.expander(f"Search: '{search_data['query']}' ({search_data['timestamp']})"):
            st.write(f"**Query:** {search_data['query']}")
            st.write(f"**Results Found:** {search_data['num_results']}")
            st.write(f"**Search Time:** {search_data['timestamp']}")
            
            # Show ratings summary
            ratings = search_data.get('ratings', {})
            relevant_count = sum(1 for rating in ratings.values() if rating == "Relevant")
            irrelevant_count = sum(1 for rating in ratings.values() if rating == "Irrelevant")
            not_rated_count = sum(1 for rating in ratings.values() if rating == "Not Rated")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Relevant", relevant_count)
            with col2:
                st.metric("Irrelevant", irrelevant_count)
            with col3:
                st.metric("Not Rated", not_rated_count)


def dashboard_page():
    """Dashboard page."""
    st.title('Evaluation Dashboard')
    
    video_id, reviewer_id, load_btn, run_evaluator_btn = get_session_data()
    
    if not video_id:
        st.info('Enter a Video ID to view dashboard.')
        return
    
    # Check if manual review data exists and regenerate dashboard if needed
    manual_review_file = f"data/manual_reviews/manual_review_{video_id}.json"
    dashboard_path = f"data/outputs/dashboard_{video_id}.png"
    
    # Check if we need to regenerate the dashboard
    should_regenerate = False
    if os.path.exists(manual_review_file):
        # Check if manual review file is newer than dashboard
        if not os.path.exists(dashboard_path):
            should_regenerate = True
        else:
            manual_time = os.path.getmtime(manual_review_file)
            dashboard_time = os.path.getmtime(dashboard_path)
            if manual_time > dashboard_time:
                should_regenerate = True
    
    # Regenerate dashboard if needed
    if should_regenerate:
        with st.spinner("Regenerating dashboard with latest data..."):
            try:
                regenerate_dashboard(video_id)
                st.success('Dashboard updated with latest manual review data!')
            except Exception as e:
                st.error(f'Failed to regenerate dashboard: {e}')
    
    # Display dashboard
    if os.path.exists(dashboard_path):
        st.image(dashboard_path, use_container_width=True)
        
        # Add manual regenerate button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button('Regenerate Dashboard'):
                try:
                    regenerate_dashboard(video_id)
                    st.success('Dashboard regenerated and saved!')
                    st.rerun()
                except Exception as e:
                    st.error(f'Failed to regenerate dashboard: {e}')
        with col2:
            st.info('Dashboard shows the latest evaluation and manual review data.')
    else:
        st.warning(f'No dashboard found for video {video_id}.')
        
        if st.button('Generate Dashboard'):
            try:
                regenerate_dashboard(video_id)
                st.success('Dashboard generated and saved!')
                st.rerun()
            except Exception as e:
                st.error(f'Failed to generate dashboard: {e}')


def main() -> None:
    st.set_page_config(page_title='Manual Annotation Tool', layout='wide')
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page",
        ["Chapter Review", "Search & Rate", "Dashboard"]
    )
    
    if page == "Chapter Review":
        chapter_review_page()
    elif page == "Search & Rate":
        search_and_rate_page()
    elif page == "Dashboard":
        dashboard_page()


if __name__ == '__main__':
    main()


 