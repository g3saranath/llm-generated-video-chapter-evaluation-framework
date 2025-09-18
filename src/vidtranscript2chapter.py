import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import openai
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")

@dataclass
class Chapter:
    title: str
    summary: str
    start_time: float
    end_time: float
    duration: float
    start_timestamp: str  # HH:MM:SS format
    end_timestamp: str    # HH:MM:SS format
    youtube_timestamp: str  # YouTube URL with timestamp

class VideoChapterGenerator:
    def __init__(self, openai_api_key: Optional[str] = None):
        """Initialize the chapter generator with OpenAI API key."""
        self.client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
        self.formatter = TextFormatter()
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to HH:MM:SS format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def _create_youtube_timestamp(self, video_id: str, seconds: float) -> str:
        """Create YouTube URL with timestamp."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            timestamp = f"{hours}h{minutes}m{secs}s"
        else:
            timestamp = f"{minutes}m{secs}s"
        
        return f"https://youtube.com/watch?v={video_id}&t={int(seconds)}s"
        
    def extract_transcript(self, video_id: str, language: str = 'en') -> str:
        """Extract transcript from YouTube video."""
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_data = ytt_api.fetch(video_id, languages=[language])
            
            # Convert FetchedTranscriptSnippet objects to dictionaries
            transcript_dicts = []
            for snippet in transcript_data:
                transcript_dicts.append({
                    'text': snippet.text,
                    'start': snippet.start,
                    'duration': snippet.duration
                })
            
            # Format transcript as clean text
            formatted_text = self.formatter.format_transcript(transcript_data)
            return formatted_text, transcript_dicts
        except Exception as e:
            raise Exception(f"Failed to extract transcript: {str(e)}")
    
    def segment_transcript(self, transcript_data: List[Dict], max_duration: float = 300.0) -> List[Dict]:
        """Segment transcript into logical chunks for chapter generation."""
        segments = []
        current_segment = []
        current_duration = 0.0
        
        for entry in transcript_data:
            current_segment.append(entry)
            current_duration += entry['duration']
            
            # Create new segment if duration exceeds threshold or natural break points
            if (current_duration >= max_duration or 
                self._is_natural_breakpoint(entry['text'])):
                
                if current_segment:
                    segments.append({
                        'text': ' '.join([item['text'] for item in current_segment]),
                        'start_time': current_segment[0]['start'],
                        'end_time': current_segment[-1]['start'] + current_segment[-1]['duration'],
                        'duration': current_duration
                    })
                    current_segment = []
                    current_duration = 0.0
        
        # Add remaining segment
        if current_segment:
            segments.append({
                'text': ' '.join([item['text'] for item in current_segment]),
                'start_time': current_segment[0]['start'],
                'end_time': current_segment[-1]['start'] + current_segment[-1]['duration'],
                'duration': current_duration
            })
        
        return segments
    
    def _is_natural_breakpoint(self, text: str) -> bool:
        """Identify natural breakpoints in transcript."""
        # Look for sentence endings, pauses, or topic indicators
        breakpoint_indicators = [
            r'\.\s*$',  # End of sentence
            r'\?\s*$',  # Question
            r'!\s*$',   # Exclamation
            r'\.\.\.',  # Pause/ellipsis
            r'\b(now|next|then|after|before|first|second|third|finally|in conclusion)\b'
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in breakpoint_indicators)
    
    def generate_chapters(self, segments: List[Dict], video_title: str = "", video_id: str = "") -> List[Chapter]:
        """Generate chapters using OpenAI API."""
        try:
            # Create prompt for chapter generation
            prompt = self._create_chapter_prompt(segments, video_title)
            
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are an expert video content analyst. Generate meaningful chapters for video content based on transcript segments."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=16384
            )
            
            # Parse the response
            chapters_data = self._parse_chapter_response(response.choices[0].message.content)
            
            # Create Chapter objects with formatted timestamps
            chapters = []
            for i, chapter_data in enumerate(chapters_data):
                start_time = chapter_data.get('start_time', 0)
                end_time = chapter_data.get('end_time', 0)
                duration = chapter_data.get('duration', 0)
                
                chapter = Chapter(
                    title=chapter_data.get('title', f'Chapter {i+1}'),
                    summary=chapter_data.get('summary', ''),
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration,
                    start_timestamp=self._format_timestamp(start_time),
                    end_timestamp=self._format_timestamp(end_time),
                    youtube_timestamp=self._create_youtube_timestamp(video_id, start_time) if video_id else ""
                )
                chapters.append(chapter)
            
            return chapters
            
        except Exception as e:
            raise Exception(f"Failed to generate chapters: {str(e)}")
    
    def _create_chapter_prompt(self, segments: List[Dict], video_title: str) -> str:
        """Create prompt for chapter generation."""
        prompt = f"""
        Analyze the following video transcript segments and generate meaningful chapters.
        Video Title: {video_title}
        
        For each segment, provide:
        1. A descriptive title 
        2. A summary of the chapter explaining key points and insights
        3. Start and end timestamps
        
        Transcript Segments:
        """
        
        for i, segment in enumerate(segments):
            prompt += f"\nSegment {i+1} (Start: {segment['start_time']:.1f}s, End: {segment['end_time']:.1f}s):\n{segment['text'][:500]}...\n"
        
        prompt += """
        
        Return your response as a JSON array with this structure:
        [
            {
                "title": "Chapter Title",
                "summary": "Summary of the chapter explaining key points and insights",
                "start_time": <start_time>,
                "end_time": <end_time>,
                "duration": <duration>
            }
        ]
        
        Guidelines:
        - Create meaningful chapters based on content flow. Only create a new chapter if there is a distinct topic change.
        - Titles should be engaging and descriptive
        - Summaries should capture the main points
        - Ensure timestamps are accurate
        - Merge very short segments if they don't warrant separate chapters
        """
        
        return prompt
    
    def _parse_chapter_response(self, response: str) -> List[Dict]:
        """Parse OpenAI response to extract chapter data."""
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                raise ValueError("No valid JSON found in response")
        except Exception as e:
            raise Exception(f"Failed to parse chapter response: {str(e)}")
    
    def format_chapters_output(self, chapters: List[Chapter]) -> str:
        """Format chapters for display or export."""
        output = "Video Chapters:\n" + "="*60 + "\n\n"
        
        for i, chapter in enumerate(chapters, 1):
            output += f"{i}. {chapter.title}\n"
            output += f"   Time: {chapter.start_timestamp} - {chapter.end_timestamp}\n"
            output += f"   Duration: {chapter.duration:.1f}s\n"
            if chapter.youtube_timestamp:
                output += f"   YouTube Link: {chapter.youtube_timestamp}\n"
            output += f"   Summary: {chapter.summary}\n\n"
        
        return output

def main():
    """Main function to demonstrate video chapter generation."""
    video_id = "P127jhj-8-Y"  # Replace with your video ID
    
    try:
        # Initialize generator
        generator = VideoChapterGenerator()
        
        # Extract transcript
        print("Extracting transcript...")
        formatted_text, transcript_data = generator.extract_transcript(video_id)
        
        # Segment transcript
        print("Segmenting transcript...")
        segments = generator.segment_transcript(transcript_data)
        
        # Generate chapters
        print("Generating chapters...")
        chapters = generator.generate_chapters(segments, video_id=video_id)
        
        # Display results
        print("\n" + generator.format_chapters_output(chapters))
        
        # Save to file
        with open(f"chapters_{video_id}.json", "w") as f:
            chapters_dict = [
                {
                    "title": ch.title,
                    "summary": ch.summary,
                    "start_time": ch.start_time,
                    "end_time": ch.end_time,
                    "duration": ch.duration,
                    "start_timestamp": ch.start_timestamp,
                    "end_timestamp": ch.end_timestamp,
                    "youtube_timestamp": ch.youtube_timestamp
                }
                for ch in chapters
            ]
            json.dump(chapters_dict, f, indent=2)
        
        print(f"\nChapters saved to: chapters_{video_id}.json")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()