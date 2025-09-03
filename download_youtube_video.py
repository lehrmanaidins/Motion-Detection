
import yt_dlp

def download_youtube_video_highest_quality_no_audio(youtube_url: str, output_path: str) -> str:
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/bestvideo',
        'outtmpl': output_path,
        'quiet': True,
        'merge_output_format': 'mp4',  # Ensures output is mp4 if needed
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([youtube_url])
    return output_path

def main() -> None:
    youtube_url = 'https://www.youtube.com/watch?v=oeXAEWJ5C4o'  # Peaceful Unedited 140 minute video of the Night Sky with Orbiting Satellites, Meteorites, Objects
    output_path = 'night_sky_140min.mp4'  # Desired output path for the video
    download_youtube_video_highest_quality_no_audio(youtube_url, output_path)

if __name__ == "__main__":
    main()
