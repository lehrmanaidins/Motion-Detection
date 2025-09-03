
from typing import Final
import yt_dlp

def get_youtube_stream_url(youtube_url: Final[str]) -> Final[str]:
    ydl_opts = {
        'format': 'best[height<=480]',
        'quiet': True,
        'noplaylist': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']
    
def main() -> None:
    get_youtube_stream_url('https://www.youtube.com/watch?v=H999s0P1Er0')
    
if __name__ == "__main__":
    main()