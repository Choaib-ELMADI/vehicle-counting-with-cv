from moviepy import VideoFileClip, concatenate_videoclips  # type: ignore


def duplicate_video(input_path, output_path, n):
    clip = VideoFileClip(input_path).without_audio()
    clips = [clip] * n
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(output_path, codec="libx264", audio=False)


if __name__ == "__main__":
    input_path = "Videos/kech2.mp4"
    output_path = "Videos/kech2_d.mp4"
    n = 10

    duplicate_video(input_path, output_path, n)
