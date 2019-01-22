from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mp

def every_seven_seconds(source, dest):
    duration = round(mp.VideoFileClip(source).duration)
    i = 0
    while i < duration:
        darray = dest.split(".")
        destin = darray[0] + str(i) + "." + darray[1]
        ffmpeg_extract_subclip(source, i, i+18, targetname=destin)
        i += 14

every_seven_seconds("practice.mp4", "testietest.mp4")