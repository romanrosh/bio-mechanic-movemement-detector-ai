from tkinter import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import moviepy.editor as mp
import cv2


def video_cut(source):
    """
    i is a starting point
    duration is ending point
    :param source:
    :return:
    """
    duration = round(mp.VideoFileClip(source).duration)
    i = 0
    while i < duration:
        darray = source.split(".")
        destin = darray[0] + str(i) + "." + darray[1]
        ffmpeg_extract_subclip(source, i, i+14, targetname=destin)
        i += 14

    cap = cv2.VideoCapture('videoplayback.mp4')

    while (cap.isOpened()):
        ret, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        print(cap.read())

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def button2(param):
    print('inside button2', param)


def button3(param):
    print('inside button3', param)


def main():
    input_width = 150
    root = Tk()
    root.resizable(width=FALSE, height=FALSE)
    root.geometry("630x700")

    headerinputtext = Label(root, text='Template for data preparation resize, flip, rename pics', anchor=W)
    headerinputtext.place(x=10, y=10, height=25)

    video_cutinputtext = Label(root, text='video_cut', anchor=W)
    video_cutinputtext.place(x=10, y=70, width=250, height=20)
    video_cutinput = Entry(root)
    video_cutinput.place(x=10, y=100, width=70 + (input_width + 20) * 2, height=25)
    video_cut_button = Button(root, text='video_cut', command=lambda: video_cut(video_cutinput.get()))
    video_cut_button.config(relief=RAISED)
    video_cut_button.place(x=450, y=100, width=100, height=25)

    button2inputtext = Label(root, text='Button 2', anchor=W)
    button2inputtext.place(x=10, y=130, width=250, height=25)
    button2input = Entry(root)
    button2input.place(x=10, y=160, width=70 + (input_width + 20) * 2, height=25)
    goButton2 = Button(root, text='button2', command=lambda: button2(button2input.get()))
    goButton2.config(relief=RAISED)
    goButton2.place(x=450, y=160, width=100, height=25)

    button3inputtext = Label(root, text='Button 3', anchor=W)
    button3inputtext.place(x=10, y=190, width=250, height=25)
    button3input = Entry(root)
    button3input.place(x=10, y=220, width=70 + (input_width + 20) * 2, height=25)
    goButton3 = Button(root, text='button3', command=lambda: button3(button3input.get()))
    goButton3.config(relief=RAISED)
    goButton3.place(x=450, y=220, width=100, height=25)

    ExitButton = Button(root, text='Exit', command=root.destroy)
    ExitButton.config(relief=RAISED)
    ExitButton.place(x=500, y=400, width=100, height=25)

    root.mainloop()git 


if __name__ == '__main__':
    main()
