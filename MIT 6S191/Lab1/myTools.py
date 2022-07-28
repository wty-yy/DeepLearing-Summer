import os  # 执行系统命令
import pathlib  # 文件目录操作
import IPython.display as ipythondisplay  # 将wav音频文件在浏览器中直接打开

abc2midi_path = pathlib.Path(r'.\tools\abc2midi.exe')
midi2wav_path = pathlib.Path(r'.\tools\timidity.exe')


def play_song(song, fname='tmp'):  # 可以将字符串形式的abc记谱法，转化为wav文件，并播放出来
    for song_path in pathlib.Path.cwd().glob('*.wav'):  # 将过去生成的音乐清空
        os.remove(song_path)
    song_path = pathlib.Path(f"{fname}.abc")
    with open(song_path, 'w') as file:
        file.write(song)  # 写入到./tmp.abc文件中
    cmd = f"{abc2midi_path} {song_path}"
    ret = os.system(cmd)  # 转换 abc -> midi
    print(f"{song_path} -> midi {'Failed' if ret else 'OK'}!")
    os.remove(song_path)  # 删除./tmp.abc文件
    for i, song_path in enumerate(pathlib.Path.cwd().glob('*.mid')):  # 查找所有生成的.mid音乐
        song_name = song_path.name[:-4]  # 获取文件名前缀
        print(f"{i + 1}. ", end='')
        cmd = f"{midi2wav_path} {song_name + '.mid'} -Ow {song_name + '.wav'}"
        ret = os.system(cmd)  # 转换 midi -> wav
        print(f"{song_path} -> wav {'Failed' if ret else 'OK'}!")
        ipythondisplay.display(ipythondisplay.Audio(song_name + '.wav'))  # 显示到Jupyter中
        os.remove(song_name + '.mid')  # 删除.mid文件


if __name__ == '__main__':
    song = r"X:1\nT:Alexander's\nZ: id:dc-hornpipe-1\nM:C|\nL:1/8\nK:D Major\n(3ABc|dAFA DFAd|fdcd FAdf|gfge fefd|(" \
           r"3efe (3dcB A2 (3ABc|!\ndAFA DFAd|fdcd FAdf|gfge fefd|(3efe dc d2:|!\nAG|FAdA FAdA|GBdB GBdB|Acec " \
           r"Acec|dfaf gecA|!\nFAdA FAdA|GBdB GBdB|Aceg fefd|(3efe dc d2:|! "
    play_song(song)
