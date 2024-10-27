import gdown
import os

def main():
    gdown.download('https://drive.google.com/file/d/1TJtrry4CNEFmLLOlMAIm6WuNvSG07NuV')
    os.system('unzip -j pre_trained.zip')

    
if __name__ == '__main__':
    main()
