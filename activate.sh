source .venv/bin/activate

# 1. Create the missing directory inside your project's .venv
mkdir -p $PWD/.venv/lib/python3.12/site-packages/cv2/qt/fonts

# 2. Copy WSL's TrueType fonts directly into it
cp /usr/share/fonts/truetype/dejavu/*.ttf $PWD/.venv/lib/python3.12/site-packages/cv2/qt/fonts/


echo successful
