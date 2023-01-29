import cv2
from keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import random

@st.cache(allow_output_mutation=True)
def load():
    return load_model('./model.h5')
model = load()

st.write('# Calculator :computer:')
st.write('Draw Any Two Number')
CANVAS_SIZE = 192

# 숫자 확인
def model_predict(canvas1, canvas2) :
        img1 = canvas1.image_data.astype(np.uint8)
        img1 = cv2.resize(img1, dsize=(28, 28))
        preview_img = cv2.resize(img1, dsize=(CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)
        x1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        x1 = x1.reshape((-1, 28, 28, 1))
        y1 = model.predict(x1).squeeze()

        
        img2 = canvas2.image_data.astype(np.uint8)
        img2 = cv2.resize(img2, dsize=(28, 28))
        x2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        x2 = x2.reshape((-1, 28, 28, 1))
        y2 = model.predict(x2).squeeze()

        return np.argmax(y1), np.argmax(y2) 

# 계산
def calculation() :
    operator = st.radio('operator', ["+", "-", "%", "*"]) 

    if operator == "+" : 
        st.write('## %d + %d = %d' % (z1,z2,z1+z2))

    elif operator == "-" : 
        st.write('## %d - %d = %d' % (z1,z2,z1-z2))

    elif operator == "%" : 
        st.write('## %d / %d = %d' % (z1,z2,z1/ z2))

    else : 
        st.write('## %d * %d = %d' % (z1,z2,z1*z2))

# canvas 설정
col1, col2 = st.beta_columns(2)

with col1:
    canvas1 = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas1'
    )

with col2:
    canvas2 = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas2'
    )
   

# 숫자 에측 
if canvas1.image_data is not None and canvas2.image_data is not None:
    z1, z2 = model_predict(canvas1, canvas2)


# 그린 숫자가 맞는지 확인
st.write("## Is the Number correct? %d, %d" % (z1 ,z2) )
[col1, col2] =st.columns(2) 

# 맞을 경우  
with col1 :
    if st.button("Yes") : 
        st.write("## Choose an operator")
calculation()


with col2 : 
    if st.button("NO")  : 
        st.write("Please try again")


    





    # 아래 단계를 차례로 실행해 주세요
    # 1. python -m pip install --upgrade pip
    # 2. conda create -n tensorflow python=3.7
    # 3. activate tensorflow
    # 4. pip install tensorflow
    # 5. pip install keras
    # 6. pip install opencv-python
    # 7. pip install streamlit_drawable_canvas
    # 8. 모듈이 없다고 오류뜨면 그것도 pip install 하기
