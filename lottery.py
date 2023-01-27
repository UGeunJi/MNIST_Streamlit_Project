# import cv2
# from keras.models import load_model
# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# import numpy as np
#
#
# @st.cache(allow_output_mutation=True)
# def load():
#     return load_model('./model.h5')
# model = load()
#
#
#
# st.header('행운의 숫자 뽑기')
#
# col1, col2,col3 = st.columns(3)
# CANVAS_SIZE = 192
# number_list=[]
# yes_btn=False
# no_btn=False
#
# class save_num:
#     def __init__(self):
#         self.numbers=[]
#     def append_number(self,number):
#         self.numbers.append(number)
#
# a1 = save_num()
#
#
# with col1:
#     st.write("숫자를 그려주세요")
#     canvas = st_canvas(
#         fill_color='#000000',
#         stroke_width=20,
#         stroke_color='#FFFFFF',
#         background_color='#000000',
#         width=CANVAS_SIZE,
#         height=CANVAS_SIZE,
#         drawing_mode='freedraw',
#         key='canvas'
#     )
#
# if canvas.image_data is not None:
#     with col2:
#         img = canvas.image_data.astype(np.uint8)
#         img = cv2.resize(img, dsize=(28, 28))
#         x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         x = x.reshape((-1, 28, 28, 1))
#         y = model.predict(x).squeeze()
#         get_number=np.argmax(y).copy()
#         st.write('## 쓰신 숫자가: %d 맞나요?' % np.argmax(y))
#         with st.container() :
#             yes_btn= st.button('yes')
#             no_btn= st.button('no')
#
# def add_num(num):
#     number_list.append(num)
#     return number_list
# if yes_btn:
#     f=open('lottery.txt','a+')
#     f.write(str(get_number.item()))
#     f.close()
#
#
# if no_btn:
#     with col2:
#         st.write("숫자를 다시 입력해주세요")
# with col3:
#     f=open('lottery.txt','r')
#     st.write(f.readline())
#     f.close()
#



import cv2
from keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np

@st.cache(allow_output_mutation=True)
def load():
    return load_model('./model.h5')
model = load()

st.header('행운의 숫자 뽑기')

col1, col2,col3 = st.columns(3)
CANVAS_SIZE = 192
yes_btn=False
no_btn=False

with col1:
    st.write("숫자를 그려주세요")
    canvas = st_canvas(
        fill_color='#000000',
        stroke_width=20,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas'
    )

# 브라우저가 가진 session을 이용하여 화면이 새로 렌더링 되더라도 계속 내 데이터를 가지고 갑니다.
if 'key' not in st.session_state:
    st.session_state['key'] = []

if canvas.image_data is not None:
    with col2:
        img = canvas.image_data.astype(np.uint8)
        img = cv2.resize(img, dsize=(28, 28))
        x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = x.reshape((-1, 28, 28, 1))
        y = model.predict(x).squeeze()
        get_number=np.argmax(y)
        st.write('## 쓰신 숫자가: %d 맞나요?' % np.argmax(y))

        with st.container() :
            yes_btn= st.button('yes')
            no_btn= st.button('no')

        # 바뀐 부분
        if yes_btn:
            st.session_state.key.append(int(get_number))

        if no_btn:
            with col2:
                st.write("숫자를 다시 입력해주세요")

# 바뀐부분
with col3:
    if st.session_state:
        st.write(st.session_state.key)