⁸# :1234: MNIST Streamlit Project :slot_machine: 

### [:pencil2: 프로젝트 결과 직접 체험하기]()

![image](https://user-images.githubusercontent.com/84713532/215334194-70184c50-0a5a-436d-ad15-7010453b9ef3.png)

## :floppy_disk: Mini Project (2023/01/27 ~ 2023/01/30) :date:

> :family: 팀명: ???
- [강동엽](https://github.com/kdy1493)
- [이상훈](https://github.com/Dawnnote)
- [지우근](https://github.com/UGeunJi)
- [최세현](https://github.com/kdy1493)

---

## :bangbang: Terminal Settings

```
    # 아래 단계를 차례로 실행해 주세요
    # 1. python -m pip install --upgrade pip
    # 2. conda create -n tensorflow python=3.7
    # 3. activate tensorflow
    # 4. pip install tensorflow
    # 5. pip install keras
    # 6. pip install opencv-python
    # 7. pip install streamlit_drawable_canvas
    # 8. 모듈이 없다고 오류뜨면 pip install <module name>
    # 9. pip install -r requirements.txt
```

---

## Code

[Main Page](#main-page)

[Calculator](#calculator)

[Lottery](#lottery)

[Phone Number](#phone-number)

---

## Main Page 

```py
import cv2
from keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from streamlit_vertical_slider import vertical_slider
from utils import set_bg

set_bg('images/mnist2.png')

st.sidebar.image('./images/sidebar_main.jpg')

st.image('./images/title.png')

st.write('### :star: :red[특별 체험]')


lotto = st.checkbox('Lotto')
if lotto:
    st.write('좋아요!')
phone_number = st.checkbox('Phone Number')
if phone_number:
    st.image('./images/따봉1.png')
calculator = st.checkbox('Calculator')
if calculator:
    st.image('./images/따봉2.png')


st.write('### :mag: :blue[기본 체험]')

@st.cache(allow_output_mutation=True)
def load():
    return load_model('./model.h5')
model = load()

CANVAS_SIZE = 340

col1, col2 = st.columns(2)

with col1:
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

if canvas.image_data is not None:
    img = canvas.image_data.astype(np.uint8)
    img = cv2.resize(img, dsize=(28, 28))
    preview_img = cv2.resize(img, dsize=(CANVAS_SIZE, CANVAS_SIZE), interpolation=cv2.INTER_NEAREST)

    col2.image(preview_img)

    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x = x.reshape((-1, 28, 28, 1))
    y = model.predict(x).squeeze()

    st.write('## :dart: 예측 결과: %d' % np.argmax(y))

tab1, tab2 = st.tabs(["예쁜 사진으로 보기", "그래프로 보기"])

with tab1:
    [col3, col4, col5] = st.columns([2, 2, 1]) 
    with col3:
        if st.button('일치'):
            if np.argmax(y) == 0:
                st.image('./images/zero.jpg')
                with open("./images/zero.jpg", "rb") as file:
                    btn = st.download_button(
                            label="이미지 소장하기",
                            data=file,
                            file_name="zero.jpg",
                            mime="image/jpg"
                        )
            elif np.argmax(y) == 1:
                st.image('./images/one.jpg')
                with open("./images/one.jpg", "rb") as file:
                    btn = st.download_button(
                            label="이미지 소장하기",
                            data=file,
                            file_name="one.jpg",
                            mime="image/jpg"
                        )
            elif np.argmax(y) == 2:
                st.image('./images/two.jpg')
                with open("./images/two.jpg", "rb") as file:
                    btn = st.download_button(
                            label="이미지 소장하기",
                            data=file,
                            file_name="two.jpg",
                            mime="image/jpg"
                        )
            elif np.argmax(y) == 3:
                st.image('./images/three.jpg')
                with open("./images/three.jpg", "rb") as file:
                    btn = st.download_button(
                            label="이미지 소장하기",
                            data=file,
                            file_name="three.jpg",
                            mime="image/jpg"
                        )
            elif np.argmax(y) == 4:
                st.image('./images/four.jpg')
                with open("./images/four.jpg", "rb") as file:
                    btn = st.download_button(
                            label="이미지 소장하기",
                            data=file,
                            file_name="four.jpg",
                            mime="image/jpg"
                        )
            elif np.argmax(y) == 5:
                st.image('./images/five.jpg')
                with open("./images/five.jpg", "rb") as file:
                    btn = st.download_button(
                            label="이미지 소장하기",
                            data=file,
                            file_name="five.jpg",
                            mime="image/jpg"
                        )
            elif np.argmax(y) == 6:
                st.image('./images/six.jpg')
                with open("./images/six.jpg", "rb") as file:
                    btn = st.download_button(
                            label="이미지 소장하기",
                            data=file,
                            file_name="six.jpg",
                            mime="image/jpg"
                        )
            elif np.argmax(y) == 7:
                st.image('./images/seven.jpg')
                with open("./images/seven.jpg", "rb") as file:
                    btn = st.download_button(
                            label="이미지 소장하기",
                            data=file,
                            file_name="seven.jpg",
                            mime="image/jpg"
                        )
            elif np.argmax(y) == 8:
                st.image('./images/eight.jpg')
                with open("./images/eight.jpg", "rb") as file:
                    btn = st.download_button(
                            label="이미지 소장하기",
                            data=file,
                            file_name="eight.jpg",
                            mime="image/jpg"
                        )
            elif np.argmax(y) == 9:
                st.image('./images/nine.jpg')
                with open("./images/nine.jpg", "rb") as file:
                    btn = st.download_button(
                            label="이미지 소장하기",
                            data=file,
                            file_name="nine.jpg",
                            mime="image/jpg"
                        )

    with col4:
        if st.button('불일치'):
            st.image('./images/불일치.png')
            st.write('#### 다시 그려주세요')
        
    with col5:
        st.write("##### 내 만족도")
        vertical_slider(
            key="slider",
            default_value=10,
            step=1,
            min_value=0,
            max_value=100,
            track_color="gray",
            thumb_color="blue", 
            slider_color="red",
        )
    
with tab2:
    st.write("해당 숫자일 확률을 나타냅니다.")
    st.bar_chart(y)

st.write('### BGM :musical_score:')
st.write('Fantasie Impromptu')
st.audio('./audios/Fantasie Impromptu.mp3')
st.write('He is a Perate')
st.audio('./audios/He is a Perate.mp3')
st.write('Merry Christmas Mr-Laurence')
st.audio('./audios/Merry Christmas Mr-Laurence.mp3')
st.write('River flows in your')
st.audio('./audios/River flows in your.mp3')
st.write('Summer')
st.audio('./audios/Summer.mp3')
```

## Calculator

```py
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
```

## Lottery

```py
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
import tensorflow as ts
import random  
import pyautogui

disable_btn=False

@st.cache(allow_output_mutation=True)
def load():
    return load_model('./model.h5')
model = load()

st.title('행운의 숫자 뽑기')

col1, col2,col3,col4 = st.columns(4)
CANVAS_SIZE = 192
yes_btn=False
no_btn=False




with col1:
    st.write("숫자를 그려주세요")
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
    st.write("숫자를 그려주세요")
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
    


# 브라우저가 가진 session을 이용하여 화면이 새로 렌더링 되더라도 계속 내 데이터를 가지고 갑니다.
if 'key' not in st.session_state:
    st.session_state['key'] = set()
if 'sort' not in st.session_state:
    st.session_state['sort'] = []
if 'rn' not in st.session_state:
    st.session_state['rn'] = random.sample(range(1,100),4)   

def lottery_check (check,win):
    same=0
    for i in check :
        if i in win:
             same+=1
    return same
    
    
if len(st.session_state.key)>3 :
    disable_btn=True
    with col1:
        st.write('뽑은 숫자들')
        st.session_state.sort=list(st.session_state.key)
        sorted_number=sorted(st.session_state.sort)
        st.write(sorted_number)
    with col2:    
        st.write('당첨번호 확인하기')
        check_number=st.button('과연 몇개?')
        st.write('나온 숫자입니다')
        lottery_num=sorted(st.session_state.rn)
        st.write(lottery_num)
        txt=lottery_check(sorted_number,lottery_num)
        st.write('%d 개 맞았습니다 ' %(txt))

    

def get_2number (c1,c2):
    img = canvas1.image_data.astype(np.uint8)
    img = cv2.resize(img, dsize=(28, 28))
    x1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x1 = x1.reshape((-1, 28, 28, 1))
    y1 = model.predict(x1).squeeze()
    get_number1=np.argmax(y1)
    
    img = canvas2.image_data.astype(np.uint8)
    img = cv2.resize(img, dsize=(28, 28))
    x2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x2 = x2.reshape((-1, 28, 28, 1))
    y2 = model.predict(x2).squeeze()
    get_number2=np.argmax(y2)
    return int(get_number1)*10+int(get_number2)

if (canvas1.image_data is not None) & (canvas2.image_data is not None):
    with col3:

        get_number=get_2number(canvas1,canvas2)
        st.subheader('쓰신 숫자가: %d 맞나요?' % get_number)

        with st.container() :
            yes_btn= st.button('yes',disabled=(disable_btn))
            no_btn= st.button('no')

        # 바뀐 부분
        if yes_btn:
            st.session_state.key.add(int(get_number))
            

        if no_btn:
            with col2:
                st.write("숫자를 다시 입력해주세요")
        st.text('중복숫자는 불가능')
# 바뀐부분
with col4:
    st.text(" 총 4개의\n 숫자를 입력\n 하세요")
    if st.session_state:
        str_set=''
        for i in st.session_state.key :
           str_set+=str(i)+' '
        st.subheader(str_set)
    with col4:
        refresh_btn=st.button("regame?")
        if refresh_btn==True:
            pyautogui.hotkey('f5')
```

## Phone Number

```py
import cv2
from keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd
from io import BytesIO


@st.cache(allow_output_mutation=True)
def load():
    return load_model('./model.h5')
model = load()

name_input = st.text_input('이름을 입력하세요')

st.subheader(" '010'을 제외한 휴대폰 번호 8자리를 입력해주세요.")

CANVAS_SIZE = 160
STROKE_SIZE = 12
st.text('중간 번호 4자리')

col1, col2, col3, col4 = st.columns(4)

with col1:
    canvas1 = st_canvas(
        fill_color='#000000',
        stroke_width=STROKE_SIZE,
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
        stroke_width=STROKE_SIZE,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas2'
    )

with col3:
    canvas3 = st_canvas(
        fill_color='#000000',
        stroke_width=STROKE_SIZE,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas3'
    )

with col4:
    canvas4 = st_canvas(
        fill_color='#000000',
        stroke_width=STROKE_SIZE,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas4'
    )
st.text('마지막 번호 4자리')
col5, col6, col7, col8 = st.columns(4)
with col5:
    canvas5 = st_canvas(
        fill_color='#000000',
        stroke_width=STROKE_SIZE,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas5'
    )

with col6:
    canvas6 = st_canvas(
        fill_color='#000000',
        stroke_width=STROKE_SIZE,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas6'
    )

with col7:
    canvas7 = st_canvas(
        fill_color='#000000',
        stroke_width=STROKE_SIZE,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas7'
    )

with col8:
    canvas8 = st_canvas(
        fill_color='#000000',
        stroke_width=STROKE_SIZE,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=CANVAS_SIZE,
        height=CANVAS_SIZE,
        drawing_mode='freedraw',
        key='canvas8'
    )


if canvas1.image_data is not None:
    img1 = canvas1.image_data.astype(np.uint8)
    img1 = cv2.resize(img1, dsize=(28, 28))

    img2 = canvas2.image_data.astype(np.uint8)
    img2 = cv2.resize(img2, dsize=(28, 28))

    img3 = canvas3.image_data.astype(np.uint8)
    img3 = cv2.resize(img3, dsize=(28, 28))

    img4 = canvas4.image_data.astype(np.uint8)
    img4 = cv2.resize(img4, dsize=(28, 28))

    img5 = canvas5.image_data.astype(np.uint8)
    img5 = cv2.resize(img5, dsize=(28, 28))

    img6 = canvas6.image_data.astype(np.uint8)
    img6 = cv2.resize(img6, dsize=(28, 28))

    img7 = canvas7.image_data.astype(np.uint8)
    img7 = cv2.resize(img7, dsize=(28, 28))

    img8 = canvas8.image_data.astype(np.uint8)
    img8 = cv2.resize(img8, dsize=(28, 28))

    x1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    x1 = x1.reshape((-1, 28, 28, 1))
    y1 = model.predict(x1).squeeze()

    x2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    x2 = x2.reshape((-1, 28, 28, 1))
    y2 = model.predict(x2).squeeze()

    x3 = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    x3 = x3.reshape((-1, 28, 28, 1))
    y3 = model.predict(x3).squeeze()

    x4 = cv2.cvtColor(img4, cv2.COLOR_BGR2GRAY)
    x4 = x4.reshape((-1, 28, 28, 1))
    y4 = model.predict(x4).squeeze()

    x5 = cv2.cvtColor(img5, cv2.COLOR_BGR2GRAY)
    x5 = x5.reshape((-1, 28, 28, 1))
    y5 = model.predict(x5).squeeze()

    x6 = cv2.cvtColor(img6, cv2.COLOR_BGR2GRAY)
    x6 = x6.reshape((-1, 28, 28, 1))
    y6 = model.predict(x6).squeeze()

    x7 = cv2.cvtColor(img7, cv2.COLOR_BGR2GRAY)
    x7 = x7.reshape((-1, 28, 28, 1))
    y7 = model.predict(x7).squeeze()

    x8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)
    x8 = x8.reshape((-1, 28, 28, 1))
    y8 = model.predict(x8).squeeze()

    st.write('#### Number: 010-%d%d%d%d-%d%d%d%d' %(np.argmax(y1),np.argmax(y2),np.argmax(y3),np.argmax(y4),
                            np.argmax(y5),np.argmax(y6),np.argmax(y7),np.argmax(y8)),' 이 번호가 맞습니까?')
    
if 'number' not in st.session_state:
    st.session_state.number = '010-'

if 'Phone_Number' not in st.session_state:
    st.session_state.Phone_Number = []

if 'Name' not in st.session_state:
    st.session_state.Name = []

def phone_number_covert():
    num = str(np.argmax(y1))+str(np.argmax(y2))+str(np.argmax(y3))+str(np.argmax(y4))+'-'+str(np.argmax(y5))+str(np.argmax(y6))+str(np.argmax(y7))+str(np.argmax(y8))
    if len(st.session_state.number) == 13:
        pass
    else:
        st.session_state.number += num
    return st.session_state.number
    
col_1, col_2 = st.columns(2)
with col_1:
    if st.button('Yes'):
        st.session_state.number = phone_number_covert()

        st.session_state.Phone_Number += [st.session_state.number]
        st.session_state.Name += [name_input]

        df = pd.DataFrame(
            {
                "Name" : st.session_state.Name,
                "Phone_Number": st.session_state.Phone_Number
            }
        )
        st.session_state.number = '010-'

        st.dataframe(df)

        st.button('추가 입력')

        csv = df.to_csv().encode('ANSI')

        st.download_button(
            label="CSV 파일 다운로드",
            data=csv,
            file_name='휴대폰 번호 정보.csv'
        )

        excel_data = BytesIO()  
        # 하나는 df2를 xlsx
        df.to_excel(excel_data)

        st.download_button(
            label="엑셀 파일 다운로드",
            data=excel_data,
            file_name='휴대폰 번호 정보.xlsx'
        )

with col_2:
    if st.button('No'):
        st.button('Reset')
````
