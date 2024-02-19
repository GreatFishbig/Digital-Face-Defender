# coding=UTF-8

import os
import uuid

import PIL.ImageQt
import streamlit as st
from PIL import Image
from io import BytesIO
import  tools
import infer
import cv2
import copy


patients_root= "./patients"
models_root="./models"
tmp_path= "temp"

st.set_page_config(layout="wide", page_title="Digital Face Defender ")
    #":dog: Try uploading an image to watch the background magically removed. "
    #"Full quality images can be downloaded from the sidebar. "
    #"This code is open source and available [here](https://github.com/tyler-simons/BackgroundRemoval) on GitHub."
    #" Special thanks to the [rembg library](https://github.com/danielgatis/rembg) :grin:"
# )
st.sidebar.write("## Select sample images ")

# Download the fixed image
# def convert_image(img):
#     buf = BytesIO()
#     img.save(buf, format="PNG")
#     byte_im = buf.getvalue()     # 将磁盘文件读入内存块。
#     return byte_im



def get_image_name(age):
    temp_name = "20"
    if  0 < age <=5:  temp_name = "05"
    if  5 < age <=10:  temp_name = "10"
    if  10< age <=15:  temp_name = "15"
    if  15< age <=20:  temp_name = "20"
    # if  20< age <=25:  temp_name = "25"
    if  20< age <=30:  temp_name = "30"
    # if  30< age <=35:  temp_name = "35"
    if  30< age <=40:  temp_name = "40"
    # if  40< age <=45:  temp_name = "45"
    if  40< age <=50:  temp_name = "50"
    # if  50< age <= 55: temp_name = "55"
    if  50< age <= 60: temp_name = "60"
    # if  60< age <= 65: temp_name = "65"
    if  60< age <= 70: temp_name = "70"
    # if  70< age <= 75: temp_name = "75"
    if  70< age      : temp_name = "80"
    return    temp_name


def auto_select_model(sex, age):
    if  sex == 1 :
        st.session_state.sb_type_m="#1 Men Model"  #Man.5-80"
        temp_name='man-'
    else :
        st.session_state.sb_type_m ="#1 Women Model"  #women.5-80"
        temp_name = 'women-'

    str_age= get_image_name(age)
    temp_name=temp_name+str_age
    # st.session_state["sb_image_m"]= temp_name +"Y.png"
    st.session_state["sl_age"]=int(str_age)

# def select_model(sex, age):
#     if  sex == 1 :
#         st.session_state.sb_type_m="#1 Men Model"  #Man.5-80"
#         temp_name='man-'
#     else :
#         st.session_state.sb_type_m ="#1 Women Model"  #women.5-80"
#         temp_name = 'women-'
#     st.session_state["sl_age"] = age
#     return temp_name+get_image_name(age)+"Y.png"



# def fix_image(upload):
#     image = Image.open(upload)
#     col1.write("Original Image :camera:")
#     col1.image(image)
#
#     fixed = image # remove(image)
#     col2.write("Virtual Human Image :wrench:")
#     col2.image(fixed)
#     st.sidebar.markdown("\n")
#     st.sidebar.download_button("Download fixed image", convert_image(fixed), "fixed.png", "image/png")

@st.cache_data
def get_photo_types(path:str=patients_root) -> list :
    # "Get a list of immediate subdirectories"
    r = next(os.walk(path))[1]
    r.sort()
    return r

@st.cache_data
def get_photo_name(dir:str=patients_root,subdir:str='') -> list :
    # "Get a list of immediate subfiles"
    if subdir is not None :
       r =  next(os.walk(os.path.join(dir,subdir)))[2]
    else :
        r = next(os.walk(dir))[2]
    r.sort()
    return r

# @st.cache_data
def get_photo_file(dir:str,name:str,typedir:str) -> Image :
    return Image.open(os.path.join(typedir,dir,name))


def draw_detected_img(faceimgae,landmarks_img,detect_result='0000',to_web=False):
    # faceimage 融合后的图片， landmarks_img 融合后的坐标集合， detect_result 判断结果， 是否在web上显示
    if isinstance(faceimgae,str):
       img_1=copy.deepcopy(cv2.imread(faceimgae))
       img_2=copy.deepcopy(img_1)
    else :
       img_1 = copy.deepcopy(faceimgae)
       img_2 = copy.deepcopy(img_1)

    RT_point = (landmarks_img[46][0, 0], landmarks_img[46][0, 1])
    RB_point = (landmarks_img[188][0, 0], landmarks_img[188][0, 1])
    LT_point = (landmarks_img[285][0, 0], landmarks_img[276][0, 1])
    LB_point = (landmarks_img[276][0, 0], landmarks_img[188][0, 1])

    L_Color = (0,0,0)
    R_Color = (0,0,0)
    if detect_result[0] == '1': L_Color_1=[255,0,0]
    elif  detect_result[0]=='2' : L_Color_1=[255,255,0]
    else :L_Color_1 = [0,255,0]

    if detect_result[1] == '1' : R_Color_1=[255,0,0]
    elif   detect_result[1] =='2' : R_Color_1=[255,255,0]
    else : R_Color_1 =[0, 255, 0]

    cv2.rectangle(img_1, RT_point, RB_point, R_Color_1, 3)
    cv2.rectangle(img_1, LT_point, LB_point, L_Color_1, 3)
    cv2.rectangle(img_1, (10,20), (250,130), (255,255,255), 2)
    cv2.putText(img_1, '--- ', (15, 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 0), 2)
    cv2.putText(img_1, 'Esotropia ', (85, 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(img_1, '--- ', (15, 85), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(img_1, 'Exotropia', (85, 85), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(img_1, '--- ', (15, 115), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img_1, 'No E*tropia', (85, 115), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    image_1 = Image.fromarray(img_1)

    if detect_result[2] == '1'  : L_Color_2=[128,0,128]
    elif detect_result[2] == '2' : L_Color_2=[255,0,255]
    else : L_Color_2 = [0,255,0]
    if detect_result[3] == '1'   : R_Color_2=[128,0,128]
    elif   detect_result[2] == '2' : R_Color_2 = [255, 0, 255]
    else : R_Color_2 = [0,255,0]
    cv2.rectangle(img_2, RT_point, RB_point, R_Color_2, 3)
    cv2.rectangle(img_2, LT_point, LB_point, L_Color_2, 3)
    cv2.rectangle(img_2, (10, 20), (750, 130), (255,255,255), 2)
    cv2.putText(img_2, '--- ', (15, 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (128,0,128), 2)
    cv2.putText(img_2, 'Narrow palpebral Fissure (or Blepharoptosisin)', (85, 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_2, '--- ', (15, 85), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 255), 2)
    cv2.putText(img_2, 'Excessive palpebral Fissure (or Thyroid Eye)', (85, 85), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img_2, '--- ', (15, 115), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(img_2, 'NO Narrow/Excessive palpebral Fissure', (85, 115), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    # thyroid eye
    # cv2.putText(img_2, 'Blepharoptosisin', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 0, 0),5)
    # cv2.putText(img_2, 'NO Blepharoptosisin', (50, 100), cv2.FONT_HERSHEY_COMPLEX, 1.5, (0, 255, 0), 5)



    image_2 = Image.fromarray(img_2)
    return image_1,image_2

def get_temp_filename():
    uuid_str = uuid.uuid4().hex
    return  os.path.join(tmp_path,uuid_str+'.png')

def sw_facephoto(img_p,img_m):
    filename_p=get_temp_filename()
    filename_m= get_temp_filename()
    img_p.save(filename_p)
    img_m.save(filename_m)

    # fusion_file,fusion_landmarks,Not_seamless_photo=tools.fs(filename_p,filename_m)    #  test orig is : filename_p,filename_m
    fusion_file, fusion_landmarks = tools.fs(filename_p, filename_m)
    # img=PIL.Image.open(fusion_file,mode='r')
    os.remove(filename_p)
    os.remove(filename_m)
    # os.remove(fusion_file)
    # 返回的是内存图片
    #return img,Not_seamless_photo,fusion_landmarks
    # return fusion_file,Not_seamless_photo,fusion_landmarks
    return fusion_file,  fusion_landmarks





def process_imgaes_folder() :
    # RAW_IMAGES_DIR = filedialog.askdirectory(
    #                  initialdir='./patients/sample',
    #                  title='choose patient path' )
    RAW_IMAGES_DIR= 'temp/sample'
    path2_ = './models/Man.5-80/006.png'
    path2_img=Image.open(path2_)
    for img_name in [f for f in os.listdir(RAW_IMAGES_DIR) if f.endswith(('png', 'jpg'))]:
        path1_ = os.path.join(RAW_IMAGES_DIR, img_name)
        path1_img=Image.open(path1_)
        # mor_img_path ,landmarks_img= fs(path2_, path1_)
        try:
            # landmarks_img = find_img_data(path1_)
            fusion_file,landmarks_img=sw_facephoto(path2_img, path1_img)
            # cv2.imread(fusion_file, cv2.IMREAD_COLOR)
            # cv2.cvtColor(np.asarray(image_name), cv2.COLOR_RGB2BGR)
            # cv2.imwrite(path1_+'_fw.png',fusion_file)
            Image.fromarray(fusion_file).save(path1_+'_fw.png')
        except Exception as r:
            print(path1_, r)
            landmarks_img = None

        if landmarks_img is None:
            print(path1_ + ' NOT find data !')
            continue


if __name__ == '__main__':

    st.title(' Welcome To Digital Face Defender web application !')
    instructions = """
          
        """
    st.write(instructions)

    file = st.file_uploader('Upload An Image')

    Patient_Types = st.sidebar.selectbox(
        " **:red[Patient photos Types]**", get_photo_types(patients_root))

    # st.write(Patient_photos)
    #
    selected_Patient_photo = st.sidebar.selectbox("**:red[Patient photos]**",
                                                  get_photo_name(patients_root, Patient_Types))
    img_p = get_photo_file(Patient_Types, selected_Patient_photo, patients_root)
    if file:  # if user uploaded file
        img_p = Image.open(file)

    # 根据患者性别， 年龄 自动选择 虚拟人照片
    sex, age = infer.get_sex_age(img_p)
    # if sex == 0:
    #     sex_temp = 'women-'
    # elif sex == 1:
    #     sex_temp = 'man-'
    # else:
    #     sex_temp = 'women-'
    auto_model = st.sidebar.checkbox("Auto Chose Virtual Human" ,True)
    if auto_model  : auto_select_model(sex, age)
    model_Types = st.sidebar.selectbox("Virtual Human Types", get_photo_types(models_root),key='sb_type_m')
    # selected_model_photo = st.sidebar.selectbox("Virtual Human photos", get_photo_name(models_root, model_Types),key='sb_image_m')
    age_temp = st.sidebar.slider("Age", min_value=5, max_value=80, value=None, step=5, format=None, key='sl_age')
    selected_model_photo=get_image_name(age_temp)+'Y.png'
    # select_model(sex,age_temp)
    img_m = get_photo_file(model_Types, selected_model_photo, models_root)
    #
    # if 'clicked' not in st.session_state:
    #     st.session_state.clicked = False


    # def click_button():
    #     st.session_state.clicked = True
    #     process_imgaes_folder()
    #
    #
    # st.sidebar.button('test', on_click=click_button)
    #
    # if st.session_state.clicked:
    #     # The message and nested widget will remain on the page
    #     st.write('Button clicked!')
    #     # st.sidebar.slider('Select a value')
    # #



    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)


    col1.write("Here is the Patient image  "
               # "sex：%s, age：%d" % ( sex_temp , age if age>0 else -1)
               )
    # img_p.save(get_temp_filename)
    # resized_image_p = img_p.resize((400, 400))
    MAX_SIZE = (400, 400)
    resized_image_p = tools.resize_image(img_p,(1024,1024))
    # resized_image_p=copy.deepcopy(img_p)
    # resized_image_p.thumbnail(MAX_SIZE)
    col1.image(resized_image_p)

    col2.write("Here is the Model image ")
    # resized_image_m=resize_image(img_m)
    # resized_image_m=copy.deepcopy(img_m)
    # resized_image_m.thumbnail(MAX_SIZE)
    # col2.image(resized_image_m)
    col2.image(img_m)
    # print('resized_image_m  ' + str(resized_image_m.size[0:2]))



    img_p=tools.resize_image(img_p)
    fs_temp,fs_landmarks=sw_facephoto(img_m, img_p)
    #eyes_result = faceswap.detect_eyes_single(img_p)
    eyes_result = tools.detect_eyes_single(img_p)


    fs_photo_1,fs_photo_2 = draw_detected_img(fs_temp, fs_landmarks, eyes_result, True)
    # fs_photo_1 = PIL.Image.fromarray(fs_temp)
    # fs_photo_2 = PIL.Image.fromarray(fs_temp)
    with col3 :
        show_detect_1 = st.checkbox("Show Detect Result ", True,key='show_detect_1')
        if not show_detect_1 : fs_photo_1 = PIL.Image.fromarray(fs_temp)
        st.image(fs_photo_1)
        byteIO = BytesIO()
        fs_photo_1.save(byteIO, format='PNG')
        img_byte = byteIO.getvalue()
        st.download_button(
                label="download image",
                data=img_byte,
                file_name="fusion_img_1.png"
        )

    with col4 :
        show_detect_2 = st.checkbox("Show Detect Result ", True,key='show_detect_2')
        if not show_detect_2: fs_photo_2 = PIL.Image.fromarray(fs_temp)
        st.image(fs_photo_2)
        byteIO_2 = BytesIO()
        fs_photo_2.save(byteIO_2, format='PNG')
        img_byte_2 = byteIO_2.getvalue()
        st.download_button(
             label="download image",
             data=img_byte_2,
             file_name="fusion_img_2.png"
        )

