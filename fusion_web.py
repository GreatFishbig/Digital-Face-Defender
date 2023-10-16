import os
import uuid
import PIL.Image
import streamlit as st
from PIL import Image
from io import BytesIO
#import faceswap
# import plot_eyes_lines
import  tools
import infer
import cv2
import copy

patients_root= "./patients"
models_root="./models"
tmp_path="./temp"

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
    temp_name = "006"
    if age in [0, 5]:  temp_name = "009"
    if age in [5, 10]:  temp_name = "008"
    if age in [10, 15]:  temp_name = "007"
    if age in [15, 20]:  temp_name = "006"
    if age in [20, 30]:  temp_name = "005"
    if age in [30, 40]:  temp_name = "004"
    if age in [40, 50]:  temp_name = "003"
    if age in [50, 60]:  temp_name = "002"
    if age in [60, 70]:  temp_name = "001"
    if age in [70, 80]:  temp_name = "000"
    return    temp_name


def auto_select_model(sex, age):
    if  sex == 1 :
        st.session_state.sb_type_m="Man.5-80"
    else :
        st.session_state.sb_type_m ="women.5-80"

    temp_name= get_image_name(age)
    st.session_state["sb_image_m"]= temp_name +".png"




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

@st.cache_data
def get_photo_file(dir:str,name:str,typedir:str) -> Image :
    return Image.open(os.path.join(typedir,dir,name))


def draw_detected_img(faceimgae,landmarks_img,detect_result='0000',to_web=False):
    # faceimage 融合后的图片， landmarks_img 融合后的坐标集合， detect_result 判断结果， 是否在web上显示
    if isinstance(faceimgae,str):
       img_1=cv2.imread(faceimgae)
       img_2=copy.deepcopy(img_1)
    else :
       img_1 = faceimgae
       img_2 = copy.deepcopy(img_1)

    RT_point = (landmarks_img[46][0, 0], landmarks_img[46][0, 1])
    RB_point = (landmarks_img[188][0, 0], landmarks_img[188][0, 1])
    LT_point = (landmarks_img[285][0, 0], landmarks_img[276][0, 1])
    LB_point = (landmarks_img[276][0, 0], landmarks_img[188][0, 1])

    L_Color = (0,0,0)
    R_Color = (0,0,0)
    if detect_result[0] == '1' or detect_result[0]=='2' : L_Color_1=[255,0,0]
    else:   L_Color_1=[0,255,0]
    if detect_result[1] == '1' or detect_result[1] =='2' : R_Color_1=[255,0,0]
    else : R_Color_1 = [0,255,0]
    cv2.rectangle(img_1, RT_point, RB_point, R_Color_1, 3)
    cv2.rectangle(img_1, LT_point, LB_point, L_Color_1, 3)
    cv2.putText(img_1, 'Squint', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    image_1 = PIL.Image.fromarray(img_1)

    if detect_result[2] == '1' or detect_result[2] =='2' : L_Color_2=[255,0,0]
    else : L_Color_2 = [0,255,0]
    if detect_result[3] == '1' or detect_result[3] =='2' : R_Color_2=[255,0,0]
    else : R_Color_2 = [0,255,0]
    cv2.rectangle(img_2, RT_point, RB_point, R_Color_2, 3)
    cv2.rectangle(img_2, LT_point, LB_point, L_Color_2, 3)
    cv2.putText(img_2, 'Ptosis OR  Grave''s disease', (10, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200),5)
    image_2 = PIL.Image.fromarray(img_2)
    return image_1,image_2

def get_temp_filename():
    uuid_str = uuid.uuid4().hex
    return  os.path.join(tmp_path,uuid_str+'.png')

def sw_facephoto(img_p,img_m):
    filename_p=get_temp_filename()
    filename_m= get_temp_filename()
    img_p.save(filename_p)
    img_m.save(filename_m)
    fusion_file,fusion_landmarks,Not_seamless_photo=tools.fs(filename_p,filename_m)    #  test orig is : filename_p,filename_m
    # img=PIL.Image.open(fusion_file,mode='r')
    os.remove(filename_p)
    os.remove(filename_m)
    # os.remove(fusion_file)
    # 返回的是内存图片
    #return img,Not_seamless_photo,fusion_landmarks
    return fusion_file,Not_seamless_photo,fusion_landmarks


if __name__ == '__main__':
    # model = load_model()
    # index_to_class_label_dict = load_index_to_label_dict()
    # all_image_files = load_s3_file_structure()
    # types_of_birds = sorted(list(all_image_files['test'].keys()))
    # types_of_birds = [bird.title() for bird in types_of_birds]
    # types_of_Patients=

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
    if sex == 0:
        sex_temp = 'Woman'
    elif sex == 1:
        sex_temp = 'Man'
    else:
        sex_temp = "N/A"
    auto_model = st.sidebar.checkbox("Auto Chose Virtual Human" ,True)
    if auto_model  : auto_select_model(sex, age)
    model_Types = st.sidebar.selectbox("Virtual Human Types", get_photo_types(models_root),key='sb_type_m')
    selected_model_photo = st.sidebar.selectbox("Virtual Human photos", get_photo_name(models_root, model_Types),key='sb_image_m')
    img_m = get_photo_file(model_Types, selected_model_photo, models_root)


    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)


    col1.write("Here is the Patient image  "
               # "sex：%s, age：%d" % ( sex_temp , age if age>0 else -1)
               )


    # img_p.save(get_temp_filename)
    resized_image_p = img_p.resize((400, 400))
    col1.image(resized_image_p)
    col2.write("Here is the Model image ")
    resized_image_m = img_m.resize((400, 400))
    col2.image(resized_image_m)


    _,fs_temp,fs_landmarks=sw_facephoto(img_m, img_p)
    #eyes_result = faceswap.detect_eyes_single(img_p)
    eyes_result = tools.detect_eyes_single(img_p)

    fs_photo_1,fs_photo_2 = draw_detected_img(fs_temp, fs_landmarks, eyes_result, True)
    with col3 :
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
        st.image(fs_photo_2)
        byteIO_2 = BytesIO()
        fs_photo_2.save(byteIO_2, format='PNG')
        img_byte_2 = byteIO_2.getvalue()
        st.download_button(
             label="download image",
             data=img_byte_2,
             file_name="fusion_img_2.png"
        )

