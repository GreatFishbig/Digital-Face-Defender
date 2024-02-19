#!/usr/bin/python
# coding=UTF-8
import cv2
import numpy
import uuid
import streamlit as st
import mediapipe as mp
import numpy as np
import  os
import PIL
from PIL import Image,  ImageOps
import scipy
import copy
# import streamlit_modal as modal
# import streamlit.components.v1 as components

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh  # 引入模型

# PREDICTOR_PATH = "/home/matt/dlib-18.16/shape_predictor_68_face_landmarks.dat"
# PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
SCALE_FACTOR = 1
FEATHER_AMOUNT = 5 # between 1-11
COLOUR_CORRECT_BLUR_FRAC = 0.6

# FACE_POINTS = list(range(17, 68))
# MOUTH_POINTS = list(range(48, 61))
# RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_EYE_POINTS = [362,256,253,339,359,387,385,384,398]                        #[464, 441, 444, 265, 449, 451, 412]
# LEFT_BROW_POINTS = [336, 296, 334, 293, 300, 353]  # [285,336,296,334,293,300]
LEFT_BROW_POINTS = [336, 296, 334, 293, 276]     # edit on 2023-11-10 by hychen
RIGHT_EYE_POINTS = [130,110,23,26,133,157,158,159,160]                       #[226, 224, 188, 231, 228, 35]
RIGHT_BROW_POINTS = [113, 70, 63, 105, 66, 107]  # [124,70,63,105,66,107]
IRIS_POINTS = list(range(468, 478))
RIGHT_BROW_POINTS_Checking = [28, 29, 224, 222, 53, 65]
LIGHT_BROW_POINTS_Checking = [258, 259, 442, 444, 295, 283]
RIGHT_EYES_DOWN_ZONE = [124, 35, 31, 228, 229, 230, 231, 232, 245, 168, 8]
LEFT_EYES_DOWN_ZONE = [8, 168, 465, 453, 452, 451, 450, 449, 261, 265] #, 383]  # remove #383  point edit 2023-11-13
WHOLE_FACE_LAYOUT = [68, 104, 69, 108, 151, 337,
                     299, 333, 298, 301, 383, 372,
                     340, 280, 411, 434, 430, 431,
                     262, 428, 199, 208, 32, 211,
                     210, 214, 213, 187, 50,
                     116, 143, 139, 71]
NOSE_ZONE = [128, 217, 209, 203, 167, 164, 423]
ALIGN_POINTS_mp =( LEFT_EYE_POINTS  + LEFT_BROW_POINTS + RIGHT_EYE_POINTS+ RIGHT_BROW_POINTS)# LEFT_BROW_POINTS = list(range(22, 27)),
FACE_LAYOUT_POINT_mp = [WHOLE_FACE_LAYOUT]# RIGHT_EYE_POINTS = list(range(36, 42))
OVERLAY_POINTS_mp = [LEFT_EYE_POINTS  + LEFT_BROW_POINTS +LEFT_EYES_DOWN_ZONE,RIGHT_EYE_POINTS+ RIGHT_BROW_POINTS+RIGHT_EYES_DOWN_ZONE] #, EYES_DOWN_ZONE+NOSE_ZONE
KeyPoints_1 = OVERLAY_POINTS_mp
KeyPoints_2 = [263,473,468,33,474,476,469,471,374,386,145,159,477,475,472,470,263,362,468,133]
KeyPoints_3 = [226,233,56,122]

class TooManyFaces(Exception):
    pass
    # except:
    # st.error("Please make sure that you choose A standard Image ")
    # st.stop()

class NoFaces(Exception):
   pass
   # st.error("Please make sure that you choose A standard Image ")
   # st.stop()

def get_datas(landmarks_img) :
    #
    # 外斜内斜   是 虹膜中心， M，N 是 内眼角 。
    A_L_2 =  landmarks_img[473][0, 0]  -  landmarks_img[362][0, 0]  #左眼瞳孔到内眼角
    B_R_2 =  landmarks_img[133][0, 0]  -  landmarks_img[468][0, 0]  #右眼瞳孔到内眼角

    # 左眼瞳孔到左眼外角 ，右眼瞳孔到右眼内角
    C_L_2 = landmarks_img[263][0, 0]  -  landmarks_img[473][0, 0]   #左眼瞳孔到左眼外角
    D_R_2 = landmarks_img[133][0, 0]  -  landmarks_img[468][0, 0]    #右眼瞳孔到右眼内角

    #眼睑下垂
    OP_L = landmarks_img[374][0, 1] - landmarks_img[386][0, 1]  # left 下眼睑中点-上眼睑中点
    OP_R = landmarks_img[145][0, 1] - landmarks_img[159][0, 1]  # right 下眼睑中点-上眼睑中点
    MN_L = landmarks_img[474][0, 0] - landmarks_img[476][0, 0]  # 左虹膜x宽
    MN_R = landmarks_img[469][0, 0] - landmarks_img[471][0, 0]  # 右虹膜x宽

    D = abs(A_L_2-B_R_2)
    L = A_L_2
    R = B_R_2

    MN_R = landmarks_img[133][0, 0] - landmarks_img[33][0, 0]  # 右眼裂x宽
    PQ_L = landmarks_img[263][0, 0] - landmarks_img[362][0, 0]  # 左眼裂x宽
    AB_L = landmarks_img[374][0, 1] - landmarks_img[386][0, 1]  # left 下眼睑中点-上眼睑中点
    CD_R = landmarks_img[145][0, 1] - landmarks_img[159][0, 1]  # right 下眼睑中点-上眼睑中点
    GH_L = landmarks_img[474][0, 0] - landmarks_img[476][0, 0]  # 左虹膜x宽
    EF_R = landmarks_img[469][0, 0] - landmarks_img[471][0, 0]  # 右虹膜x宽


    ABdivGH=round(AB_L / GH_L,3)
    CDdivEF=round(CD_R / EF_R,3)


    # CDdivMN=round(CD_R / MN_R,3)
    # ABdivPQ=round(AB_L / PQ_L,3)
    #row_contents =[AB_L,CD_R,EF_R,GH_L,MN_R,PQ_L,ABdivGH,CDdivEF,CDdivMN,ABdivPQ]
    #row_contents =[A_L_2,B_R_2,abs(A_L_2-B_R_2)]+[C_L_2,D_R_2,abs(C_L_2-D_R_2)]+[OP_L,MN_L,OP_L/MN_L,OP_R,MN_R,OP_R/MN_R]

    row_contents=[D,L,R,ABdivGH,CDdivEF]
    return row_contents

def detect_eyes_single(IMG_PATH):
    path1_ = IMG_PATH
    try:
        landmarks_img = find_img_data(path1_)
    except Exception as r:
        print(path1_, r)
        landmarks_img = None

    if landmarks_img is None:
        print(path1_ + ' NOT find data !')
        # 错误处理
        return '-1'   #can''t detect face data'
    # 如 图片检测错误， 下面的眼部地标数据获取 操作都会出错的。
    # row_contents = plot_eyes_lines.get_datas(landmarks_img)
    row_contents = get_datas(landmarks_img)
    D = row_contents[0]
    L = row_contents[1]
    R = row_contents[2]
    ABdivGH_L = row_contents[3]
    CDdivEF_R = row_contents[4]
    # 置信区间给定值
    Gmax = 11
    Lmax = 60
    Lmin = 46
    Rmax = 62.4
    Rmin = 45
    D_Right=   [0.558,0.870]
    D_Left=    [0.569,0.872]
    result1 = "0"
    result2 = "0"
    result3 = '0'
    result4 = '0'
    # 判断是否斜视  ,
    if D >Gmax:  #  EDIT BY HYCHEN 2023/12/11 !!
        st.info('This Picture maybe not a standard image.[pose?] ', icon="ℹ️")

    if L > Lmax:
        result1="1" #"Left-Exotropia"
    if L < Lmin:
        result1='2' #"Left-Esotropia"
    if R > Rmax:
        result2='1' #"Right-Exotropia"
    if R < Rmin:
        result2='2' #"Right-Esotropia"
    #else:
    #    result = result1+result2


    if  ABdivGH_L > D_Left[1] : result3 = '2'    # 左眼睑下垂
    elif ABdivGH_L< D_Left[0] : result3 = '1'    # 左甲状腺眼
    if  CDdivEF_R > D_Right[1] :  result4 = '2'    # 右眼睑下垂
    elif CDdivEF_R< D_Right[0] :  result4 = '1'    # 右甲状腺眼


    result =format(result1+result2+result3+result4)

    #print(path1_+"  "+result)
    return(result)

def find_img_data (img_file_name):
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,  # TRUE:静态图片/False:摄像头实时读取
        refine_landmarks=True,  # 使用Attention Mesh模型    获得iris landmarks .

        max_num_faces=1,  # 最大人脸检测数
        min_detection_confidence=0.3,  # 置信度阈值，监测是否检测人脸成功
        min_tracking_confidence=0.1,  # 追踪阈值，若高于上值则将再次进行人脸检测
    )
    _, landmarks= read_im_and_landmarks_mp(img_file_name, face_mesh)
    return landmarks


@st.cache_resource    #缓存模型数据
def get_face_mesh():
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,  # TRUE:静态图片/False:摄像头实时读取
        refine_landmarks=True,  # 使用Attention Mesh模型  false 眼睛保持原形状 ， ture 眼睛将改变为目标的形状  ?   获得iris landmarks ?

        max_num_faces=1,  # 最大人脸检测数
        min_detection_confidence=0.3,  # 置信度阈值，监测是否检测人脸成功
        min_tracking_confidence=0.1,  # 追踪阈值，若高于上值则将再次进行人脸检测
    )
    return face_mesh

def resize_image( Aimage:PIL.Image,new_size:tuple=(1024,1024)) -> PIL.Image:
    resized_image = copy.deepcopy(Aimage)
    # old_size=Aimage.size
    # preserve aspect ratio
    x, y =Aimage.size
    if x > new_size[0]:
        y = max(round(y * new_size[0] / x), 1)
        x = round(new_size[0])
    if y > new_size[1]:
        x = max(round(x * new_size[1] / y), 1)
        y = round(new_size[1])

    resized_image.thumbnail(new_size) 
    img_array = copy.deepcopy(np.asarray(resized_image))
    if img_array[x:,y:].shape[0:2]!=(0,0) :
        padded_x = round((new_size[0]-x ) / 2)
        padded_y = round((new_size[1]-y) / 2)
        # img_array.resize((new_size[0],new_size[1],4))
        # img_array[x:,y:,0:4]=np.zeros((x,y,4))
        # im = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        # im = cv2.resize(im, (new_size[0],new_size[1]))
        padded_array=np.pad(img_array,((padded_y,padded_y),(padded_x,padded_x),(0,0)),mode='constant',constant_values=((255,255),(255,255),(0,0)))
        tmp_img= PIL.Image.fromarray(padded_array, Aimage.mode)
        # print('tmp_img  '+str(tmp_img.size[0:2]))
        return tmp_img

    # print('resized_image  '+str(resized_image.size[0:2]))
    return resized_image

def read_im_and_landmarks_mp(image_name, face_mesh):
    if isinstance(image_name,str):
       im=cv2.imread(image_name,cv2.IMREAD_COLOR)
    else :
       im =  cv2.cvtColor(np.asarray(image_name), cv2.COLOR_RGB2BGR)

    #im = cv2.imread(img, cv2.IMREAD_COLOR)

    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks_mp(im, face_mesh)

    return im, s

def get_landmarks_mp(im, face_mesh):  # 与get_landmarks 函数等效
    # image = cv2.imread(im)
    results = face_mesh.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    landmark_list = []
    image_rows, image_cols, _ = im.shape  # 获得图片信息
    if results.multi_face_landmarks  is None :
        st.error("NO face detect in your  Image ")
        return    #raise NoFaces
    # if len(results.multi_face_landmarks) > 1:
    #     raise TooManyFaces
    if len(results.multi_face_landmarks) == 0:
        st.error("NO face detect in your  Image ")
        return  # raise NoFaces


    # if (results.multi_face_landmarks != None):
    for face_landmarks in results.multi_face_landmarks:
        for l in face_landmarks.landmark:
            pt = mp_drawing._normalized_to_pixel_coordinates(l.x, l.y, image_cols, image_rows)
            landmark_list.append(pt)

    return numpy.matrix(landmark_list)


def get_temp_filename(mark='',filename_ori=''):
    uuid_str = uuid.uuid4().hex
    return  os.path.join('.', 'temp', os.path.basename(filename_ori) + '_' + uuid_str + mark + '.png')


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)

    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = numpy.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)

def get_face_mask(im, landmarks, keypoints=OVERLAY_POINTS_mp, with_gauss=True):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
    # for group in OVERLAY_POINTS:  # comm 12-18
    # 给定几组点， 分别划出对边形，合并得到mask
    for group in keypoints:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)   # 1-->255

    #不带边缘羽化的mask 23-8-20
    im = numpy.array([im, im, im]).transpose((1, 2, 0))
    im_No_gauss = im

    # maybe is working . 23-8-20  带边缘羽化的mask
    im= (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)
    im_gauss=im

    #
    if with_gauss :
       return im_gauss
    else :
       return im_No_gauss

def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)  #cv2.WARP_INVERSE_MAP    , INTER_NEAREST
    return output_im

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
        numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    # function correct_colours can produce some bad case in which there are some white part on the edge of result face when the two faces have a large color difference.
    # You    could    try changing the diameter of GaussianBlur
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
            im2_blur.astype(numpy.float64))

def image_align(src_file,dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=False):
    # Align function from FFHQ dataset pre-processing step
    # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

    lm = np.array(face_landmarks)
    try :
      lm_eye_right = [lm[362],lm[385],lm[386],lm[263],lm[373],lm[380]]
      lm_eye_left = [lm[33],lm[159],lm[158],lm[133],lm[153],lm[144]]  # left-clockwise
    except :
       st.error("Please make sure that you choose a standard Image ")
       st.stop()
    # lm_mouth_outer = lm[48: 60]  # left-clockwise
    # lm_mouth_inner = lm[60: 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm[308]    #lm_mouth_outer[0]
    mouth_right = lm[61]   #lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_file)
    img = ImageOps.exif_transpose(img)  # 处理图片横拍， 显示时自动旋转回来。  2023-1-10

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink

    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]),
            min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # Pad.
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
           int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0),
           max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]),
                          1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    # Transform.
    img = img.transform((transform_size, transform_size), PIL.Image.Transform.QUAD, (quad + 0.5).flatten(), PIL.Image.Resampling.BILINEAR,fillcolor=(255,255,255))
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.Resampling.LANCZOS)

    # Save aligned image.
    # img.save(dst_file, 'PNG')
    # return  dst_file
    return img

def fs(file1, file2):
    # im1, landmarks1 = read_im_and_landmarks(file1)
    # im2, landmarks2 = read_im_and_landmarks(file2)     # comm 12-18

    face_mesh=get_face_mesh()
    # file1是model图片， file2是患者图片
    im1, landmarks1 = read_im_and_landmarks_mp(file1,face_mesh)       #    im1 type is  [ndarray],  file1, face_mesh)
    im2, landmarks2 = read_im_and_landmarks_mp(file2,face_mesh)


    # 图片对齐  # 特殊情况下 人脸无法找到。
    im2_tmp=get_temp_filename('图片对齐_1')  # './temp/align_tmp.jpg'   # 临时文件名，， 需改进使用随机数保证不同名。 2023-4-18

    #align图片对齐代码不稳定， 计划暂时停用 2023-4-27
    # 将模特图片进行旋转使得图片人物居中 。
    #gmp_faceswap.image_align(file2,im2_tmp,landmarks2)      #gmp_faceswap.image_align(file2,im2_tmp,landmarks2)
    im2_im = image_align(file2, im2_tmp, landmarks2)  # gmp_faceswap.image_align(file2,im2_tmp,landmarks2)


    # 将居中处理后的图片重新计算地标
    im2, landmarks2 = read_im_and_landmarks_mp(im2_im, face_mesh)

    # M = transformation_from_points(landmarks1[ALIGN_POINTS],
    #                                landmarks2[ALIGN_POINTS])    # comm 12-18
    # 根据ALIGN_POINTS_mp 对应的点计算转置矩阵 。
    # 获得整个脸部的转换矩阵 M 是 患者图片转换到模特的 矩阵
    M = transformation_from_points(landmarks1[ALIGN_POINTS_mp],
                                   landmarks2[ALIGN_POINTS_mp])

    # 获得模特的mask  , 两个版本的mask
    mask = get_face_mask(im2, landmarks2)
    # cv2.imwrite('mask_im2.jpg', mask)

    warped_mask = warp_im(mask, M, im1.shape)
    # cv2.imwrite('warped_mask_im1.jpg', warped_mask)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],axis=0)
    # cv2.imwrite('./temp/combined_mask.jpg', combined_mask)   2023-4

    # 旋转（等于目标区域的角度）并调整尺寸（等于目标图片）以适应目标图片
    warped_im2 = warp_im(im2, M, im1.shape)
    # cv2.imwrite(get_temp_filename('_2_调整姿势匹配模特'), warped_im2)

    #  调整图片颜色 ， 眼部变黑的原因 ！ im1 model图片融合到warped_im2 患者的图片了！
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    # cv2.imwrite(get_temp_filename('_3_调整颜色匹配模特'), warped_corrected_im2)
    # 此时 获得图片output_im 是直接粘帖患者图片到模特相应位置， 最能反映患者的颜色，皮肤纹理。
    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    output_im = np.maximum(output_im, 0)
    output_im = np.minimum(output_im, 255)
    output_im = output_im.astype(np.uint8)
    # 此时 获得图片output_im 是直接粘帖患者图片到模特相应位置， 最能反映患者的颜色，皮肤纹理
    # temp_output=cv2.cvtColor(output_im,cv2.COLOR_BGR2RGB)

    # output_im_seamlessClone = output_im  #temp_output  #不使用无缝融合方式。 直接使用亮度融合图片 。
    # output_file_seamless=get_temp_filename(f'_4_融合结果_高斯={FEATHER_AMOUNT}')
    # cv2.imwrite(output_file_seamless, output_im_seamlessClone)
    # annotated_output_im,annotated_landmarks2= read_im_and_landmarks_mp(output_file_seamless, face_mesh)
    # annotated_output_im,annotated_landmarks2= read_im_and_landmarks_mp(output_im_seamlessClone, face_mesh)
    annotated_output_im, annotated_landmarks2 = read_im_and_landmarks_mp(output_im, face_mesh)

    #return output_file_seamless, annotated_landmarks2, temp_output
    # return output_im_seamlessClone, annotated_landmarks2, temp_output
    return annotated_output_im, annotated_landmarks2    #, temp_output