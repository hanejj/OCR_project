
import cv2
#import matplotlib.pyplot as plt
import numpy as np
import os
import time
from win32 import win32api

def get_exe_path():
  if os.name == "posix":
    return os.readlink("/proc/self/exe")
  else:
    return win32api.GetModuleFileName(0)


def derive_graym(impath):
    ''' The intensity value m is calculated as (r+g+b)/3, yet 
        grayscalse will do same operation!
        opencv uses default formula Y = 0.299 R + 0.587 G + 0.114 B
    '''
    # return cv2.imread(impath, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    return cv2.imread(impath, cv2.IMREAD_GRAYSCALE)

def derive_m(img, rimg):
    ''' Derive m (intensity) based on paper formula '''

    (rw, cl, ch) = img.shape
    for r in range(rw):
        for c in range(cl):
            rimg[r,c] = int(np.sum(img[r,c])/3.0)
            
    return rimg

def derive_saturation(img, rimg):
    ''' Derive staturation value for a pixel based on paper formula '''

    s_img = np.array(rimg)
    (r, c) = s_img.shape
    for ri in range(r):
        for ci in range(c):
            #opencv ==> b,g,r order
            s1 = img[ri,ci][0] + img[ri,ci][2]
            s2 = 2 * img[ri,ci][1] 
            if  s1 >=  s2:
                s_img[ri,ci] = 1.5*(img[ri,ci][2] - rimg[ri,ci])
            else:
                s_img[ri,ci] = 1.5*(rimg[ri,ci] - img[ri,ci][0])

    return s_img

def check_pixel_specularity(mimg, simg):
    ''' Check whether a pixel is part of specular region or not'''

    m_max = np.max(mimg) * 0.5
    s_max = np.max(simg) * 0.33

    (rw, cl) = simg.shape

    spec_mask = np.zeros((rw,cl), dtype=np.uint8)
    for r in range(rw):
        for c in range(cl):
            if mimg[r,c] >= m_max and simg[r,c] <= s_max:
                spec_mask[r,c] = 255
    
    return spec_mask

def enlarge_specularity(spec_mask):
    ''' Use sliding window technique to enlarge specularity
        simply move window over the image if specular pixel detected
        mark center pixel is specular
        win_size = 3x3, step_size = 1
    '''

    win_size, step_size = (3,3), 1
    enlarged_spec = np.array(spec_mask)
    for r in range(0, spec_mask.shape[0], step_size):
        for c in range(0, spec_mask.shape[1], step_size):
            # yield the current window
            win = spec_mask[r:r + win_size[1], c:c + win_size[0]]
            
            if win.shape[0] == win_size[0] and win.shape[1] == win_size[1]:
                if win[1,1] !=0:
                    enlarged_spec[r:r + win_size[1], c:c + win_size[0]] = 255 * np.ones((3,3), dtype=np.uint8)

    return enlarged_spec 

# 현재 스크립트가 위치한 디렉토리 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
path_dir_B = os.path.join(script_dir, 'input')  # 입력 이미지 디렉토리: input 폴더
path_dir_C = script_dir  # 출력 이미지 디렉토리: ocr_project.py가 있는 디렉토리

# 출력 디렉토리 생성
os.makedirs(os.path.join(path_dir_C, 'telea'), exist_ok=True)
os.makedirs(os.path.join(path_dir_C, 'ns'), exist_ok=True)
os.makedirs(os.path.join(path_dir_C, 'result'), exist_ok=True)

#exe_path = get_exe_path()
#exe_dir_path = os.path.dirname(exe_path)
#path_dir_B = os.path.join(exe_dir_path, "input")
#path_dir_C = os.path.join(exe_dir_path, "output")

file_list2 = os.listdir(path_dir_B)

k = 0.5

for n in file_list2:
    img_gray = derive_graym(path_dir_B+'/'+n)
#    img_gray = cv2.equalizeHist(img_gray)
    img = cv2.imread(path_dir_B+'/'+n, cv2.IMREAD_COLOR)
#    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    

    size = img.shape[0]
    nu = 0.4

    r, g, b = cv2.split(img)

    r = np.reshape(r, (size * size, 1))
    g = np.reshape(g, (size * size, 1))
    b = np.reshape(b, (size * size, 1))
    I_min = []

    for i in range(0, size * size):
        I_min.append(min(r[i], g[i], b[i]))
    T_v = np.mean(I_min) + nu * np.std(I_min)

    beta_s = (I_min - T_v) * (I_min > T_v)

    IHighlight = np.reshape(beta_s, (size, size))


    r = np.reshape(r, (size, size))
    g = np.reshape(g, (size, size))
    b = np.reshape(b, (size, size))

    r = r - k*IHighlight
    g = g - k*IHighlight
    b = b - k*IHighlight

    im = cv2.merge((r,g,b))

    #img_gray = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    cv2.imwrite(path_dir_C + '/' + n, im)
    
    
    r_img = m_img = np.array(img_gray)

    rimg = derive_m(img, r_img)
    s_img = derive_saturation(img, rimg)
    spec_mask = check_pixel_specularity(rimg, s_img)
    enlarged_spec = enlarge_specularity(spec_mask)
    radius = 12 
    telea = cv2.inpaint(img, enlarged_spec, radius, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(img, enlarged_spec, radius, cv2.INPAINT_NS)

    ########
    cv2.imwrite(path_dir_C + '/telea/' + n,telea)
    cv2.imwrite(path_dir_C + '/ns/' + n,ns)
    
#    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    
    height, width, channel = img.shape
#    plt.style.use('dark_background')
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gray = derive_graym(path_dir_B+'/'+n)
#    gray = cv2.equalizeHist(gray)
    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

#    plt.figure(figsize=(12, 10))
#    plt.imshow(gray, cmap='gray')
    
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        img_blurred, 
        maxValue=255.0, 
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        thresholdType=cv2.THRESH_BINARY_INV, 
        blockSize=19, 
        C=9
    )
#
#    plt.figure(figsize=(12, 10))
#    plt.imshow(img_thresh, cmap='gray')

    
    contours, _ = cv2.findContours(
        img_thresh, 
        mode=cv2.RETR_LIST, 
        method=cv2.CHAIN_APPROX_SIMPLE
    )


    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))


#    plt.figure(figsize=(12, 10))
#    plt.imshow(temp_result)
    
    
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    
        # insert to dict
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

#    plt.figure(figsize=(12, 10))
#    plt.imshow(temp_result, cmap='gray')
    
    
    MIN_AREA = 80
    MIN_WIDTH, MIN_HEIGHT = 2, 8
    MIN_RATIO, MAX_RATIO = 0.25, 1.0

    possible_contours = []

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']
        ratio = d['w'] / d['h']
    
        if area > MIN_AREA \
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
        and MIN_RATIO < ratio < MAX_RATIO:
            d['idx'] = cnt
            cnt += 1
            possible_contours.append(d)
        
    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
         #cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
         cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

#    plt.figure(figsize=(12, 10))
#    plt.imshow(temp_result, cmap='gray')
    
    
    MAX_DIAG_MULTIPLYER = 5 # 5
    MAX_ANGLE_DIFF = 12.0 # 12.0
    MAX_AREA_DIFF = 0.5 # 0.5
    MAX_WIDTH_DIFF = 0.5 # 0.8
    MAX_HEIGHT_DIFF = 0.1 #0.2
    MIN_N_MATCHED = 3 # 3

    def find_chars(contour_list):
        matched_result_idx = []
    
        for d1 in contour_list:
            matched_contours_idx = []
            for d2 in contour_list:
                if d1['idx'] == d2['idx']:
                    continue

                dx = abs(d1['cx'] - d2['cx'])
                dy = abs(d1['cy'] - d2['cy'])

                diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

                distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
                if dx == 0:
                    angle_diff = 90
                else:
                    angle_diff = np.degrees(np.arctan(dy / dx))
                area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
                width_diff = abs(d1['w'] - d2['w']) / d1['w']
                height_diff = abs(d1['h'] - d2['h']) / d1['h']

                if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                    matched_contours_idx.append(d2['idx'])

            # append this contour
            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
            
            # recursive
            recursive_contour_list = find_chars(unmatched_contour)
        
            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx
    
    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    # visualize possible contours
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
#         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

#    plt.figure(figsize=(12, 10))
#    plt.imshow(temp_result, cmap='gray')

    
    
    PLATE_WIDTH_PADDING = 1.3 # 1.3
    PLATE_HEIGHT_PADDING = 1.5 # 1.5
    MIN_PLATE_RATIO = 2
    MAX_PLATE_RATIO = 20

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(matched_result):
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    
        sum_height = 0
        for d in sorted_chars:
            sum_height += d['h']

        plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    
        triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
        triangle_hypotenus = np.linalg.norm(
            np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
            np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
        )
    
        angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
        rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    
        img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
    
        img_cropped = cv2.getRectSubPix(
            img_rotated, 
            patchSize=(int(plate_width), int(plate_height)), 
            center=(int(plate_cx), int(plate_cy))
        )
    
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
            continue
    
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })
    
#        plt.subplot(len(matched_result), 1, i+1)
#        plt.imshow(img_cropped, cmap='gray')
        cv2.imwrite(path_dir_C + '/result/' + n,img_cropped)


