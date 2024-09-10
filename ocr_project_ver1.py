import cv2
import numpy as np
import os


def derive_graym(impath):
    return cv2.imread(impath, cv2.IMREAD_GRAYSCALE)


def derive_m(img, rimg):
    rimg[:] = np.mean(img, axis=2)
    return rimg


def derive_saturation(img, rimg):
    b, g, r = cv2.split(img)
    s_img = np.maximum(r + b - 2 * g, 0)
    return 1.5 * np.abs(s_img - rimg)


def pix_specularity(mimg, simg):
    m_max = np.max(mimg) * 0.5
    s_max = np.max(simg) * 0.33
    return np.where((mimg >= m_max) & (simg <= s_max), 255, 0).astype(np.uint8)


def enlarge_specularity(spec_mask):
    ''' Use sliding window technique to enlarge specularity
        simply move window over the image if specular pixel detected
        mark center pixel as specular
        win_size = 3x3, step_size = 1
    '''

    win_size, step_size = (3, 3), 1
    enlarged_spec = np.zeros_like(spec_mask)
    for r in range(0, spec_mask.shape[0] - win_size[0] + 1, step_size):
        for c in range(0, spec_mask.shape[1] - win_size[1] + 1, step_size):
            win = spec_mask[r:r + win_size[1], c:c + win_size[0]]

            if win.shape[0] == win_size[0] and win.shape[1] == win_size[1]:
                if win[1, 1] != 0:
                    enlarged_spec[r:r + win_size[1], c:c + win_size[0]] = 255

    return enlarged_spec


# 현재 스크립트가 위치한 디렉토리 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
path_dir_B = os.path.join(script_dir, 'input')  # 입력 이미지 디렉토리: input 폴더
path_dir_C = script_dir  # 출력 이미지 디렉토리: ocr_project.py가 있는 디렉토리

# 출력 디렉토리 생성
os.makedirs(os.path.join(path_dir_C, 'telea'), exist_ok=True)
os.makedirs(os.path.join(path_dir_C, 'ns'), exist_ok=True)
os.makedirs(os.path.join(path_dir_C, 'result'), exist_ok=True)

# 입력 디렉토리 내의 파일 리스트 가져오기
file_list2 = os.listdir(path_dir_B)

k = 0.5

for n in file_list2:
    img_path = os.path.join(path_dir_B, n)
    img_gray = derive_graym(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if img is None:
        print(f"Error: Unable to load image {img_path}")
        continue

    height, width, _ = img.shape
    nu = 0.4

    r, g, b = cv2.split(img)
    I_min = np.minimum(np.minimum(r, g), b)
    T_v = np.mean(I_min) + nu * np.std(I_min)

    beta_s = np.maximum(I_min - T_v, 0)

    # Ensure beta_s has the correct shape
    if beta_s.shape != (height, width):
        beta_s = np.resize(beta_s, (height, width))

    IHighlight = beta_s

    r = r.astype(np.float64)
    r -= k * IHighlight
    r = np.clip(r, 0, 255).astype(np.uint8)

    g = g.astype(np.float64)
    g -= k * IHighlight
    g = np.clip(g, 0, 255).astype(np.uint8)

    b = b.astype(np.float64)
    b -= k * IHighlight
    b = np.clip(b, 0, 255).astype(np.uint8)

    im = cv2.merge((r, g, b))
    cv2.imwrite(os.path.join(path_dir_C, n), im)

    r_img = derive_m(img, np.array(img_gray))
    s_img = derive_saturation(img, r_img)
    spec_mask = pix_specularity(r_img, s_img)
    enlarged_spec = enlarge_specularity(spec_mask)
    radius = 12
    telea = cv2.inpaint(img, enlarged_spec, radius, cv2.INPAINT_TELEA)
    ns = cv2.inpaint(img, enlarged_spec, radius, cv2.INPAINT_NS)

    cv2.imwrite(os.path.join(path_dir_C, 'telea', n), telea)
    cv2.imwrite(os.path.join(path_dir_C, 'ns', n), ns)

    gray = derive_graym(img_path)
    gray = cv2.equalizeHist(gray)
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

    blur_img = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

    img_thresh = cv2.adaptiveThreshold(
        blur_img,
        maxValue=255.0,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=19,
        C=9
    )

    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
    rst_tmp = np.zeros_like(img)
    cv2.drawContours(rst_tmp, contours, -1, (255, 255, 255))

    contours_dict = [{'contour': c, 'x': cv2.boundingRect(c)[0], 'y': cv2.boundingRect(c)[1],
                      'w': cv2.boundingRect(c)[2], 'h': cv2.boundingRect(c)[3],
                      'cx': cv2.boundingRect(c)[0] + cv2.boundingRect(c)[2] / 2,
                      'cy': cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] / 2}
                     for c in contours]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(rst_tmp, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2)
        })

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

    height, width, channel = img.shape
    rst_tmp = np.zeros((height, width, channel), dtype=np.uint8)

    for d in possible_contours:
        cv2.rectangle(rst_tmp, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

    # hyperparameter
    MAX_DIAG_MULTIPLYER = 5
    MAX_ANGLE_DIFF = 12.0
    MAX_AREA_DIFF = 0.5
    MAX_WIDTH_DIFF = 0.5
    MAX_HEIGHT_DIFF = 0.1
    MIN_N_MATCHED = 3


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

            # coutours mesh orderly
            matched_contours_idx.append(d1['idx'])

            if len(matched_contours_idx) < MIN_N_MATCHED:
                continue

            matched_result_idx.append(matched_contours_idx)

            unmatched_contour_idx = []
            for d4 in contour_list:
                if d4['idx'] not in matched_contours_idx:
                    unmatched_contour_idx.append(d4['idx'])

            unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

            recursive_contour_list = find_chars(unmatched_contour)

            for idx in recursive_contour_list:
                matched_result_idx.append(idx)

            break

        return matched_result_idx


    result_idx = find_chars(possible_contours)

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    rst_tmp = np.zeros((height, width, channel), dtype=np.uint8)

    for r in matched_result:
        for d in r:
            cv2.rectangle(rst_tmp, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                          thickness=2)

    cv2.imwrite(os.path.join(path_dir_C, 'result', n), rst_tmp)
