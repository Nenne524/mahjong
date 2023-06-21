import cv2
import numpy as np
import os

def read_img():
    img_path = input("imageのパス： ") 
    img = cv2.imread("/Users/yamaneami/image_media/detect_images/"+img_path)
    return img

# def richi_judge():
#     richi = input("0:立直していない, 1:立直した →　")
#     return int(richi)

def tri(img):
    #グレースケール
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h,w = img.shape[:2]

    # 閾値の設定
    threshold = 200
    # 二値化(閾値100を超えた画素を255にする)
    ret, img_thresh = cv2.threshold(img_g, threshold, 255, cv2.THRESH_BINARY)
    for iii in range(5):
        img_thresh = cv2.medianBlur(img_thresh, ksize=3)
        kernel = np.ones((5,5), np.uint8)
        img_thresh = cv2.morphologyEx(img_thresh, cv2.MORPH_OPEN, kernel)

    # cv2.imshow("ttt.jpg",img_thresh)
    # cv2.waitKey(0)
    #輪郭を抽出
    contours = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    #輪郭の座標をリストに代入
    x1 = [] #x座標の最小値
    y1 = [] #y座標の最小値
    x2 = [] #x座標の最大値
    y2 = [] #y座標の最大値
    for i in range(1, len(contours)):#i = 1は画像全体の外枠になるのでカウントに入れない
        ret = cv2.boundingRect(contours[i])
        x1.append(ret[0])
        y1.append(ret[1])
        #ret[2]は幅、ret[3]は高さ
        x2.append(ret[0] + ret[2])
        y2.append(ret[1] + ret[3])

    # 輪郭の一番外枠を切り抜き
    x1_min = min(x1)
    y1_min = min(y1)
    x2_max = max(x2)
    y2_max = max(y2)
    cv2.rectangle(img, (x1_min, y1_min), (x2_max, y2_max), (0, 255, 0), 3)

    crop_img = img[y1_min:y2_max, x1_min:x2_max]
    cv2.imwrite("crop.jpg",crop_img)
    return crop_img

def tri_14(img):
    h,w = img.shape[:2]
    w_pt = round(w/14)
    tehai_list = []
    for tri in range(14):
        crop_img = img[0:h, 0+(tri*w_pt):w_pt*(tri+1)]
        tehai_list.append(crop_img)

    return tehai_list


def main():
    #画像データ保存先
    os.makedirs("tri_img", exist_ok=True)

    #検出画像読み込み
    img = read_img()

    #立直の有無
    #richi = richi_judge()

    #トリミング
    tri_img = tri(img)
    #14分割トリミング
    tri14_img = tri_14(tri_img)

    k = 0
    for i in tri14_img:
        cv2.imwrite("tri_img/hai{0}.jpg".format(k),i)
        k += 1

if __name__ == '__main__':
    main()
