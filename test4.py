import cv2
import numpy as np
import pprint
import name

def read_tmp():
    #テンプレート
    #萬子
    itim = cv2.imread("hai/itim.jpg") #0
    nim = cv2.imread("hai/nim.jpg") #1
    sanm = cv2.imread("hai/sanm.jpg") #2
    yonm = cv2.imread("hai/yonm.jpg") #3
    gom = cv2.imread("hai/gom.jpg") #4
    rokum = cv2.imread("hai/rokum.jpg") #5
    nanam = cv2.imread("hai/nanam.jpg") #6
    hatim = cv2.imread("hai/hatim.jpg") #7
    kyum = cv2.imread("hai/kyum.jpg") #8

    #筒子
    itip = cv2.imread("hai/itip.jpg") #9
    nip = cv2.imread("hai/nip.jpg") #10
    sanp = cv2.imread("hai/sanp.jpg") #11
    yonp = cv2.imread("hai/yonp.jpg") #12
    gop = cv2.imread("hai/gop.jpg") #13
    rokup = cv2.imread("hai/rokup.jpg") #14
    nanap = cv2.imread("hai/nanap.jpg") #15
    hatip = cv2.imread("hai/hatip.jpg") #16
    kyup = cv2.imread("hai/kyup.jpg") #17

    #索子
    itis = cv2.imread("hai/itis.jpg") #18
    nis = cv2.imread("hai/nis.jpg") #19
    sans = cv2.imread("hai/sans.jpg") #20
    yons = cv2.imread("hai/yons.jpg") #21
    gos = cv2.imread("hai/gos.jpg") #22
    rokus = cv2.imread("hai/rokus.jpg") #23
    nanas = cv2.imread("hai/nanas.jpg") #24
    hatis = cv2.imread("hai/hatis.jpg") #25
    kyus = cv2.imread("hai/kyus.jpg") #26

    #字
    ton = cv2.imread("hai/ton.jpg") #27
    nan = cv2.imread("hai/nan.jpg") #28
    sya = cv2.imread("hai/sya.jpg") #29
    pe = cv2.imread("hai/pe.jpg") #30
    haku = cv2.imread("hai/haku.jpg") #31
    hatsu = cv2.imread("hai/hatsu.jpg") #32
    tyun = cv2.imread("hai/tyun.jpg") #33

    hai = [itim,nim,sanm,yonm,gom,rokum,nanam,hatim,kyum,itip,nip,sanp,yonp,gop,rokup,nanap,hatip,kyup,itis,nis,sans,yons,gos,rokus,nanas,hatis,kyus,ton,nan,sya,pe,haku,hatsu,tyun]
    
    return hai


def ruizido(tehai_img,hai):
    count = 0
    match_num = []
    #AKAZE検出器の生成
    detector = cv2.AKAZE_create()

    for hai_i in hai:
        # 特徴量の検出と特徴量ベクトルの計算
        hai_kp, hai_des = detector.detectAndCompute(hai_i, None)
        tehai_kp, tehai_des = detector.detectAndCompute(tehai_img, None)

        #白の処理
        #どちらも白の時
        if len(tehai_kp) <= 5 and len(hai_kp) <= 5:
            match_num.append((10000,count,0.0))
            count += 1
            continue
        #テンプレートが白の時,または手牌が白の時
        if (len(tehai_kp) >= 5 and len(hai_kp) <= 5) or (len(tehai_kp) <= 5 and len(hai_kp) > 5):
            match_num.append((0,count,500.0))
            count += 1
            continue

        #マッチング
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(hai_des,tehai_des)

        #上位30の距離平均
        matches30 = matches[:30]

        distance = 0
        for dis in matches30:
            distance = distance + dis.distance
        dis_av = distance / 30
        match_num.append((len(matches),count,dis_av))
        count += 1

    return match_num
        
def judge(tuple_list):

    return_list = []
    temp_hai = read_tmp()

    for i in tuple_list:

        img = cv2.imread("tri_img/"+(i[0]))
        
        ruizido_list = ruizido(img,temp_hai)
        ruizido_sorted = sorted(ruizido_list, key=lambda x: x[0],reverse=True)

        if i[1] == 100:
            hai = ruizido_sorted[0][1]
            if hai == 18 and ruizido_sorted[0][0] < 400:
                    hai = ruizido_sorted[1][1]
            return_list.append(hai)
            continue

        no = name.name_to_no(i[1])
        rate = i[2]

        akaze_sum = 1.0
        ai_sum = 0.0
        n = 1.0
        p = 1.0

        if i[1] == '2s' or i[1] == '3s':
            if i[2] > 0.4:
                return_list.append(no)
                continue

        #マッチ数
        for k in ruizido_sorted:
            if no == k[1]:
                ai_sum = ai_sum + n
            n += 1.0

        #距離
        dis_sorted = sorted(ruizido_sorted, key=lambda x: x[2])
        for j in dis_sorted:
            if no == j[1]:
                ai_sum = ai_sum + p
            if ruizido_sorted[0][1] == j[1]:
                akaze_sum = akaze_sum + p
            p += 1.0

        if no != ruizido_sorted[0][1]:
            akaze_sum = (akaze_sum / 11.0) + rate
            ai_sum = ai_sum / 11.0
            if akaze_sum < ai_sum:
                hai = ruizido_sorted[0][1]
                if hai == 18 and ruizido_sorted[0][0] < 400:
                    hai = ruizido_sorted[1][1]
            else:
                hai = no
        else:
            hai = no

        return_list.append(hai)
    
    return return_list

    

    
