import torch
import cv2
import os
import numpy as np
import pprint
import test4
import name

def check_tuple_list(tuple_list, a):
    count = 0
    for item in tuple_list:
        if item[0] == a:
            count += 1
    return count

#学習済みモデルのパス
model_path = 'best.pt'

#GPUを使用する場合
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#学習済みモデルの読み込み
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

#確定結果を格納
results = []

#候補を格納
kouho = []

os.makedirs("result_images", exist_ok = True)

#推論実行
for i in range(14):
    detect_img = cv2.imread("tri_img/hai{0}.jpg".format(i))
    result = model(detect_img)

    # 推論結果のバウンディングボックスを描画
    imgs = result.render()

    #保存
    cv2.imwrite("result_images/hai{0}.jpg".format(i),imgs[0])

    df = result.pandas().xyxy[0]
    # 検出されたクラスと信頼度を表示
    for index, row in df.iterrows():
        if row["confidence"] >= 0.6:   
            if check_tuple_list(results, 'hai{0}.jpg'.format(i)) == 0:       
                results.append(("hai{0}.jpg".format(i),row["name"],row["confidence"]))
            else:
                jg_1 = results.pop()
                if jg_1[2] > row["confidence"]:
                    results.append(jg_1)
                else:
                    results.append(("hai{0}.jpg".format(i),row["name"],row["confidence"]))

        if row["confidence"] >= 0.3 and row["confidence"] < 0.6:
            if check_tuple_list(kouho, 'hai{0}.jpg'.format(i)) == 0:
                if check_tuple_list(results, 'hai{0}.jpg'.format(i)) == 0:
                    kouho.append(("hai{0}.jpg".format(i),row["name"],row["confidence"]))
            else:
                jg_2 = kouho.pop()
                if jg_2[2] > row["confidence"]:
                    kouho.append(jg_2)
                else:
                    kouho.append(("hai{0}.jpg".format(i),row["name"],row["confidence"]))

    
    if check_tuple_list(results, "hai{0}.jpg".format(i)) == 0 and check_tuple_list(kouho, "hai{0}.jpg".format(i)) == 0:
        kouho.append(("hai{0}.jpg".format(i),100,100))

return_list = test4.judge(kouho)

img_name = []
for y in return_list:
    img_name.append(name.no_to_name(y))

for x in range(len(kouho)):
    results.append((kouho[x][0],img_name[x]))

results = sorted(results,key = lambda x: x[0])
pprint.pprint(results)



