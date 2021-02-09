import my_system_lib as ml
import sys
import cv2

def name():
    name_id = k.real_time_haar()
    sys.stdout.write("\r {}さんです。".format(name_id))
    return name_id

if __name__ == "__main__":
    # 推論クラスのインスタンス化
    k = ml.Suiron()


    while True:
        # 名前の戻り値
        name()
        
        # サーモセンサ画像が保存されているパスを指定（相対パス）
        k.imshow("lena.jpg")


        