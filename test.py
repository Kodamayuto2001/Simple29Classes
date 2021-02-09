import my_system_lib as ml
import sys
import cv2

def name():
    s,name_id = k.real_time_haar()
    sys.stdout.write("\r {}さんです。".format(s))
    return s

if __name__ == "__main__":
    # 推論クラスのインスタンス化
    k = ml.Suiron()


    while True:
        # 名前の戻り値
        name()
        
        # サーモセンサ画像が保存されているパスを指定（相対パス）
        k.imshow("lena.jpg")


        