import os 
import sys
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms

class Net(torch.nn.Module):
    def __init__(self,num,inputSize,Neuron):
        super(Net,self).__init__()
        self.iSize = inputSize
        self.fc1 = torch.nn.Linear(self.iSize*self.iSize,Neuron)
        self.fc2 = torch.nn.Linear(Neuron,num)
        
    def forward(self,x):
        x = x.view(-1,self.iSize*self.iSize)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)


class Suiron:
    r"""
        顔認証機能
            real_time_haar関数
    """
    CAP_CHANNEL         =   0     #   0か1にしてください
    WINDOW_WIDTH        =   1920
    WINDOW_HEIGHT       =   1080
    CASCADEPATH         =   "haarcascades/haarcascade_frontalface_default.xml"
    PATH                =   "models/29classes.pt"
    inputSize           =   160
    model               =   Net(num=29,inputSize=inputSize,Neuron=320)
    
    IS_CAP_INIT         =   0
    FRAME_WIDTH         =   600
    FRAME_HEIGHT        =   600
    x                   =   100
    y                   =   100
    BODY_TEMP           =   36.5
    BODY_TEMP_SAFE      =   (255,0,0)
    BODY_TEMP_OUT       =   (255,0,255)
    COLOR               =   BODY_TEMP_SAFE

    CNT             =   0
    CNT_MAX         =   50
    PROGRESS_BAR_LEN=   100
    DELAY_MSEC      =   1

    MOJI_OOKISA         =   1.0
    percent             =   0
    name                =   "-------"
    MOJI_HYOUJI         =   "6.0"

    NAME = [
        "ando",
        "enomaru",
        "hamada",
        "higashi",
        "kataoka",
        "kawano",
        "kodama",
        "masuda",
        "matsuzaki",
        "matsui",
        "miyatake",
        "mizuki",
        "nagao",
        "okamura",
        "ooshima",
        "ryuuga",
        "shinohara",
        "soushi",
        "suetomo",
        "takemoto",
        "tamejima",
        "teppei",
        "toriyabe",
        "tsuchiyama",
        "uemura",
        "wada",
        "watanabe",
        "yamaji",
        "yamashita",
        #   29番目をguestとする
        "guest"     
    ]

    list_label = [
        [] for i in NAME
    ]
    list_percent = [
        [] for i in NAME
    ]

    

    def __init__(self):
        #   カメラの初期設定
        #   モデルのロード
        self.cap = cv2.VideoCapture(self.CAP_CHANNEL)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,   self.WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  self.WINDOW_HEIGHT)
        self.cascade = cv2.CascadeClassifier(self.CASCADEPATH)
        self.model.load_state_dict(torch.load(self.PATH))
        self.model.eval()

        for i,_ in enumerate(self.NAME):
            self.list_label[i] = 0
            self.list_percent[i] = 0

    def __del__(self):
        #   終了処理
        self.cap.release()
        cv2.destroyAllWindows()

    def real_time_haar(self):
        #   画像を読み込む
        success,img = self.cap.read()
        try:
            imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        except cv2.error:
            if self.IS_CAP_INIT == 0:
                self.CAP_CHANNEL    ^=  1
                self.cap = cv2.VideoCapture(self.CAP_CHANNEL)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,   self.WINDOW_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  self.WINDOW_HEIGHT)
            success,img = self.cap.read()
            imgGray     = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            self.IS_CAP_INIT        =   1
        

        #   カスケード分類器で顔を抽出
        imgResult   = img.copy()
        facerect    = self.cascade.detectMultiScale(imgGray,scaleFactor=1.1,minNeighbors=2,minSize=(200,200))


        #   顔セット用の正方形を正しい位置にするために必要
        H,W,C = img.shape
        self.x = int((W - self.FRAME_WIDTH)/2)
        self.y = int((H - self.FRAME_HEIGHT)/2)

        #   もし体温が37.0度以上の時は赤、未満は青
        if self.BODY_TEMP >= 37.0:
            self.COLOR  =   self.BODY_TEMP_OUT
        else:
            self.COLOR  =   self.BODY_TEMP_SAFE

        cv2.putText(img,self.MOJI_HYOUJI,(self.x+self.FRAME_WIDTH+40,40),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)

        cv2.rectangle(img,(self.x,self.y),(self.x+self.FRAME_WIDTH,self.y+self.FRAME_HEIGHT),self.COLOR,thickness=10)
        #   もし顔と識別できたら
            #   顔認識機能で誰かを分類する
        if len(facerect) > 0:
            for (x,y,w,h) in facerect:
                cv2.rectangle(imgResult,(x,y),(x+w,y+h),self.COLOR,thickness=2)
                imgTrim = img[y:y+h,x:x+w]

                #   ラベル・パーセント・最大ラベル・平均パーセンテージ
                label,percent,max_label,avg_percent = self.maesyori_suiron(imgTrim)

                cv2.line(
                        img,
                        (self.x+self.FRAME_WIDTH+50,                        int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        (self.x+self.FRAME_WIDTH+50+self.PROGRESS_BAR_LEN,  int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        (204,204,204),
                        15
                )
                if self.CNT == 0:
                    self.percent    = avg_percent
                    self.name = str(self.NAME[max_label])
                else:
                    cv2.putText(img,"Please",(self.x+self.FRAME_WIDTH+40,int((self.y+self.FRAME_HEIGHT)/2)+40),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                    cv2.putText(img,"wait.",(self.x+self.FRAME_WIDTH+40,int((self.y+self.FRAME_HEIGHT)/2)+40*2),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                    cv2.line(
                        img,
                        (self.x+self.FRAME_WIDTH+50,                                                    int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        (self.x+self.FRAME_WIDTH+50+(int(self.PROGRESS_BAR_LEN/self.CNT_MAX))*self.CNT, int((self.y+self.FRAME_HEIGHT)/2)+40*3),
                        self.COLOR,
                        15
                    )
                cv2.putText(img, self.name+" "+str(self.percent)+"%", (40, 40), cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                cv2.putText(img,"Body TEMP",(40,40*2),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                    cv2.putText(img,str(self.BODY_TEMP),(40,40*3),cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA,self.COLOR,thickness=2)
                cv2.imshow("Image",img)
                cv2.waitKey(self.DELAY_MSEC)

                ############  戻り値は識別番号  ############
                return max_label
        #   もし顔と識別できていなかったら
        else:
            #   カウンタ初期化
            self.CNT = 0
            #   平均のラベルと平均のパーセントを初期化
            for i,_ in enumerate(self.NAME):
                self.list_label[i] = 0
                self.list_percent[i] = 0
            cv2.putText(img, "Set Face", (40*2, 40*2), cv2.FONT_HERSHEY_SIMPLEX,self.MOJI_OOKISA*2,self.COLOR,thickness=4)
            cv2.imshow("Image",img)
            cv2.waitKey(self.DELAY_MSEC)
            return -1
        pass 

    def maesyori_suiron(self,imgCV):
        # チャンネル数を１
        imgGray = cv2.cvtColor(imgCV,cv2.COLOR_BGR2GRAY)
        
        #リサイズ
        img = cv2.resize(imgGray,(self.inputSize,self.inputSize))

        # リシェイプ
        img = np.reshape(img,(1,self.inputSize,self.inputSize))

        # transpose h,c,w
        img = np.transpose(img,(1,2,0))

        # ToTensor 正規化される
        img = img.astype(np.uint8)
        mInput = transforms.ToTensor()(img)  

        #推論
        output = self.model(mInput[0])

        #予測値
        p1 = self.model.forward(mInput).exp()

        #予測値のパーセンテージ
        x1 = p1.to('cpu').detach().numpy().copy()
        x1 = x1[0]

        #予測値のうち最大値を求める
        p1 = p1.argmax().to('cpu').detach().numpy().copy()

        #予測最大パーセント
        percent = x1[p1]*100


        label = 0
        #29人の中で人工知能が最大であると識別したラベルのパーセントpercentが40%以上なら識別できているとする
        if percent >= 40:
            label = int(p1)
            
        #29人の中で人工知能が最大であると識別したラベルのパーセントpercentが40%未満のなら識別できていないとする
        else:
            label   =   29  #   ゲストの識別番号
            percent =   0   #   ゲストの場合、パーセントを0とする
            p1      =   29  #   予測値をゲストにする

        #   一定の時間内での平均値を求める。
        #   最大予測ラベルとラベルに対応するパーセントを格納する
        self.list_label[p1] += 1
        self.list_percent[p1] += percent

        self.CNT    += 1    #カウンタをインクリメント
        avg_percent = 0
        max_label   = 29
        #   もしカウンタが最大値になったら
        if self.CNT == self.CNT_MAX:
            #   カウンタ初期化
            self.CNT = 0
            #   最大予測回数
            max_value = max(self.list_label)
            #   最大のラベル
            max_index = self.list_label.index(max_value)
            #   平均パーセンテージ計算
            avg_percent = self.list_percent[max_index] / max_value

            max_label = max_index
        #   戻り値：
        #       識別ラベル      ※0番目～29番目の30人(ゲストを含む)
        #       パーセント      ※ゲストの場合、パーセントを0とする
        #       平均ラベル      ※確実な識別番号    
        #       平均パーセント  ※CNT_MAX間での平均求める、ゲストの場合は0となる
        return label,percent,max_label,avg_percent

    def imshow(self,path):
        img = cv2.imread(path)
        cv2.imshow("Thermo sensor",img)
        cv2.waitKey(self.DELAY_MSEC)


if __name__ == "__main__":
    k = Suiron()

    while True:
        k.real_time_haar()
