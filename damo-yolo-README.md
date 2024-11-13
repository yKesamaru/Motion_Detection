## はじめに
首都圏を中心に強盗事件が多くなり、防犯カメラを購入される世帯が増えているそうです。

そこでこのシリーズでは、防犯カメラに必要な技術、具体的には動体検知に関する技術の解説と実装に取り組みました。

この記事は[防犯カメラの動体検知を実装する①](https://zenn.dev/ykesamaru/articles/6fa5bf4cfc38b6)の第二弾となります。

防犯カメラの動体検知を実装する①では、枯れた技術（Contour Detection with Area Filtering）である画像処理を用いて、ラズベリーパイでも動作可能な動体検知システムを構築しました。

とはいえ、最低限、動体検知機能とは主に以下の3つの機能をカバーしています。

1. 画面上のピクセルの変化をどのように感知するか「検出方法」
2. 動体検知の感度を調整可能な「感度調整」
3. 動体検知の対象となる領域（検知範囲）を指定する「マスキング」

この記事では、最新のAI技術を用いて動体検知を行います。具体的にはDAMO-YOLOについて解説・実装していきます。

![](https://raw.githubusercontent.com/yKesamaru/Motion_Detection/refs/heads/master/assets/eye-catch-2.png)

## 動作環境
```bash
(venv) user@user:~/ドキュメント/Motion_Detection$ inxi -SG --filter
System:
  Kernel: 6.8.0-48-generic x86_64 bits: 64 Desktop: GNOME 42.9
    Distro: Ubuntu 22.04.5 LTS (Jammy Jellyfish)
Graphics:
  Device-1: NVIDIA TU116 [GeForce GTX 1660 Ti] driver: nvidia v: 555.42.06
  Display: x11 server: X.Org v: 1.21.1.4 driver: X: loaded: nvidia
    unloaded: fbdev,modesetting,nouveau,vesa gpu: nvidia
    resolution: 2560x1440~60Hz
  OpenGL: renderer: NVIDIA GeForce GTX 1660 Ti/PCIe/SSE2
    v: 4.6.0 NVIDIA 555.42.06

(venv) user@user:~/ドキュメント/Motion_Detection$ python -V
Python 3.10.12
(venv) user@user:~/ドキュメント/Motion_Detection$ pip list
Package       Version
------------- ---------
beepy         1.0.7
numpy         2.1.3
opencv-python 4.10.0.84
pip           24.3.1
setuptools    75.4.0
simpleaudio   1.0.4
wheel         0.45.0
```
実装コードはラズベリーパイで動作可能なようにしてあります。

### 入力用動画ファイル
[「コラー！」で追い払う…住人は侵入者による被害をなぜ食い止められたのか 専門家が勧める“攻めの防犯”](https://www.youtube.com/watch?v=ZtQw5E3PA5c)

上記動画をinput.mp4としました。

## 主な画像処理方法

Pythonで防犯カメラの動体検知を実装する際の(AIを使わない)一般的な手法をリストアップします。

1. **背景差分法（Background Subtraction）**
   - 現在のフレームと背景フレームを比較し、動きのある部分を検出する方法。
   - 固定された背景がある場合に有効で、移動体が現れるとその差分が強調される。

2. **フレーム間差分法（Frame Differencing）**
   - 連続する2つのフレーム間の差分を取る手法。
   - 移動体のあるフレーム間でピクセルの変化を捉え、動きを検出する。

3. **光学フロー法（Optical Flow）**
   - 画像の特定の点（特徴点）の動きを追跡する手法。
   - 特に追跡したい対象がある場合に適しており、Lucas-Kanade法やFarneback法がよく使われる。

4. **輪郭抽出と面積フィルタリング（Contour Detection with Area Filtering）**
   - 動体検知後、輪郭を抽出し、その形状や面積で動体の大きさや形状を判別する手法。
   - 不要な小さな動きを除外し、特定サイズ以上の物体のみを検出する場合に有効。

5. **差分画像の二値化（Binary Thresholding of Difference Image）**
   - フレーム間の差分を二値化して、動体の輪郭を浮き彫りにする方法。
   - 背景との差を明確にするためにしきい値処理を行い、動きを特定する。

6. **ヒストグラム差分法（Histogram Difference Method）**
   - フレームのヒストグラムを計算し、連続するフレーム間でヒストグラムの差分を計測する手法。
   - 光量が一定の環境であれば、移動体があるかどうかの簡易判定に適している。

この記事では先述した3つの機能を実現するため、4番目の**輪郭抽出と面積フィルタリング（Contour Detection with Area Filtering）**を実装します。

1. 画面上のピクセルの変化をどのように感知するか「検出方法」
2. 動体検知の感度を調整可能な「感度調整」
3. 動体検知の対象となる領域（検知範囲）を指定する「マスキング」

## 実装
実装にあたり気に留めた事項をリストアップします。
- ラズベリーパイのGPIO端子を使いアクティブブザーを鳴らすようにする
  - PCで動作させる場合はbeepyを使いブザー音を再現する
  - 検出処理やブザー処理がfpsに影響しないようスレッドを分ける
- 感度調節は百分率で指定できるようにする
- マスキング範囲をtop, bottom, left, rightで指定できるようにする

```python
"""指定した領域において動体検知を行うPythonスクリプト.
Summary:
    OpenCVライブラリを使用してビデオストリームからのフレームを処理し、特定の領域に動きがあるかを検知します。
    検知領域は 'left', 'right', 'top', 'bottom' のいずれかを指定することで柔軟に調整可能です。
    また、検出する動体の面積は、フレーム全体の何パーセント以上であるかで指定できます。

Note:
    beepyのインストールには
    `sudo apt-get install libasound2-dev`
    が必要な場合があります。

Args:
    - `video_source`: 使用するビデオソースを指定（カメラのインデックスまたはビデオファイルのパス）。
    - `area_threshold_ratio`: 動体とみなす面積の閾値をフレーム全体の割合（0.0 - 1.0）で指定。
    - `detection_region`: 検知する領域を 'left', 'right', 'top', 'bottom' のいずれかで指定。

Example:
    `python motion-detection-cv.py`
"""

import concurrent.futures
import time

import beepy
import cv2
import numpy as np

# GPIOのモジュールがインポートできるかを確認
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    # GPIOモジュールが利用できない場合はbeepyを使用する
    GPIO_AVAILABLE = False


# アクティブブザーの制御関数
def buzzer_control(duration=1):
    """
    アクティブブザーを制御する関数。

    Args:
        duration (int): ブザーを鳴らす時間（秒単位）。
    """
    if GPIO_AVAILABLE:
        # GPIOが利用できる場合、ブザーを物理的に鳴らす
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)

        try:
            GPIO.output(18, GPIO.HIGH)  # ブザーを鳴らす
            time.sleep(duration)
            GPIO.output(18, GPIO.LOW)   # ブザーを止める
        finally:
            GPIO.cleanup()
    else:
        # GPIOが利用できない場合、beepyで音を鳴らす
        beepy.beep(sound="coin")


# 動体検知を行う関数
def detect_motion(video_source="video.mp4", area_threshold_ratio=0.1, detection_region='right'):
    """
    動体検知を行う関数。

    Args:
        video_source (int or str): カメラのインデックス、またはビデオファイルのパス。
        area_threshold_ratio (float): 動体とみなす最小面積の割合（0.0 - 1.0）。フレーム全体の面積に対する割合で指定。
        detection_region (str): 検知対象の領域。'left', 'right', 'top', 'bottom' のいずれかを指定。
    """
    # ビデオキャプチャの初期化
    cap = cv2.VideoCapture(video_source)

    # 背景差分法の初期化
    fgbg = cv2.createBackgroundSubtractorMOG2()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        while True:
            # フレームの取得
            ret, frame = cap.read()
            if not ret:
                break

            # フレームサイズの取得
            height, width, _ = frame.shape

            # 検知対象領域を設定
            if detection_region == 'right':
                roi = frame[:, width // 2:]
            elif detection_region == 'left':
                roi = frame[:, :width // 2]
            elif detection_region == 'top':
                roi = frame[:height // 2, :]
            elif detection_region == 'bottom':
                roi = frame[height // 2:, :]
            else:
                raise ValueError("Invalid detection_region. Use 'left', 'right', 'top', or 'bottom'.")

            # 背景差分を計算
            fgmask = fgbg.apply(roi)

            # ノイズ除去のための処理（モルフォロジー演算）
            kernel = np.ones((5, 5), np.uint8)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            # 輪郭の検出
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 検出された輪郭をループで処理
            frame_area = height * width
            for contour in contours:
                # 面積がフレーム全体の指定割合以上の輪郭のみを処理
                area = cv2.contourArea(contour)
                if area > (frame_area * area_threshold_ratio):
                    # フレームに描画
                    x, y, w, h = cv2.boundingRect(contour)
                    if detection_region == 'right':
                        cv2.rectangle(frame, (x + width // 2, y), (x + w + width // 2, y + h), (0, 255, 0), 2)
                    elif detection_region == 'left':
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    elif detection_region == 'top':
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    elif detection_region == 'bottom':
                        cv2.rectangle(frame, (x, y + height // 2), (x + w, y + h + height // 2), (0, 255, 0), 2)

                    # ブザーを鳴らす処理を別スレッドで実行
                    executor.submit(buzzer_control, duration=1)

            # フレームの表示
            cv2.imshow('Motion Detection', frame)

            # 'q'キーで終了
            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    # リソースの解放
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # カメラソース、検知面積のしきい値（割合）、検知領域を指定して動体検知を開始
    detect_motion(
        video_source="assets/input_1.mp4",
        area_threshold_ratio=0.1,         # 10%の領域を動体検知する
        detection_region='right'          # 入力フレームの右半分を処理対象とする
    )
```

### 解説
![](https://raw.githubusercontent.com/yKesamaru/Motion_Detection/refs/heads/master/assets/region.png)

ピンクに網掛けをしてある部分が感知指定領域です。
また感度調節のため、面積の10%を指定しています。

## 出力結果

<video controls width="600">
  <source src="https://github.com/yKesamaru/Motion_Detection/raw/refs/heads/master/assets/output.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

※ ブザー音が鳴ります。音量に注意してください。

## さいごに
この記事ではAIを使わず**従来の画像処理を使って動体検知を行い**、検知した場合にブザーが鳴るようにしました。

実際の防犯カメラの動画を使いました。この際、光量が変化するとこれを検知してしまいます。感度設定しているためカメラに小動物などが映り込んでも誤検知はしませんが、検知設定範囲全体の光量が変化すると、それを「動体」としてしまいます。

このような誤検知を防止するには複数の画像処理を行うか、AIを用いる必要があります。

以上です。ありがとうございました。




