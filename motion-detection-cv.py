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
