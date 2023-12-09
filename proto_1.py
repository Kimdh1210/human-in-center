from pytube import YouTube
import os
import glob
import os
import cv2
from ultralytics import YOLO
import shutil
import torch
import torchvision
from tqdm import tqdm
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 디바이스: {device}")
if device.type == "cuda":
    print(f"사용 중인 GPU: {torch.cuda.get_device_name(0)}")

##########################################################################
def download_video(video_url):
    os.makedirs('./human_in_center/save_video', exist_ok=True)
    video = YouTube(video_url)
    video.streams.filter(adaptive=True, file_extension='mp4').first().download('./human_in_center/save_video')   
    print(video.title)

##########################################################################

def extract_frames(video_path, frame_interval=15):
    # 동영상 파일 열기
    cap = cv2.VideoCapture(video_path)
    video_name = video_path.split('/')[-1]
    os.makedirs(f'./human_in_center/extracted_image/{video_name}', exist_ok=True)
    # 동영상 파일 열기에 실패한 경우 종료
    if not cap.isOpened():
        print("동영상 파일을 열 수 없습니다.")
        return

    frame_count = 0
    interval_counter = 0

    # tqdm으로 루프 감싸기
    for _ in tqdm(range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        # 프레임 읽기
        ret, frame = cap.read()

        # 프레임 읽기에 실패하거나 동영상의 끝에 도달한 경우 종료
        if not ret:
            break

        frame_count += 1

        # 일정 간격마다 이미지 파일로 저장
        if interval_counter == frame_interval:
            frame_filename = os.path.join(f'./human_in_center/extracted_image/{video_name}', f"frame_{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            interval_counter = 0

        interval_counter += 1

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 종료 후 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

def calculate_iou(box1,box2):
    x1,y1,w1,h1 =box1
    x2,y2,w2,h2 =box2

    x_intersection = max(x1,x2)
    y_intersection=max(y1,y2)

    x_union=min(x1+w1, x2+w2)
    y_union=min(y1+h1,y2+h2)

    intersection_area=max(0,x_union-x_intersection) *max(0, y_union-y_intersection)
    box1_area = w1*h1
    box2_area= w2*h2

    iou=intersection_area / float(box1_area+box2_area-intersection_area)
    print(f"iou: {iou}")

    return iou


def image_determine(boxes,cls,conf):
    deter_result=1
    if len(boxes) == 1:  # 추가된 코드 한 줄
        if int(cls[0]) == 0 and float(conf[0]) > 0.8:
            deter_result = 0
        else:
            pass
    elif len(boxes) == 2:
        if (int(cls[0]) == 0 and int(cls[1]) == 1) or (int(cls[0]) == 1 and int(cls[1]) == 0):
            deter_result = 0
        else:
            pass
    return deter_result


def func_only1_save(pt_path, video_path):
    model = YOLO(pt_path)
    video_name = video_path.split('/')[-1]
    image_path_list = glob.glob(os.path.join(f'./human_in_center/extracted_image/{video_name}', '*.png'))
    save_number = 1

    for path in image_path_list:
        results = model.predict(
            path,
            save=False,
            imgsz=640,
            conf=0.5,
            device='cpu'
        )
        

        for r in results:
            boxes = r.boxes.xyxy
            cls = r.boxes.cls
            conf = r.boxes.conf
            cls_dict = r.names
            image = cv2.imread(path)           
            det_result=image_determine(boxes,cls,conf)
            if det_result==0:
                for box, cls_number, conf in zip(boxes, cls, conf):
                    conf_number = float(conf.item())
                    cls_number_int = int(cls_number.item())
                    cls_name = cls_dict[cls_number_int]
                    x1, y1, x2, y2 = box
                    x1_int = int(x1.item())
                    x2_int = int(x2.item())
                    y1_int = int(y1.item())
                    y2_int = int(y2.item())

                    print(x1_int, y1_int, x2_int, y2_int, cls_name, conf_number)
                    color_map = {'human': (0, 255, 0), 'body': (255, 0, 0)} # human 초록 body 파랑
                    bbox_color = color_map.get(cls_name, (255, 255, 255))
                    image = cv2.rectangle(image, (x1_int, y1_int), (x2_int, y2_int), bbox_color, 2)

                os.makedirs(f'./human_in_center/all/{video_name}', exist_ok=True)
                cv2.imwrite(f'./human_in_center/all/{video_name}/all_frame_no_{save_number}.png', image)

                save_number += 1
            else:
                pass
                

if __name__ == "__main__":
    # video_url = 'https://www.youtube.com/watch?v=ThM_-ZyaqwQ'
    video_url = 'https://www.youtube.com/watch?v=AzIhz0kE8SA'
    download_video(video_url)
    
    # video_path = "./human_in_center/save_video/미국인은 동양인을 구별할 수 있을까.mp4"  # 동영상 파일 경로 설정
    video_path = "./human_in_center/save_video/삼산텍 CEO 수지의 갓벽한 3분 피칭.mp4"
    frame_interval = 15  # 프레임 간격 (0.5초에 해당하는 값)
    
    extract_frames(video_path, frame_interval)

    # pt 파일 주소
    pt_path = "best.pt"
    func_only1_save(pt_path, video_path)





