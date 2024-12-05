#!/usr/bin/env python3

# ROS için gerekli kütüphanelerin eklenmesi
import rclpy
from rclpy.node import Node
import torch
from sensor_msgs.msg import Image  # Görüntüyü aktarabilmek için kullanılan mesaj tipi Image'in eklenmesi
from cv_bridge import CvBridge, CvBridgeError  # Image mesaj tipinde gelen verinin görüntüye dönüştürülmesi veya tersi işlem için kullanılan köprünün eklenmesi
import cv2  # Görüntünün ekranda gösterilebilmesi için OpenCV'nin eklenmesi
import numpy as np
from pathlib import Path
import sys

sys.path.append('/home/uoral/utils')

from utils import (
    time_synchronized,
    select_device,
    increment_path,
    scale_coords,
    xyxy2xywh,
    non_max_suppression,
    split_for_trace_model,
    driving_area_mask,
    lane_line_mask,
    plot_one_box,
    show_seg_result,
    AverageMeter,
    letterbox
)


class CameraAndControlNode(Node):
    def __init__(self):
        # ROS düğümünün başlatılması için gerekli kodlar # Bir kez çalışır
        super().__init__("camera_n_control")

        # Arabadaki stereo kameranın bir tanesine düğümün abone edilmesi
        self.subscriber_ = self.create_subscription(
            Image,  # Düğümün bağlandığı konudaki mesajın tipi
            "/camera",  # Düğümün bağlandığı konu
            self.callback_camera_n_control,  # Her mesaj geldiğinde gidilecek fonksiyon
            10  # Gelen mesajlar için oluşturulan sıra sayısı
        )

        self.bridge = CvBridge()  # Gelen görüntü için dönüşüm köprüsünün tanımlanması

        self.get_logger().info("Control from camera has started.")  # Düğümün başladığına dair bildirim yapılması

        self.weights = '/home/uoral/Downloads/YOLOPv2-main/data/weights/yolopv2.pt'
        self.imgsz = 640
        self.stride = 32
        self.model = torch.jit.load(self.weights)
        self.device = select_device("cpu")
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA
        self.model = self.model.to(self.device)

        if self.half:
            self.model.half()  # to FP16  
        self.model.eval()

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    def callback_camera_n_control(self, msg):
        try:
            # Görüntünün alınması ve küçültülmesi
            cv_image = cv2.resize(
                self.bridge.imgmsg_to_cv2(msg, 'bgr8'),  # Abone olunan konudan gelen mesajın görüntüye dönüştürülmesi
                None,  # Küçültülecek görüntünün istenilen çözünürlüğü
                fx=1, fy=1,  # Küçültülecek görüntünün küçültme parametreleri
                interpolation=cv2.INTER_LINEAR  # Küçültülecek görüntünün hangi yöntemle küçültüleceği
            )
        except CvBridgeError as e:
            print(e)
            return

        #### KOD YAZILABİLECEK ARALIK BAŞLANGICI ####
        img0 = cv_image
        img0 = cv2.resize(img0, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        [pred, anchor_grid], seg, ll = self.model(img)

        # Apply NMS
        pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)
        for i, det in enumerate(pred):
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    plot_one_box(xyxy, img0, line_thickness=3)

                # Show segmentation results
                show_seg_result(img0, (da_seg_mask, ll_seg_mask), is_demo=True)

        cv2.imshow("img", img0)  # Görüntünün ekrana gösterilmesi
        key = cv2.waitKey(1)  # Görüntünün ekranda kalması için 1ms'lik key bekleme ve gecikme
        #### KOD YAZILABİLECEK ARALIK SONU ####


def main(args=None):
    rclpy.init(args=args)
    node = CameraAndControlNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

