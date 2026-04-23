
from cv_bridge import CvBridge
import rospy
import numpy as np
import sys
import time
import threading
import cv2
import threadpoolctl
import os
from PIL import Image
from PIL import Image as PILImage
from sensor_msgs.msg import Image as RosImage
sys.path.append("/home/k202/gsmini_ws/src")
try:
    import gelSight_SDK.examples.gsdevice as gsdevice
except ImportError:
    raise ImportError("无法导入 gelSight_SDK,请检查系统路径 /home/ywl/gsmini_ws/src 是否存在。")

def process_and_resize_frame(frame_bgr, target_size):
    """
    对内存中的 NumPy 图像阵列进行中心裁剪和缩放
    """
    if frame_bgr is None:
        return None
        
    h, w = frame_bgr.shape[:2]
    short_edge = min(h, w)
    start_x = (w - short_edge) // 2
    start_y = (h - short_edge) // 2
    
    img_cropped = frame_bgr[start_y:start_y+short_edge, start_x:start_x+short_edge]
    img_resized = cv2.resize(img_cropped, target_size)
    
    return img_resized

class GelSightManager:
    def __init__(self, 
                 dev1_id="GelSight Mini R0B 2DAT-2LMZ", 
                 dev2_id="GelSight Mini R0B 2DPF-C3HB"):
        
        self.dev1 = gsdevice.Camera(dev1_id)
        self.dev2 = gsdevice.Camera(dev2_id)
        self.dev1.connect()
        self.dev2.connect()

        self.frame_1 = None
        self.frame_2 = None
        self.timestamp_1 = 0.0
        self.timestamp_2 = 0.0
        
        self.running = True
        
        # 为两路视频分配独立的锁
        self.lock_1 = threading.Lock()
        self.lock_2 = threading.Lock()

        # 为两路视频分配独立的读取线程，防止单一设备超时阻塞另一设备
        self.thread_1 = threading.Thread(target=self._update_loop_1, daemon=True)
        self.thread_2 = threading.Thread(target=self._update_loop_2, daemon=True)
        
        self.thread_1.start()
        self.thread_2.start()
        
        print("GelSightManager 初始化完成，双路独立读取线程已启动。")

    def _update_loop_1(self):
        while self.running:
            f1 = self.dev1.get_raw_image()
            if f1 is not None:
                with self.lock_1:
                    self.frame_1 = f1
                    self.timestamp_1 = time.time()

    def _update_loop_2(self):
        while self.running:
            f2 = self.dev2.get_raw_image()
            if f2 is not None:
                with self.lock_2:
                    self.frame_2 = f2
                    self.timestamp_2 = time.time()

    def get_tactile_frame(self):
        out_f1, out_f2 = None, None
        
        with self.lock_1:
            if self.frame_1 is not None:
                out_f1 = self.frame_1.copy()
                
        with self.lock_2:
            if self.frame_2 is not None:
                out_f2 = self.frame_2.copy()
                
        # 返回时间戳以最新的一路为准，或自行调整逻辑
        return out_f1, out_f2, max(self.timestamp_1, self.timestamp_2)

    def release(self):
        self.running = False
        self.thread_1.join()
        self.thread_2.join()

class GoproManager:
    def __init__(self, device_id=10, width=1280, height=720, fps=15):
        cv2.setNumThreads(1)
        threadpoolctl.threadpool_limits(1)

        self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频设备: {device_id}")

        self.running = True
        self.lock = threading.Lock()
        
        self._buffer_frame = None 
        self.current_frame = None
        self.kernel_timestamp = 0.0

        self.thread = threading.Thread(target=self._update_loop, daemon=True)
        self.thread.start()

    def _update_loop(self):
        while self.running:
            grabbed = self.cap.grab()
            if not grabbed:
                continue
            
            v4l2_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            timestamp_sec = v4l2_msec / 1000.0
            ret, self._buffer_frame = self.cap.retrieve()
            
            if ret and self._buffer_frame is not None:
                with self.lock:
                    self.current_frame = self._buffer_frame.copy()
                    self.kernel_timestamp = timestamp_sec

    def get_latest_frame(self):
        with self.lock:
            if self.current_frame is not None:
                # 传入 NumPy 数组，返回处理后的 NumPy 数组
                processed_frame = process_and_resize_frame(self.current_frame, (224, 224))
                return processed_frame, self.kernel_timestamp
        return None, 0.0


    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()


class RealsenseRosManager:
    def __init__(self, topic_name="/camera/color/image_raw", save_dir=""):
        try:
            rospy.init_node("data_record_node", anonymous=True, disable_signals=True)
        except rospy.exceptions.ROSException:
            pass

        self.save_dir = save_dir
        self.bridge = CvBridge()
        
        self.lock = threading.Lock()
        self.current_frame = None
        self.timestamp = 0.0

        # 订阅话题
        self.sub_rs = rospy.Subscriber(topic_name, RosImage, self._callback, queue_size=10)
        print(f"ROS RealSense 订阅节点初始化完成，监听话题: {topic_name}")

    def _callback(self, msg):
        try:
            # 绕过 cv_bridge，直接解析数据
            img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            
            # ROS 默认通常是 RGB，OpenCV 需要 BGR
            if msg.encoding == "rgb8":
                cv_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            else:
                cv_img = img_np # 如果已经是 bgr8 则直接赋值

            t = msg.header.stamp.to_sec()
            
            with self.lock:
                self.current_frame = cv_img.copy()
                self.timestamp = t
        except Exception as e:
            print(f"RealSense 图像转换解析失败: {e}")

    def get_latest_frame(self):
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy(), self.timestamp
        return None, 0.0

    def release(self):
        if hasattr(self, 'sub_rs'):
            self.sub_rs.unregister()


def main():
    save_dir = "multimodal_records"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 实例化触觉与视觉管理器
    # manager = GelSightManager()
    
    # 启用 GoproManager 以避免主线程被 read() 阻塞
    try:
        gopro_manager = GoproManager(device_id=10, width=1280, height=720, fps=30)
    except RuntimeError as e:
        print(f"警告: {e}")
        gopro_manager = None

    last_save_time = time.time()
    save_interval = 10.0
    save_counter = 0

    print("开始获取传感器与相机图像。在图像窗口按 'q' 键退出程序。")

    try:
        while True:
            # 主循环不再产生硬阻塞，直接获取最新内存拷贝
            # frame_1, frame_2, gs_timestamp = manager.get_tactile_frame()
            
            gopro_frame, gp_timestamp = None, 0.0
            if gopro_manager is not None:
                gopro_frame, gp_timestamp = gopro_manager.get_latest_frame()

            # 三路数据就绪后执行渲染
            # if frame_1 is not None and frame_2 is not None and gopro_frame is not None:
            if  gopro_frame is not None:
                # cv2.imshow("Sensor 1 - 2DDX-DPZE", frame_1)
                # cv2.imshow("Sensor 2 - 2DAT-2LMZ", frame_2)
                cv2.imshow("GoPro Camera", gopro_frame)

                current_time = time.time()
                
                if current_time - last_save_time >= save_interval:
                    time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time))
                    
                    # save_f1 = frame_1.copy()
                    # save_f2 = frame_2.copy()
                    save_gopro = gopro_frame.copy()
                    
                    # cv2.putText(save_f1, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # cv2.putText(save_f2, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(save_gopro, time_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    filename_1 = os.path.join(save_dir, f"gelsight1_{save_counter:04d}.png")
                    filename_2 = os.path.join(save_dir, f"gelsight2_{save_counter:04d}.png")
                    filename_gopro = os.path.join(save_dir, f"gopro_{save_counter:04d}.png")
                    
                    # cv2.imwrite(filename_1, save_f1)
                    # cv2.imwrite(filename_2, save_f2)
                    # cv2.imwrite(filename_gopro, save_gopro)
                    
                    print(f"[{time.strftime('%H:%M:%S')}] 已保存三路图像，序号: {save_counter:04d}")
                    
                    last_save_time = current_time
                    save_counter += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n检测到键盘中断信号。")
    finally:
        # manager.release()
        if gopro_manager is not None:
            gopro_manager.release()
        cv2.destroyAllWindows()
        print("程序已退出。")

if __name__ == "__main__":
    main()