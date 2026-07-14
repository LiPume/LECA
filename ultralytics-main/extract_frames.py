import cv2
import os
import argparse


def extract_frames(video_path, output_folder, interval_sec):
    """
    从视频中按指定的时间间隔（秒）截取帧并保存到文件夹。

    参数:
    video_path (str): 输入视频文件的路径。
    output_folder (str): 保存截取帧的文件夹路径。
    interval_sec (float): 截取帧的时间间隔（例如：1.0 表示每秒截1张）。
    """

    # 1. 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误: 视频文件未找到: {video_path}")
        return

    # 2. 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建文件夹: {output_folder}")

    # 3. 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件: {video_path}")
        return

    # 4. 获取视频的帧率 (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("警告: 无法获取FPS，将默认使用 30 FPS。")
        fps = 30

    # 5. 计算要跳过的帧数
    # 例如：interval_sec=1.0, fps=30 -> 每 30 帧保存一次
    frame_skip = int(fps * interval_sec)
    if frame_skip < 1:
        frame_skip = 1  # 至少跳过1帧

    print(f"视频路径: {video_path}")
    print(f"输出文件夹: {output_folder}")
    print(f"视频帧率 (FPS): {fps:.2f}")
    print(f"截取间隔: {interval_sec} 秒 (即每 {frame_skip} 帧截取一张)")

    frame_count = 0  # 当前读取的总帧数
    saved_count = 1  # 已保存的图片序号

    while True:
        # 6. 读取一帧
        ret, frame = cap.read()

        # 如果 ret 为 False，表示视频已结束
        if not ret:
            break

        # 7. 检查是否达到了保存条件
        if frame_count % frame_skip == 0:
            # 8. 构建保存的文件名，使用0填充（例如：0001.jpg, 0002.jpg）
            output_filename = f"{saved_count:04d}.jpg"
            output_path = os.path.join(output_folder, output_filename)

            # 9. 保存图片
            cv2.imwrite(output_path, frame)
            print(f"已保存: {output_path}")

            saved_count += 1

        frame_count += 1

    # 10. 释放资源
    cap.release()
    print("\n处理完成。")
    print(f"总共保存了 {saved_count - 1} 张图片到 {output_folder}")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 使用 argparse 来接收命令行参数
    parser = argparse.ArgumentParser(description="视频帧截取工具")

    parser.add_argument("--video_path", type=str, required=True,
                        help="输入视频文件的路径 (例如: 'my_video.mp4')")

    parser.add_argument("--output_folder", type=str, required=True,
                        help="保存图片的文件夹路径 (例如: 'hard_test_set')")

    parser.add_argument("--interval_sec", type=float, default=1.0,
                        help="截取图片的时间间隔（秒）。默认为 1.0 (每秒1张)")

    args = parser.parse_args()

    # 调用主函数
    extract_frames(args.video_path, args.output_folder, args.interval_sec)