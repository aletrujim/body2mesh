import os
import shutil
import argparse
import subprocess

def get_first_unprocessed_png(png_folder, review_folder):
    png_files = [f for f in os.listdir(png_folder) if f.endswith('.png')]
    png_names = [os.path.splitext(f)[0] for f in png_files]

    names_to_remove = []
    for name in png_names:
        if os.path.isdir(os.path.join(review_folder, name)):
            names_to_remove.append(name)

    final_names = [name for name in png_names if name not in names_to_remove]

    first_unprocessed = f"{final_names[0]}.png" if final_names else None
    return first_unprocessed


def main(args):

    png_folder = args.input_dir
    review_folder = args.output_dir
    repo_path = args.repo_path

    print("=== BUSCANDO IMAGEN ===")
    file_name = get_first_unprocessed_png(png_folder, review_folder)

    if file_name is None:
        print("No hay imágenes pendientes")
        return

    print(f"Procesando: {file_name}")

    # paths
    image_path_ori = os.path.join(png_folder, file_name)
    sample_dir = os.path.join(repo_path, "pifuhd/sample_images")
    image_path = os.path.join(sample_dir, file_name)

    # copiar input
    shutil.copy(image_path_ori, sample_dir)

    # ===== HUMAN POSE =====
    print("=== HUMAN POSE ===")
    os.chdir(os.path.join(repo_path, "human-pose"))

    import torch
    import cv2
    import numpy as np
    from models.with_mobilenet import PoseEstimationWithMobileNet
    from modules.keypoints import extract_keypoints, group_keypoints
    from modules.load_state import load_state
    from modules.pose import Pose
    import demo

    def get_rect(net, images, height_size):
        net = net.eval()
        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts

        for image in images:
            rect_path = image.replace('.png', '_rect.txt')
            img = cv2.imread(image)
            heatmaps, pafs, scale, pad = demo.infer_fast(net, img, height_size, stride, upsample_ratio, cpu=False)

            total_keypoints_num = 0
            all_keypoints_by_type = []

            for kpt_idx in range(num_keypoints):
                total_keypoints_num += extract_keypoints(
                    heatmaps[:, :, kpt_idx],
                    all_keypoints_by_type,
                    total_keypoints_num
                )

            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

            rects = []
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue

                center = [img.shape[1]//2, img.shape[0]//2]
                radius = max(img.shape[1]//2, img.shape[0]//2)

                x1 = center[0] - radius
                y1 = center[1] - radius
                rects.append([x1, y1, 2*radius, 2*radius])

            np.savetxt(rect_path, rects, fmt='%d')

    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(
        os.path.join(repo_path, "checkpoint_iter_370000.pth"),
        map_location='cpu'
    )
    load_state(net, checkpoint)

    get_rect(net.cuda(), [image_path], 512)

    # ===== PIFUHD =====
    print("=== RUNNING PIFUHD ===")

    os.chdir(os.path.join(repo_path, "pifuhd"))

    subprocess.run([
        "python",
        "-m",
        "apps.simple_test",
        "-r", "256",
        "--use_rect",
        "-i", sample_dir
    ])


    # ===== GUARDAR RESULTADOS =====

    import glob
    import time

    print("=== BUSCANDO ARCHIVO OBJ ===")

    results_path = os.path.join(repo_path, "pifuhd", "results", "pifuhd_final", "recon")

    obj_files = []

    # Esperar hasta 60 segundos
    for i in range(12):
        obj_files = glob.glob(os.path.join(results_path, "*.obj"))
        if obj_files:
            break
        print(f"Esperando resultados... intento {i+1}")
        time.sleep(5)

    if obj_files:
        obj_file = obj_files[0]
        print("OBJ encontrado:", obj_file)

        folder_name = file_name.split(".")[0]
        save_results_path = os.path.join(review_folder, folder_name)
        os.makedirs(save_results_path, exist_ok=True)

        shutil.copy(obj_file, save_results_path)

        print("OBJ copiado a:", save_results_path)
    else:
        print("No se encontró ningún OBJ en:", results_path)

    print("=== DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--repo_path", required=True)

    args = parser.parse_args()
    main(args)
