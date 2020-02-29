

import os
import cv2
import numpy as np

import ar


IMG_DIR = "input_images"
VID_DIR = "input_videos"
OUT_DIR = "./"
if not os.path.isdir(OUT_DIR):
    os.makedirs(OUT_DIR)

ds_w, ds_h = 800, 600

def test_1():
    """
    """
    print("\nTest Step 1: Extract markers in pictures.")

    input_images = ['sim_clear_scene.jpg', 'sim_noisy_scene_1.jpg',
                    'sim_noisy_scene_2.jpg']
    output_images = ['ar-1-a-1.png', 'ar-1-a-2.png', 'ar-1-a-3.png']

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):

        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))

        marker_positions = ar.find_markers(scene, template)

        for marker in marker_positions:
            mark_location(scene, marker)

        save_image(img_out, scene)


def test_2():

    print("\nTest Step 2: Draw Box around Detected Marker Zone.")

    input_images = ['ar-2-a_base.jpg', 'ar-2-b_base.jpg',
                    'ar-2-c_base.jpg', 'ar-2-d_base.jpg', 'ar-2-e_base.jpg']
    output_images = ['ar-2-a-1.png', 'ar-2-a-2.png', 'ar-2-a-3.png',
                     'ar-2-a-4.png', 'ar-2-a-5.png']

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):

        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))
        scene = cv2.resize(scene, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 

        markers = ar.find_markers(scene, template)
        image_with_box = ar.draw_box(scene, markers, 1)

        save_image(img_out, image_with_box)


def test_3():

    print("\nTest Step 3: Project a Regular Image into the Detected Zone.")

    input_images = ['ar-3-a_base.jpg', 'ar-3-b_base.jpg', 'ar-3-c_base.jpg']
    output_images = ['ar-3-a-1.png', 'ar-3-a-2.png', 'ar-3-a-3.png']

    # Advertisement image
    advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
    advert = cv2.resize(advert, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 

    src_points = ar.get_corners_list(advert)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    for img_in, img_out in zip(input_images, output_images):
        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))
        scene = cv2.resize(scene, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 

        markers = ar.find_markers(scene, template)

#         homography = ar.find_four_point_transform(src_points, markers)
        homography,_ = cv2.findHomography(np.float32(src_points), np.float32(markers))

        projected_img = ar.project_imageA_onto_imageB(advert, scene, homography)

        projected_img = ar.draw_box(projected_img, markers, 1)

        save_image(img_out, projected_img)


def test_3_mirror():

    print("\nTest Step 3M: Project a Mirror Image in the Right Perspective.")

    input_images = ['ar-2-a_base.jpg', 'ar-2-b_base.jpg','ar-2-c_base.jpg', 'ar-2-d_base.jpg', 'ar-2-e_base.jpg']
    output_images = ['ar-3-m-1.png', 'ar-3-m-2.png', 'ar-3-m-3.png', 'ar-3-m-4.png', 'ar-3-m-5.png']
    output_images_mirror = ['ar-3-m-1-mirror.png', 'ar-3-m-2-mirror.png', 'ar-3-m-3-mirror.png', 'ar-3-m-4-mirror.png', 'ar-3-m-5-mirror.png']

    # Advertisement image
    advert = cv2.imread(os.path.join(IMG_DIR, "static_subj.jpg"))
    advert = cv2.resize(advert, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 
    src_points = ar.get_corners_list(advert)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    # for img_in, img_out in zip(input_images, output_images):
    L = len(input_images)
#     L = 3
    for i in range(L):
        img_in = input_images[i]
        img_out = output_images[i]
        img_mirror = output_images_mirror[i]

        print("Input image: {}".format(img_in))

        # Open image and identify the four marker positions
        scene = cv2.imread(os.path.join(IMG_DIR, img_in))
        scene = cv2.resize(scene, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 

        markers = ar.find_markers(scene, template)

#         homography = ar.find_four_point_transform(src_points, markers)
        homography,_ = cv2.findHomography(np.float32(src_points), np.float32(markers))
        projected_img = ar.project_imageA_onto_imageB(advert, scene, homography)
        projected_img = ar.draw_box(projected_img, markers, 1)
        save_image(img_out, projected_img)
        
#         m_homology = np.linalg.inv(homography)
#         scale = np.diag([0.1,0.1,0.1])
#         m_homology = scale * m_homology * np.linalg.inv(scale)
        src_pts = np.float32(src_points)
        dst_pts = np.float32(markers)
        len_src = np.sum([np.linalg.norm([src_pts[3],src_pts[0]]),np.linalg.norm(src_pts[0:2]),np.linalg.norm(src_pts[1:3]),np.linalg.norm(src_pts[2:])])
        len_dst = np.sum([np.linalg.norm([dst_pts[3],dst_pts[0]]),np.linalg.norm(dst_pts[0:2]),np.linalg.norm(dst_pts[1:3]),np.linalg.norm(dst_pts[2:])])
        scale = 1.5*len_src / len_dst  
        dst_pts = np.mean(dst_pts,0) + (dst_pts-np.mean(dst_pts,0))*scale
#         print(markers,src_points,scale,advert.shape)
        m_homography,_ = cv2.findHomography(src_pts,dst_pts)
        m_homography = np.linalg.inv(m_homography)
        projected_img_mirror = ar.project_mirror_onto_imageB(advert, scene, homography, m_homography)
        projected_img_mirror = ar.draw_box(projected_img_mirror, markers, 1)
        save_image(img_mirror, projected_img_mirror)

def test_4():

    print("\nTest Step 4: Real-Time Marker Location by Feature Detection.")

    frame_ids = [50, 100, 200, 300]
    fps = 40

    unit_for_test_4_and_5("scene_1.mov", fps, frame_ids, "scene_1a", 1, False)
    unit_for_test_4_and_5("scene_2.mov", fps, frame_ids, "scene_2a", 1, False)
    unit_for_test_4_and_5("scene_3.mov", fps, frame_ids, "scene_3a", 1, False)
    unit_for_test_4_and_5("scene_4.mov", fps, frame_ids, "scene_4a", 1, False)
    unit_for_test_4_and_5("scene_5.mov", fps, frame_ids, "scene_5a", 1, False)
    unit_for_test_4_and_5("scene_6.mov", fps, frame_ids, "scene_6a", 1, False)
    unit_for_test_4_and_5("scene_7.mp4", fps, frame_ids, "scene_7a", 1, False)
    unit_for_test_4_and_5("scene_8.mp4", fps, frame_ids, "scene_8a", 1, False)
    unit_for_test_4_and_5("scene_9.mp4", fps, frame_ids, "scene_9a", 1, False)
    unit_for_test_4_and_5("scene_x.mp4", fps, frame_ids, "scene_xa", 1, False)

def test_5():

    print("\nTest Step 5: Real-Time Box Drawing on Videos.")

    frame_ids = [50, 100, 200, 300]
    fps = 40
    unit_for_test_4_and_5("scene_1.mov", fps, frame_ids, "scene_1b", 1, True)
    unit_for_test_4_and_5("scene_2.mov", fps, frame_ids, "scene_2b", 1, True)
    unit_for_test_4_and_5("scene_3.mov", fps, frame_ids, "scene_3b", 1, True)
    unit_for_test_4_and_5("scene_4.mov", fps, frame_ids, "scene_4b", 1, True)
    unit_for_test_4_and_5("scene_5.mov", fps, frame_ids, "scene_5b", 1, True)
    unit_for_test_4_and_5("scene_6.mov", fps, frame_ids, "scene_6b", 1, True)
    unit_for_test_4_and_5("scene_7.mp4", fps, frame_ids, "scene_7b", 1, True)
    unit_for_test_4_and_5("scene_8.mp4", fps, frame_ids, "scene_8b", 1, True)
    unit_for_test_4_and_5("scene_9.mp4", fps, frame_ids, "scene_9b", 1, True)
    unit_for_test_4_and_5("scene_x.mp4", fps, frame_ids, "scene_xb", 1, True)

def test_6():

    print("\nTest Step 6: Real-Time Mirror Projection on Videos.")

    my_video = "dyn_subj_1.mov"  
    frame_ids = [50, 100, 200, 300]
    fps = 40
    
    unit_for_test_6("scene_1.mov", fps, frame_ids, "scene_1c", 1, my_video)
    unit_for_test_6("scene_2.mov", fps, frame_ids, "scene_2c", 1, my_video)
    unit_for_test_6("scene_3.mov", fps, frame_ids, "scene_3c", 1, my_video)
    unit_for_test_6("scene_4.mov", fps, frame_ids, "scene_4c", 1, my_video)
    unit_for_test_6("scene_5.mov", fps, frame_ids, "scene_5c", 1, my_video)
    unit_for_test_6("scene_6.mov", fps, frame_ids, "scene_6c", 1, my_video)
    unit_for_test_6("scene_7.mp4", fps, frame_ids, "scene_7c", 1, my_video)
    unit_for_test_6("scene_8.mp4", fps, frame_ids, "scene_8c", 1, my_video)
    unit_for_test_6("scene_9.mp4", fps, frame_ids, "scene_9c", 1, my_video)
    unit_for_test_6("scene_x.mp4", fps, frame_ids, "scene_xc", 1, my_video)

def test_7():

    print("\nTest Step 7: Real-Time Webcam Mirror Projection on Videos.")
    
    my_video = 0  
    frame_ids = [50, 100, 200, 300]
    fps = 40

    unit_for_test_6("scene_1.mov", fps, frame_ids, "scene_1d", 1, my_video,True)
    unit_for_test_6("scene_2.mov", fps, frame_ids, "scene_2d", 1, my_video,True)
    unit_for_test_6("scene_3.mov", fps, frame_ids, "scene_3d", 1, my_video,True)
    unit_for_test_6("scene_4.mov", fps, frame_ids, "scene_4d", 1, my_video,True)
    unit_for_test_6("scene_5.mov", fps, frame_ids, "scene_5d", 1, my_video,True)
    unit_for_test_6("scene_6.mov", fps, frame_ids, "scene_6d", 1, my_video,True)
    unit_for_test_6("scene_7.mp4", fps, frame_ids, "scene_7d", 1, my_video,True)
    unit_for_test_6("scene_8.mp4", fps, frame_ids, "scene_8d", 1, my_video,True)
    unit_for_test_6("scene_9.mp4", fps, frame_ids, "scene_9d", 1, my_video,True)
    unit_for_test_6("scene_x.mp4", fps, frame_ids, "scene_xd", 1, my_video,True)


def unit_for_test_6(video_name, fps, frame_ids, output_prefix,counter_init,my_video,isWebcam=False):

    video = os.path.join(VID_DIR, video_name)
    image_gen = ar.video_frame_generator(video)

    if isWebcam:
        image_gen2 = ar.video_frame_generator(0)
    else:    
        video2 = os.path.join(VID_DIR, my_video)
        image_gen2 = ar.video_frame_generator(video2)
    
    image = image_gen.__next__()
    image = cv2.resize(image, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 

    h, w, d = image.shape

    image2 = image_gen2.__next__()
    image2 = cv2.resize(image2, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 

    h2, w2, d2 = image2.shape

    out_path = "ar_{}-{}".format(output_prefix[4:], video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

 #     advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
    src_points = ar.get_corners_list(image2)

    output_counter = counter_init

    frame_num = 1

    while image is not None and image2 is not None:

        print("Processing fame {}".format(frame_num))

        markers = ar.find_markers(image, template)
#         homography = ar.find_four_point_transform(src_points, markers)
        homography,_ = cv2.findHomography(np.float32(src_points), np.float32(markers))
#         image = ar.project_imageA_onto_imageB(image2, image, homography)

        src_pts = np.float32(src_points)
        dst_pts = np.float32(markers)
        len_src = np.sum([np.linalg.norm([src_pts[3],src_pts[0]]),np.linalg.norm(src_pts[0:2]),np.linalg.norm(src_pts[1:3]),np.linalg.norm(src_pts[2:])])
        len_dst = np.sum([np.linalg.norm([dst_pts[3],dst_pts[0]]),np.linalg.norm(dst_pts[0:2]),np.linalg.norm(dst_pts[1:3]),np.linalg.norm(dst_pts[2:])])
        scale = len_src / len_dst  
        
        dst_pts = np.mean(dst_pts,0) + (dst_pts-np.mean(dst_pts,0))*scale
#         print(markers,src_points,scale,advert.shape)
        m_homography,_ = cv2.findHomography(src_pts,dst_pts)
        m_homography = np.linalg.inv(m_homography)
        
        projected_img_mirror = ar.project_mirror_onto_imageB(image2, image, homography, m_homography)
        image = ar.draw_box(projected_img_mirror, markers, 2)
#         save_image(img_mirror, projected_img_mirror)

        frame_id = frame_ids[(output_counter - 1) % 3]

        if frame_num == frame_id:
            out_str = output_prefix + "-{}.png".format(output_counter)
            save_image(out_str, image)
            output_counter += 1

        video_out.write(image)
        
        if isWebcam:
            # Display the resulting frame 
            cv2.imshow('frame',image) 
            if cv2.waitKey(1) & 0xFF == ord('q'): break 

        image = image_gen.__next__()
        image2 = image_gen2.__next__()
        if image is not None: image = cv2.resize(image, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 
        if image2 is not None: image2 = cv2.resize(image2, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 


        frame_num += 1

    video_out.release()

def unit_for_test_4_and_5(video_name, fps, frame_ids, output_prefix,
                            counter_init, is_part5):

    video = os.path.join(VID_DIR, video_name)
    image_gen = ar.video_frame_generator(video)

    image = image_gen.__next__()
    image = cv2.resize(image, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 

    h, w, d = image.shape

    out_path = "ar_{}-{}".format(output_prefix[4:], video_name)
    video_out = mp4_video_writer(out_path, (w, h), fps)

    # Optional template image
    template = cv2.imread(os.path.join(IMG_DIR, "template.jpg"))

    if is_part5:
        advert = cv2.imread(os.path.join(IMG_DIR, "img-3-a-1.png"))
        src_points = ar.get_corners_list(advert)

    output_counter = counter_init

    frame_num = 1

    while image is not None:

        print("Processing fame {}".format(frame_num))

        markers = ar.find_markers(image, template)

        if is_part5:
#             homography = ar.find_four_point_transform(src_points, markers)
            homography,_ = cv2.findHomography(np.float32(src_points), np.float32(markers))
            image = ar.project_imageA_onto_imageB(advert, image, homography)

        else:
            
            for marker in markers:
                mark_location(image, marker)

        frame_id = frame_ids[(output_counter - 1) % 3]

        if frame_num == frame_id:
            out_str = output_prefix + "-{}.png".format(output_counter)
            save_image(out_str, image)
            output_counter += 1

        video_out.write(image)

        image = image_gen.__next__()
        if image is not None: image = cv2.resize(image, (ds_w,ds_h), interpolation = cv2.INTER_AREA) 

        frame_num += 1

    video_out.release()


def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)


def save_image(filename, image):
    """Convenient wrapper for writing images to the output directory."""
    cv2.imwrite(os.path.join(OUT_DIR, filename), image)


def mark_location(image, pt):
    """Draws a dot on the marker center and writes the location as text nearby.

    Args:
        image (numpy.array): Image to draw on
        pt (tuple): (x, y) coordinate of marker center
    """
    color = (0, 50, 255)
    cv2.circle(image, pt, 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)


if __name__ == '__main__':
    print("--- Real Time Projective Mirror Effect ---")
    # Comment out the sections you want to skip

    test_1()
    test_2()
    test_3()
    test_3_mirror()
    test_4()
    test_5()
    test_6()
    test_7()
