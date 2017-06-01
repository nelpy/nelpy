""" This file currently uses opencv to select points for homography and/
or rectification. Ideally, I really want to move away from opencv, and
we can do a lot of the selection just with matplotlib or similar instead.
However, I would have to figure out how to draw those polygons with mpl.

Some of this code was mofified from https://www.cs.drexel.edu/~kon/introcompvis/

To fix the Trodes videos, use ffmpeg -r 60 -f h264 -i sine_camera_test_05-30-2017\(17_01_35\).h264 -c copy output.mp4

or ffmpeg -r 60 -f h264 -i sine_camera_test_05-30-2017\(17_01_35\).h264 -vcodec copy -an output.mp4

see e.g. https://superuser.com/questions/320045/video-encode-frame-rate-change
and https://superuser.com/questions/538829/repairing-corrupt-mp4
"""

import cv2
import numpy as np
import sys

from scipy.misc import imread

W_MAZE = True

def corr_picker_callback(event, x, y, ignored, data):
    image = data[5]
    window_name = data[6]
    n_pts_to_pick = data[7]
    mouse_pos = np.array((x, y))

    if event == cv2.EVENT_LBUTTONDOWN or event == cv2.EVENT_RBUTTONDOWN:
        data[3] = False
        data[2] = True

        if len(data[0]) < n_pts_to_pick:
            data[0].append(mouse_pos)
            data[4] = True
        else:
            # Once the user has picked points, switch to move mode.
            # Clicking and data[3] selects the nearest point and moves it to
            # the cursor.

            # Find the nearest picked point to the clicked position.
            min_dist = np.linalg.norm(mouse_pos - data[0][0])
            data[1] = 0
            for i in range(1, n_pts_to_pick):
                if np.linalg.norm(mouse_pos - data[0][i]) < min_dist:
                    min_dist = np.linalg.norm(mouse_pos - data[0][i])
                    data[1] = i

    elif event == cv2.EVENT_LBUTTONUP:
        data[3] = False
        data[2] = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if data[2]:
            data[3] = True
            data[2] = False
        elif data[3] and len(data[0]) == n_pts_to_pick:
            # If we are in move mode, update the position of the picked point
            # to follow the cursor.
            data[0][data[1]] = mouse_pos
            data[4] = True
    elif event == cv2.EVENT_RBUTTONUP:
        data[4] = True

    # This callback is called on mouse movement and presses, we only want to
    # redraw if something changed.
    if data[4]:
        pick_image = image.copy()

        # if len(data[0]) == n_pts_to_pick:
        #     cv2.polylines(pick_image, np.array([data[0]]), True, (101/255, 170/255, 211/255), thickness=2)
        if W_MAZE:
            if len(data[0]) == n_pts_to_pick:
                data_ = np.array([data[0]])
                # outer arms
                cv2.polylines(pick_image, data_[:,:5,:], False, (101/255, 170/255, 211/255), thickness=2)
                # central arm
                cv2.polylines(pick_image, data_[:,[2,5],:], False, (101/255, 170/255, 211/255), thickness=2)
        else:
            if len(data[0]) == n_pts_to_pick:
                cv2.polylines(pick_image, np.array([data[0]]), True, (101/255, 170/255, 211/255), thickness=2)
                # cv2.fillPoly(pick_image, np.array([data[0]]), (101/255, 170/255, 211/255, 0.8))

        for i, pt in enumerate(data[0]):
            cv2.circle(pick_image, tuple(pt), 3, (151/255, 207/255, 0), -1)
            cv2.putText(pick_image, str(i+1), tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 1, (151/255, 207/255, 0))

        cv2.imshow(window_name, pick_image)
        data[4] = False


def pick_corrs(images, n_pts_to_pick=4):
    data = [ [[], 0, False, False, False, image, "Image %d" % i, n_pts_to_pick]
            for i, image in enumerate(images)]

    for d in data:
        win_name = d[6]
        cv2.namedWindow(win_name)
        cv2.setMouseCallback(win_name, corr_picker_callback, d)
        cv2.startWindowThread()
        cv2.imshow(win_name, d[5])

    key = None
    while key != '\n' and key != '\r' and key != 'q':
        key = cv2.waitKey(33)
        key = chr(key & 255) if key >= 0 else None

    cv2.destroyAllWindows()

    if key == 'q':
        return None
    else:
        return [d[0] for d in data]

if __name__ == "__main__":
    """
    Example:
    ========
    >>> python homography.py ../examples/homography1.jpg 
    """

    image = imread(sys.argv[1]).astype(np.float32) / 255.
    pts = pick_corrs([image[:, :, ::-1]], n_pts_to_pick=6)
    if pts is None:
        print("You must pick some points!")
        exit(1)
    pts = np.array(pts[0])

    print(pts)