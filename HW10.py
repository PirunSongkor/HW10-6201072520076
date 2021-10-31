import numpy as np
import cv2

def lucas_kanade_optical_flow(video_device) :

    cap = cv2.VideoCapture(video_device)
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 900,
                        qualityLevel = 0.03,
                        minDistance = 7,
                        blockSize = 50 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (21,21),
                    maxLevel = 3,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    #Create some random colors
    colors = np.random.randint(0, 125, (900, 3)) # 500 values 3 channel

    #Take first frame and find corner
    ret, old_frame = cap.read()
    ##############################################################
    stencil = np.zeros(old_frame.shape).astype(old_frame.dtype)
    myROI = [(720,476), (530,25 ), (169, 25), (0,476)]  # (x, y)
    cv2.fillPoly(stencil, [np.array(myROI)], (255,255,255))
    old_frame = cv2.bitwise_and(old_frame, stencil)
    ##############################################################
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params) # Feature detection, Harris corner with Shi-Tomasi response function

    # Create a mask image for drawing overlay
    mask = np.zeros_like(old_frame)

    while cap.isOpened() :
        
        ret, frame = cap.read()
        ##############################################################
        stencil = np.zeros(frame.shape).astype(frame.dtype)
        myROI = [(720,476), (530,25 ), (169, 25), (0,476)]  # (x, y)
        cv2.fillPoly(stencil, [np.array(myROI)], (255,255,255))
        frame = cv2.bitwise_and(frame, stencil)
        ##############################################################
        if ret :
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #calculate optical flow 
            p1, st, err = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, p0, None, **lk_params
            )

            # Select good points
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            # Traceline drawing
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                mask = cv2.line(mask, (a, b), (c, d), colors[i].tolist(), 1)
                frame = cv2.circle(frame, (a,b), 5, colors[i].tolist(), -1)
            
            compare_img = cv2.hconcat([frame, mask])
            disp_img = cv2.add(frame, mask)
            
            cv2.imshow('frame', disp_img)

            key = cv2.waitKey(27) & 0xFF
            if key == 27 or key == ord('q') :
                break
            elif key == ord('c') : # clear mask
                mask = np.zeros_like(old_frame)
                p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
            else :
                #Update the previous frame and previous points
                old_gray = frame_gray.copy()
                p0 = good_new.reshape(-1, 1, 2)
        else :
            break

    cap.release()
    cv2.destroyAllWindows()

lucas_kanade_optical_flow('C:\Users\PC_THONGBAI\Documents\VScode\010723305-main\010723305-main\videos')

cv2.waitKey(0)
cv2.destroyAllWindows()