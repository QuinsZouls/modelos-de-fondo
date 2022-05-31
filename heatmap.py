import numpy as np
import cv2
import copy


def main():
    capture = cv2.VideoCapture('video.mp4')
    #Utilizamos el algoritmo para sustraer el fondo
    background_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
    length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    first_iteration_indicator = 1
    for i in range(0, length):

        ret, frame = capture.read()

        # Si es el primer frame
        if first_iteration_indicator == 1:
          #creamos el fondo
            first_frame = copy.deepcopy(frame)
            height, width = frame.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:

            filter = background_subtractor.apply(frame)  # quitamos el fondo

            threshold = 2
            maxValue = 2
            ret, th1 = cv2.threshold(filter, threshold, maxValue, cv2.THRESH_BINARY)

            # Acumulamos el movimiento
            accum_image = cv2.add(accum_image, th1)

            color_image_video = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)

            video_frame = cv2.addWeighted(frame, 0.7, color_image_video, 0.7, 0)

            cv2.imshow("Camara", video_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
    result_overlay = cv2.addWeighted(first_frame, 0.7, color_image, 0.7, 0)
    cv2.imshow("Camara", color_image)

    # cleanup
    capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()