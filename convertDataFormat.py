import os
import cv2

sets_list = ["train", "test"]

dataDir = "/home/swei/workdir/landmark/landmark"

for set_list in sets_list:

    org_txt = os.path.join(dataDir, "%s_list.txt"%(set_list))
    dst_txt = os.path.join(os.getcwd(), "%s_list.txt"%(set_list))

    filenames_queue = open(org_txt, "r").readlines()
    dst_list = open(dst_txt, "w")
    for filename_queue in filenames_queue:
        filename = filename_queue.split('\n')[0]
        imagefilename = os.path.join(dataDir, "all_image/%s.jpg"%filename)
        imageshape = cv2.imread(imagefilename).shape
        print(imageshape)

        rect_queue = open(os.path.join(dataDir, "all_rect/%s.rct"%filename), "r").readline().split('  ')
        dst_list.write('%s %s %s %s %s'%(imagefilename, rect_queue[0], rect_queue[1], rect_queue[2], rect_queue[3]))

        pointfilename = os.path.join(dataDir, "all_point/%s.pts"%filename)
        points_queue = open(pointfilename, "r").readlines()
        for point_queue in points_queue:
            point = point_queue.split('\n')[0].split('  ')
         #   print(point)
            dst_list.write(' %s %s'%(point[0], point[1]))

        dst_list.write('\n')

    dst_list.close()
