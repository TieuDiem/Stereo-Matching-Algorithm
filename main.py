__doc__ = """
{Stereo_Matching}  Stereo Matching Code From Scratch 
"""
from pickle import TRUE
from lib import * 
from Core import core

global CWD
CWD = os.path.dirname(os.path.realpath(__file__))
def main():
    img_left  =cv2.imread(os.path.join(CWD,"images\Aloe_left_1.png"),0)
    img_right_1 =cv2.imread(os.path.join(CWD,"images\Aloe_right_1.png"),0)
    img_right_2 =cv2.imread(os.path.join(CWD,"images\Aloe_right_2.png"),0)
    img_right_3 =cv2.imread(os.path.join(CWD,"images\Aloe_right_3.png"),0)

    depth_11 = core.pixel_wise_matching(img_left,img_right_1,16,"l1",True)
    depth_12 = core.pixel_wise_matching(img_left,img_right_1,16,"l2",True)

    depth_21 = core.window_base_matching_v1(img_left,img_right_2,16,5,"l1",True)
    depth_22 = core.window_base_matching_v1(img_left,img_right_2,16,5,"l2",True)

    depth_31 = core.window_base_matching_v2(img_left,img_right_2,64,3,"cosine_similarity",True)
    depth_32 = core.window_base_matching_v2(img_left,img_right_3,64,3,"correlation_cofficient",True)

    stop  = "Wating ..."
    return None

if  __name__ =="__main__":
    main()