
__doc__=="""

{1} Convert to float32 type 
{2} Initial depth map 
{3} Get disparity at [y,x] with minimum cost value



"""
from lib import *
from numpy.linalg import norm

PATH_SAVE =  os.path.dirname(os.path.realpath(__file__))
PATH_SAVE =os.path.join(PATH_SAVE,"results")
def __distance_l1(x:int, y:int):
    return abs(x-y)

def __distance_l2(x:int,y:int):
    return (x-y)**2

def __distance_cosine_similarity(x:np.ndarray, y:np.ndarray):
    x   =   x.flatten()  # Convert to one dimention vector
    y   =   y.flatten()  # Convert to one dimention vector
    return np.dot(x,y)/(norm(x)*norm(y))

def __distance_correlation_cofficient(x:np.ndarray, y:np.ndarray):
    x   =   x.flatten()  # Convert to one dimention vector
    y   =   y.flatten()  # Convert to one dimention vector

    covariance=np.dot(x- [np.mean(x)] * len(x),y -[np.mean(y)] * len(y)) / len(x)
    corre= covariance / ( np.sqrt(np.var(x)) *  np.sqrt(np.var(y)))
    return  corre


def pixel_wise_matching(image_left :np.ndarray,
                        image_right:np.ndarray,
                        disparity_range:int,
                        compute_type:str,
                        save_image=False)->None :

    image_left  =       image_left.astype(np.float32)
    image_right =       image_right.astype(np.float32)    

    height  = image_left.shape[0]
    width   = image_left.shape[1]

    depth_map = np.zeros((height,width),dtype=np.uint8)
    scale = 255/ disparity_range
    print(f'-'*100)
    if compute_type =="l1":
        distance = __distance_l1
        print(f'Compute Stereo Matching with l1 distance ...')
        path_save = os.path.join(PATH_SAVE,"pixel_wise_matching_l1.png")
    elif compute_type=="l2":
        print(f'Compute Stereo Matching with l2 distance ...')
        path_save = os.path.join(PATH_SAVE,"pixel_wise_matching_l2.png")
        distance =__distance_l2    

    print(f'Computing ...')
    for h in range(height):
        for w in range(width):
            cost_min = abs (image_left[h,w] - image_right[h,w])
            disparity =0
            for d in range(disparity_range):
                if w -d < 0:
                    cost =cost_min
                else:
                    cost =distance (image_left[h,w] ,image_right[h,w-d])
            if cost < cost_min:
                cost_min =cost
                disparity =  d     
            else:
                continue    
            depth_map[h,w] = disparity  *scale 
        printProgressBar(h+1, int(height), prefix = 'Progress:', suffix = 'Complete', length = 50)  
    if  save_image ==True:
        cv2.imwrite(path_save,depth_map)        
    print(f'Successfully ... ')        
    return depth_map 

def window_base_matching_v1(image_left :np.ndarray,
                        image_right:np.ndarray,
                        disparity_range:int,
                        k_size:int,
                        compute_type:str,
                        save_image=False):

    image_left = image_left.astype(np.float32)
    image_right  =image_right.astype(np.float32)
    
    height  = image_left.shape[0]
    width   = image_right.shape[1]

    scale = 255 /disparity_range 

    depth_map = np.zeros((height,width),dtype=np.uint8)
    print(f'-'*100)
    if compute_type =="l1":
        print(f'Compute window_base_matching with l1 distance')
        path_save = os.path.join(PATH_SAVE,"window_base_matching_l1.png")
        max_value = 255
        distance  =__distance_l1
    elif compute_type =="l2":
        print(f'Compute window_base_matching with l2 distance')
        path_save = os.path.join(PATH_SAVE,"window_base_matching_l2.png")
        max_value =255**2
        distance = __distance_l2

    kernal_half = int((k_size-1)/2)

    print(f'Computing ...')    
    for h in range(kernal_half,height - kernal_half):
        for w in range(kernal_half, width - kernal_half):
            disparity = 0
            cost_min=max_value
            for d in range(disparity_range):
                sum_cost = 0
                for u in range(-kernal_half,kernal_half +1):
                    for v in  range(-kernal_half ,kernal_half +1):
                        if w+v-d >= 0:
                            cost = distance(image_left[h + u,w +v],image_right[h+u,w +v -d])
                        else:
                            cost=max_value 
                            
                        sum_cost += cost
                if sum_cost < cost_min:
                    cost_min =sum_cost
                    disparity =d
                depth_map[h,w] = disparity*scale     
        printProgressBar(h+1, int(height), prefix = 'Progress:', suffix = 'Complete', length = 50)         
    if save_image ==  True:
        cv2.imwrite(path_save,depth_map) 
    print(f'Successfully ... ')            
    return depth_map

def window_base_matching_v2(image_left :np.ndarray,
                        image_right:np.ndarray,
                        disparity_range:int,
                        k_size:int,
                        compute_type:str,
                        save_image=False):

    image_left = image_left.astype(np.float32)
    image_right  =image_right.astype(np.float32)
    
    height  = image_left.shape[0]
    width   = image_right.shape[1]

    scale = 255 /disparity_range 

    depth_map = np.zeros((height,width),dtype=np.uint8)
    print(f'-'*100)
    if compute_type =="cosine_similarity":
        print(f'Compute Stereo Matching with cosine similarity distance')
        path_save = os.path.join(PATH_SAVE,"window_base_matching_cosine_similarity.png")
        distance  =__distance_cosine_similarity
    elif compute_type =="correlation_cofficient":
        print(f'Compute Stereo Matching with correlation cofficient distance')
        path_save = os.path.join(PATH_SAVE,"window_base_matching_correlation_cofficient.png")
        distance = __distance_correlation_cofficient

    kernel_half = int((k_size-1)/2)
    print(f'Computing ...')    
    for h in range(kernel_half,height - kernel_half):
        for w in range(kernel_half,width -kernel_half):
            disparity =0
            cost_optimal = -1
            for j in range(disparity_range):
                d = w -j
                cost =-1
                if d-kernel_half >= 0:
                    w_left = image_left [h-kernel_half:h+kernel_half+1 ,w-kernel_half:w+kernel_half+1  ]
                    w_right = image_right [h-kernel_half: (h+kernel_half)+1 ,d-kernel_half: d+kernel_half+1  ]
                    cost =distance(w_left,w_right)
                if cost > cost_optimal:
                    cost_optimal=cost
                    disparity =j
            depth_map[h,w] = disparity*scale
        printProgressBar(h+1, int(height), prefix = 'Progress:', suffix = 'Complete', length = 50)     
    if save_image ==  True:
        cv2.imwrite(path_save,depth_map)  
    print(f'Successfully ... ')      
    return depth_map