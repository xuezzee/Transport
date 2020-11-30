import cv2


def create_video(step):

    img_root = './image/'
    fps = 1
    image = cv2.imread('./image/22.png')
    a = image.shape
    size = (a[0], a[1])  # 1200
    #
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videoWriter = cv2.VideoWriter('./image/test.mp4', fourcc, fps, size)  #

    # for(i=1;i<471;++i)
    for i in range(10, step+1):
        frame = cv2.imread(img_root + str(i) + '.png')
        videoWriter.write(frame)
    videoWriter.release()


if __name__ == "__main__":
    name = 'Materials Transport'
    conf = {
        'n_player': 2,  # 玩家数量
        'board_width': 11,  # 地图宽
        'board_height': 11,  # 地图高
        'n_cell_type': 5,  # 格子的种类
        'materials': 4,  # 集散点数量
        'cars': 2,  # 汽车数量
        'planes': 0,  # 飞机数量
        'barriers': 12,  # 固定障碍物数量
        'max_step': 100,  # 最大步数
        'game_name': name,  # 游戏名字
        'K': 5,  # 每个K局更新集散点物资数目
        'map_path': 'env/map.txt',  # 存放初始地图
        'cell_range': 6,  # 单格中各维度取值范围（tuple类型，只有一个int自动转为tuple）##?
        'ob_board_width': None,  # 不同智能体观察到的网格宽度（tuple类型），None表示与实际网格相同##?
        'ob_board_height': None,  # 不同智能体观察到的网格高度（tuple类型），None表示与实际网格相同##?
        'ob_cell_range': None,  # 不同智能体观察到的单格中各维度取值范围（二维tuple类型），None表示与实际网格相同##?
    }
    create_video(int(conf['max_step'])+1)