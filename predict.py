import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2, vgg
from draw_box_utils import draw_objs

label_names2 = [" ","DT_CRACK", "DT_GAP","DT_LACUNAS","DT_SUBSIDENCE","GROUND_LAMP","PIP","PIP_L","PIP_S"]
def create_model(num_classes):
    # # mobileNetv2+faster_RCNN
    # https://download.pytorch.org/models/vgg16-397923af.pth
    # 如果使用vgg16的话就下载对应预训练权重并取消下面注释，接着把mobilenetv2模型对应的两行代码注释掉
    vgg_feature = vgg(model_name="vgg16", weights_path="./backbone/vgg16.pth").features
    backbone = torch.nn.Sequential(*list(vgg_feature._modules.values())[:-1])  # 删除features中最后一个Maxpool层
    backbone.out_channels = 512

    # https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
    # backbone = MobileNetV2(weights_path="./backbone/mobilenet_v2.pth").features
    # backbone.out_channels = 1280  # 设置对应backbone输出特征矩阵的channels

    anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],  # 在哪些特征层上进行roi pooling
                                                    output_size=[7, 7],   # roi_pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    # backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    # model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    model = create_model(num_classes=9)

    # load train weights
    weights_path = "./save_weightsvgg16GRAPH/mobile-model-26.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict)
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    if not os.path.exists("./input"):
        os.makedirs("./input")
    if not os.path.exists("./input/detection-result"):
        os.makedirs("./input/detection-result")
    # load image
    with open("VOCdevkit/VOC2012/ImageSets/Main/val.txt", "r") as f:
        namelists = f.read().splitlines()
        for name in tqdm(namelists):
            name1 = name + '.jpg'
            name1 = "20324_0_264_448_8_4_processed-104193.jpg"
            original_img = Image.open(os.path.join("VOCdevkit/VOC2012/JPEGImages", name1))
            # from pil image to tensor, do not normalize image
            data_transform = transforms.Compose([transforms.ToTensor()])
            img = data_transform(original_img)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            model.eval()  # 进入验证模式
            with torch.no_grad():
                # init
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                t_start = time_synchronized()
                predictions = model(img.to(device))[0]
                t_end = time_synchronized()
                print("inference+NMS time: {}".format(t_end - t_start))
                # FPS
                t1 = time.time()
                for _ in range(100):
                    predictions = model(img.to(device))[0]                                        
                t2 = time.time()
                tact_time = (t2 - t1) / 100
                print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')
            break
                # predict_boxes = predictions["boxes"].to("cpu").numpy()
                # predict_classes = predictions["labels"].to("cpu").numpy()
                # predict_scores = predictions["scores"].to("cpu").numpy()
            
            
                # if len(predict_boxes) == 0:
                #     print("没有检测到任何目标!")
                # with open("./input/detection-result/" + name + ".txt", 'a+') as f2:
                #     for i in range(len(predict_classes)):
                #         f2.write("%s %.2f %d %d %d %d\n" %(label_names2[predict_classes[i]] ,predict_scores[i] ,predict_boxes[i][0] ,predict_boxes[i][1] ,predict_boxes[i][2] ,predict_boxes[i][3]))


                # xie ru wen jian

                # plot_img = draw_objs(original_img,
                #                     predict_boxes,
                #                     predict_classes,
                #                     predict_scores,
                #                     category_index=category_index,
                #                     box_thresh=0.5,
                #                     line_thickness=3,
                #                     font='arial.ttf',
                #                     font_size=20)
                # plt.imshow(plot_img)
                # plt.show()
                # # 保存预测的图片结果
                # plot_img.save("test_result.jpg")


if __name__ == '__main__':
    main()
