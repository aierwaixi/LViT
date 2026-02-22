import torch.optim
from Load_Dataset import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore")
import Config as config
from tqdm import tqdm
import os
from nets.LViT import LViT
from utils import *
import cv2


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1), tmp_3dunet.reshape(-1))
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name == "MoNuSeg":
        predict_save = cv2.pyrUp(predict_save, (448, 448))
        predict_save = cv2.resize(predict_save, (2000, 2000))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path, predict_save * 255)
    else:
        cv2.imwrite(save_path, predict_save * 255)
    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred


def _save_overlay(input_chw, gt_mask, pred_mask, save_path):
    # input_chw: [3,H,W] float in [0,1], gt/pred: [H,W] {0,1}
    img = np.transpose(input_chw, (1, 2, 0))
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    out = img.copy()
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    # Red fill for prediction
    out[pred] = (0.6 * out[pred] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)

    # Green contour for GT
    gt_u8 = (gt.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 255, 0), 1)

    cv2.imwrite(save_path, out)


def vis_and_save_heatmap(model, input_img, text, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda(), text.cuda())
    pred_class = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs,
                                                  save_path=vis_save_path + '_predict' + model_type + '.jpg')
    _save_overlay(
        input_chw=input_img[0].cpu().numpy(),
        gt_mask=labs.astype(np.uint8),
        pred_mask=predict_save.astype(np.uint8),
        save_path=vis_save_path + '_overlay' + model_type + '.jpg'
    )
    return dice_pred_tmp, iou_tmp


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session

    if config.task_name == "MoNuSeg":
        model_type = config.model_name
        model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"

    elif config.task_name == "Covid19":
        model_type = config.model_name
        if test_session:
            model_path = "./Covid19/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth.tar"
        else:
            model_path = config.model_path + "best_model-" + model_type + ".pth.tar"
    else:
        model_type = config.model_name
        model_path = config.model_path + "best_model-" + model_type + ".pth.tar"
    
    save_path = config.task_name + '/' + model_type + '/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')

    if model_type == 'LViT':
        config_vit = config.get_CTranS_config()
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)

    elif model_type == 'LViT_pretrain':
        config_vit = config.get_CTranS_config()
        model = LViT(config_vit, n_channels=config.n_channels, n_classes=config.n_labels)


    else:
        raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
       print("Let's use {0} GPUs!".format(torch.cuda.device_count()))
       model = nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_text = read_text(config.test_text_file)
    # Keep strict test filtering for MosMed official CSV split.
    use_strict_test_filter = str(getattr(config, "dataset_name", "")).lower() == "mosmed"
    test_allowed = set(test_text.keys()) if use_strict_test_filter else None
    test_dataset = ImageToImage2D(config.test_dataset, config.task_name, test_text, tf_test,
                                  image_size=config.img_size, allowed_names=test_allowed)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_num = len(test_dataset)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            # print(names)
            test_data, test_label, test_text = sampled_batch['image'], sampled_batch['label'], sampled_batch['text']
            arr = test_data.numpy()
            arr = arr.astype(np.float32())
            lab = test_label.data.numpy()
            lab2d = np.reshape(lab, (lab.shape[1], lab.shape[2]))
            vis_name = str(names)
            cv2.imwrite(os.path.join(vis_path, vis_name + "_lab.jpg"), (lab2d * 255).astype(np.uint8))
            input_img = torch.from_numpy(arr)
            dice_pred_t, iou_pred_t = vis_and_save_heatmap(model, input_img, test_text, None, lab2d,
                                                           os.path.join(vis_path, vis_name),
                                                           dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred += dice_pred_t
            iou_pred += iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print("dice_pred", dice_pred / test_num)
    print("iou_pred", iou_pred / test_num)
